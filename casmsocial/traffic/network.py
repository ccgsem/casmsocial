"""Congestion-aware view of a ``RoadNetwork`` for the vehicle mobsim.

``RoadNetwork`` stores free-flow travel times in minutes and routes between
places. The mobsim needs edge-level paths, per-edge capacity, a BPR link
performance function, and routing that can avoid a congested edge or use a
time-dependent travel-time profile from an earlier run.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from casmsocial.road_network import RoadEdge, RoadNetwork

# Rough per-direction capacities (vehicles / hour) by OSM highway class. These
# are placeholders until edge tables carry lanes or calibrated capacities.
DEFAULT_CAPACITY_VPH: dict[str, float] = {
    "motorway": 4000.0,
    "motorway_link": 1500.0,
    "trunk": 3000.0,
    "trunk_link": 1200.0,
    "primary": 1800.0,
    "primary_link": 1000.0,
    "secondary": 1200.0,
    "secondary_link": 800.0,
    "tertiary": 900.0,
    "tertiary_link": 600.0,
    "residential": 600.0,
    "unclassified": 600.0,
    "living_street": 300.0,
    "service": 300.0,
}
FALLBACK_CAPACITY_VPH = 600.0
MIN_EDGE_TIME_S = 1.0
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class TrafficEdge:
    """Edge attributes the mobsim reads on every link entry."""

    edge_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    free_flow_s: float
    capacity_vph: float

    @property
    def occupancy_capacity(self) -> float:
        """Vehicles on the edge when it carries capacity flow at free-flow speed."""
        return max(1.0, self.capacity_vph * self.free_flow_s / 3600.0)


class TrafficNetwork:
    """Edge-indexed road network with BPR travel times and A* routing.

    Every rank holds a full copy, as in BATSim. Only the rank that owns an
    edge's start node changes that edge's vehicle count.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        mode: str = "drive",
        capacity_vph: Mapping[int, float] | None = None,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        link_model: str = "implied_flow",
    ) -> None:
        if link_model not in ("implied_flow", "batsim"):
            raise ValueError(f"unknown link_model {link_model!r}")
        self.link_model = link_model
        self.road_network = road_network
        self.mode = mode
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.nodes = road_network.nodes
        self.edges: dict[int, TrafficEdge] = {}
        self.out_edges: dict[int, list[int]] = {node_id: [] for node_id in road_network.nodes}
        overrides = capacity_vph or {}

        for node_edges in road_network.adjacency.values():
            for edge in node_edges:
                if edge.mode != mode:
                    continue
                if edge.edge_id in self.edges:
                    raise ValueError(f"duplicate edge_id {edge.edge_id} in road network")
                self.edges[edge.edge_id] = TrafficEdge(
                    edge_id=edge.edge_id,
                    from_node_id=edge.from_node_id,
                    to_node_id=edge.to_node_id,
                    length_m=edge.length_m,
                    free_flow_s=max(MIN_EDGE_TIME_S, edge.travel_time_min * 60.0),
                    capacity_vph=float(overrides.get(edge.edge_id, _default_capacity(edge))),
                )
                self.out_edges.setdefault(edge.from_node_id, []).append(edge.edge_id)

        self._geographic = _looks_geographic(self.nodes.values())
        self._max_speed_mps = max(
            (edge.length_m / edge.free_flow_s for edge in self.edges.values() if edge.length_m > 0),
            default=1.0,
        )
        # A* needs straight-line distance <= road length on every edge; if the
        # coordinates don't satisfy that (unknown units, toy networks), fall
        # back to Dijkstra rather than return non-optimal paths.
        self._use_heuristic = all(
            self._straight_line_m(edge.from_node_id, edge.to_node_id) <= edge.length_m * 1.001 + 1e-6
            for edge in self.edges.values()
        )
        # (edge_id, bin_index) -> mean traversal seconds from a previous run.
        self._profile: dict[tuple[int, int], float] = {}
        self._profile_bin_s = 900

    @classmethod
    def from_tables(
        cls,
        nodes: list[dict],
        edges: list[dict],
        place_snaps: list[dict],
        mode: str = "drive",
        **kwargs,
    ) -> TrafficNetwork:
        """Build from the same records ``RoadNetwork.from_tables`` takes.

        An optional ``capacity_vph`` column on edge records overrides the
        road-type default.
        """
        capacity = {
            int(edge["edge_id"]): float(edge["capacity_vph"]) for edge in edges if edge.get("capacity_vph") is not None
        }
        road_network = RoadNetwork.from_tables(nodes, edges, place_snaps)
        return cls(road_network, mode=mode, capacity_vph=capacity, **kwargs)

    # ------------------------------------------------------------------
    # Link performance
    # ------------------------------------------------------------------
    def bpr_time_s(self, edge_id: int, vehicles_on_edge: int) -> float:
        """BPR traversal time for an edge holding ``vehicles_on_edge`` vehicles.

        ``link_model="implied_flow"`` (default) reads the count as a flow:
        ``n`` vehicles each spending ``t`` seconds on the edge imply a flow of
        ``n / t``. It solves ``t = ff·(1 + α·(n / (c·t))^β)`` for ``t``, which
        keeps the implied flow near capacity as queues build.

        ``link_model="batsim"`` reproduces BATSim's ``Link::timeOnLink``:
        ``ff·(1 + α·(n / c_occ)^β)`` with the raw count, which grows with
        ``n^4`` and gives hour-long link times once a few dozen cars queue.
        """
        edge = self.edges[edge_id]
        if vehicles_on_edge <= 0:
            return edge.free_flow_s
        if self.link_model == "batsim":
            ratio = vehicles_on_edge / edge.occupancy_capacity
            return edge.free_flow_s * (1.0 + self.bpr_alpha * ratio**self.bpr_beta)
        return _implied_flow_bpr(
            edge.free_flow_s,
            vehicles_on_edge / (edge.capacity_vph / 3600.0),
            self.bpr_alpha,
            self.bpr_beta,
        )

    def saturation(self, edge_id: int, vehicles_on_edge: int) -> float:
        return vehicles_on_edge / self.edges[edge_id].occupancy_capacity

    # ------------------------------------------------------------------
    # Time-dependent costs (MATSim-style feedback between runs)
    # ------------------------------------------------------------------
    def set_travel_time_profile(self, rows: Iterable[Mapping], bin_s: int = 900) -> None:
        """Load mean traversal times per edge and time bin from a previous run.

        Rows need ``edge_id``, ``bin_start_s`` and ``mean_travel_time_s``.
        """
        self._profile_bin_s = int(bin_s)
        self._profile = {
            (int(row["edge_id"]), int(row["bin_start_s"]) // self._profile_bin_s): float(row["mean_travel_time_s"])
            for row in rows
        }

    def expected_time_s(self, edge_id: int, at_s: float) -> float:
        if self._profile:
            profiled = self._profile.get((edge_id, int(at_s) // self._profile_bin_s))
            if profiled is not None:
                return profiled
        return self.edges[edge_id].free_flow_s

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def free_flow_path_time_s(self, path: Iterable[int]) -> float:
        return sum(self.edges[edge_id].free_flow_s for edge_id in path)

    def node_for_place(self, place_id: int) -> int | None:
        return self.road_network.nearest_node_for_place(place_id)

    def route(
        self,
        origin_node_id: int,
        destination_node_id: int,
        depart_s: float = 0.0,
        avoid_edge_id: int | None = None,
        live_counts: Mapping[int, int] | None = None,
    ) -> tuple[int, ...] | None:
        """A* fastest path as a tuple of edge ids, or None when unreachable.

        Edge costs come from the travel-time profile at the departure time
        (free flow when no profile is loaded). ``avoid_edge_id`` is excluded,
        which is how BATSim's strategic agents route around a congested link.
        ``live_counts`` (edge id -> vehicles on it now) replaces the expected
        time with the current BPR time on those edges, so a reroute can see
        congestion the rank knows about. BATSim reroutes on free-flow costs.
        """
        if origin_node_id == destination_node_id:
            return ()
        if origin_node_id not in self.nodes or destination_node_id not in self.nodes:
            return None

        target = self.nodes[destination_node_id]
        best: dict[int, float] = {origin_node_id: 0.0}
        via_edge: dict[int, int] = {}
        queue: list[tuple[float, float, int]] = [(self._heuristic_s(origin_node_id, target), 0.0, origin_node_id)]

        while queue:
            _, cost, node_id = heapq.heappop(queue)
            if cost > best.get(node_id, math.inf):
                continue
            if node_id == destination_node_id:
                return self._edges_back(via_edge, origin_node_id, destination_node_id)
            for edge_id in self.out_edges.get(node_id, ()):
                if edge_id == avoid_edge_id:
                    continue
                edge = self.edges[edge_id]
                load = live_counts.get(edge_id, 0) if live_counts else 0
                if load > 0:
                    edge_time = max(self.bpr_time_s(edge_id, load), self.expected_time_s(edge_id, depart_s + cost))
                else:
                    edge_time = self.expected_time_s(edge_id, depart_s + cost)
                next_cost = cost + edge_time
                if next_cost >= best.get(edge.to_node_id, math.inf):
                    continue
                best[edge.to_node_id] = next_cost
                via_edge[edge.to_node_id] = edge_id
                heapq.heappush(
                    queue,
                    (next_cost + self._heuristic_s(edge.to_node_id, target), next_cost, edge.to_node_id),
                )
        return None

    def _edges_back(self, via_edge: dict[int, int], origin: int, destination: int) -> tuple[int, ...]:
        path: list[int] = []
        node_id = destination
        while node_id != origin:
            edge_id = via_edge[node_id]
            path.append(edge_id)
            node_id = self.edges[edge_id].from_node_id
        path.reverse()
        return tuple(path)

    @property
    def uses_astar(self) -> bool:
        return self._use_heuristic

    def _straight_line_m(self, node_a: int, node_b: int) -> float:
        a, b = self.nodes[node_a], self.nodes[node_b]
        if self._geographic:
            return _haversine_m(a.x, a.y, b.x, b.y)
        return math.hypot(a.x - b.x, a.y - b.y)

    def _heuristic_s(self, node_id: int, target) -> float:
        # Admissible while profiled times stay >= free flow, which BPR guarantees.
        if not self._use_heuristic:
            return 0.0
        return self._straight_line_m(node_id, target.node_id) / self._max_speed_mps


def _implied_flow_bpr(free_flow_s: float, load_s: float, alpha: float, beta: float) -> float:
    """Solve ``t = ff + K·t^(-β)`` with ``K = ff·α·load^β`` (``load = n / c`` in seconds).

    ``g(t) = t - ff - K·t^(-β)`` is increasing and concave, so Newton's method
    from below converges monotonically. Both ``ff`` and ``K^(1/(β+1))`` are
    <= the root, so start at the larger of the two.
    """
    k = free_flow_s * alpha * load_s**beta
    t = max(free_flow_s, k ** (1.0 / (beta + 1.0)))
    for _ in range(50):
        g = t - free_flow_s - k * t ** (-beta)
        if abs(g) <= 1e-9 * t:
            break
        t -= g / (1.0 + beta * k * t ** (-beta - 1.0))
    return t


def _default_capacity(edge: RoadEdge) -> float:
    return DEFAULT_CAPACITY_VPH.get(str(edge.road_type or ""), FALLBACK_CAPACITY_VPH)


def _looks_geographic(nodes: Iterable) -> bool:
    return all(-180.0 <= node.x <= 180.0 and -90.0 <= node.y <= 90.0 for node in nodes)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))
