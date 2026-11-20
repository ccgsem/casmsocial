"""Event-driven, MPI-distributed vehicle mobsim for repast4py.

The design follows BATSim (Barthelemy & Carletti, smart-facility/BATSim) with
three changes for casmsocial:

* **Events, not 1 s steps.** A vehicle's link exit time is fixed with the BPR
  function when it enters the link. Ranks agree on the next global event time
  with one ``allreduce(MIN)`` and skip empty stretches (nights, quiet
  periods). BATSim has this code but commented it out in favour of 1 s steps.
* **Partition follows places.** Nodes are assigned to ranks from the METIS
  place partition (``assign_node_ranks``), not a geographic grid.
* **Vehicles, not persons, migrate.** See ``casmsocial.traffic.vehicle``.

Strategic rerouting uses BATSim's linear rule at each node:
reroute when ``x1·cos α + x2·sin α > θ``, where ``x1`` is elapsed trip time
over free-flow time so far and ``x2`` is the next edge's saturation. ``x1`` is
about 1 on an uncongested trip, so θ below ``cos α`` makes every strategic
driver reroute at every node. By default the new route also prices edges
with the loads this rank sees now (``reroute_with_live_loads``); BATSim
uses free-flow costs and only avoids the next edge.

``advance_to`` runs inside one casmsocial tick, so the outer model can keep
its 15-minute step. Per-edge mean travel times can be written out and loaded
into the next run's router (``TrafficNetwork.set_travel_time_profile``), which
is the MATSim-style feedback loop.

Accuracy note: within one sync window, a vehicle that migrates can enter its
next edge up to ``sync_window_s`` later than its true event time on the
receiving rank. Smaller windows are more exact and need more collectives.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from mpi4py import MPI
from repast4py import context as ctx

from casmsocial.traffic.network import TrafficNetwork
from casmsocial.traffic.vehicle import NO_EDGE, Vehicle


@dataclass(frozen=True)
class Arrival:
    person_id: int
    trip_id: int
    dest_node_id: int
    depart_s: float
    arrive_s: float
    free_flow_s: float
    n_reroutes: int
    n_migrations: int
    rank: int

    @property
    def travel_time_s(self) -> float:
        return self.arrive_s - self.depart_s

    @property
    def fitness(self) -> float:
        """BATSim's trip fitness: free-flow time over actual time (1.0 = no delay)."""
        travel = self.travel_time_s
        return 1.0 if travel <= 0 else self.free_flow_s / travel


@dataclass
class MobsimStats:
    spawned: int = 0
    unroutable: int = 0
    arrived: int = 0
    edge_entries: int = 0
    reroutes: int = 0
    migrations: int = 0
    sync_rounds: int = 0
    exchange_rounds: int = 0


class VehicleMobsim:
    """Moves vehicles over a ``TrafficNetwork`` across MPI ranks.

    All public methods that take ``comm`` collectives (``advance_to``,
    ``global_stats``, ``gather_travel_time_profile``) must be called by every
    rank, in the same order.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        network: TrafficNetwork,
        node_rank: Mapping[int, int],
        *,
        sync_window_s: float = 1.0,
        strategic_share: float = 0.0,
        strategy_alpha: float = math.pi / 4,
        strategy_theta: float = 1.5,
        reroute_with_live_loads: bool = True,
        record_bin_s: int = 900,
        seed: int = 0,
    ) -> None:
        if sync_window_s <= 0:
            raise ValueError("sync_window_s must be positive")
        self.comm = comm
        self.rank = comm.Get_rank()
        self.network = network
        self.node_rank = dict(node_rank)
        self.sync_window_s = float(sync_window_s)
        self.strategic_share = float(strategic_share)
        self.cos_alpha = math.cos(strategy_alpha)
        self.sin_alpha = math.sin(strategy_alpha)
        self.theta = float(strategy_theta)
        self.reroute_with_live_loads = bool(reroute_with_live_loads)
        self.record_bin_s = int(record_bin_s)
        self.seed = int(seed)

        # A dedicated context keeps vehicles out of casmsocial's place projections.
        self.context = ctx.SharedContext(comm)
        self.link_count: defaultdict[int, int] = defaultdict(int)
        self.stats = MobsimStats()
        self.now_s = 0.0

        self._vehicles: dict[tuple, Vehicle] = {}
        self._heap: list[tuple[float, int, tuple]] = []
        self._seq = 0
        self._next_local_id = 0
        self._outbound: list[tuple[tuple, int]] = []
        self._arrivals: list[Arrival] = []
        # (edge_id, bin_index) -> [traversals, total seconds]
        self._tt: defaultdict[tuple[int, int], list[float]] = defaultdict(lambda: [0, 0.0])

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
    def spawn(
        self,
        person_id: int,
        trip_id: int,
        origin_node_id: int,
        dest_node_id: int,
        depart_s: float,
        path: Iterable[int] | None = None,
    ) -> bool:
        """Start a trip. Returns False when no route exists.

        The trip can be spawned on any rank; it migrates to the origin node's
        owner on the next ``advance_to`` exchange.
        """
        route = tuple(path) if path is not None else self.network.route(origin_node_id, dest_node_id, depart_s)
        if route is None:
            self.stats.unroutable += 1
            return False

        vehicle = Vehicle(
            self._next_local_id,
            self.rank,
            person_id,
            trip_id,
            origin_node_id,
            dest_node_id,
            depart_s,
            list(route),
            strategic=self._is_strategic(person_id, trip_id),
        )
        self._next_local_id += 1
        self.context.add(vehicle)
        self._vehicles[vehicle.uid] = vehicle
        self.stats.spawned += 1

        owner = self.node_rank[origin_node_id]
        if owner != self.rank:
            self._outbound.append((vehicle.uid, owner))
        else:
            self._schedule(vehicle)
        return True

    def spawn_between_places(
        self,
        person_id: int,
        trip_id: int,
        origin_place_id: int,
        dest_place_id: int,
        depart_s: float,
    ) -> bool:
        origin = self.network.node_for_place(origin_place_id)
        dest = self.network.node_for_place(dest_place_id)
        if origin is None or dest is None:
            self.stats.unroutable += 1
            return False
        return self.spawn(person_id, trip_id, origin, dest, depart_s)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def advance_to(self, t_end_s: float) -> None:
        """Process every event before ``t_end_s`` on all ranks (collective)."""
        while True:
            local_next = self._heap[0][0] if self._heap else math.inf
            reduced = np.empty(2, dtype=np.float64)
            self.comm.Allreduce(
                np.array([local_next, -float(len(self._outbound))], dtype=np.float64),
                reduced,
                op=MPI.MIN,
            )
            global_next, any_outbound = reduced[0], reduced[1] < 0
            if any_outbound:
                self._exchange()
                continue
            if global_next >= t_end_s:
                break
            self.stats.sync_rounds += 1
            self._process_before(min(t_end_s, global_next + self.sync_window_s))
        self.now_s = max(self.now_s, t_end_s)

    def _process_before(self, horizon_s: float) -> None:
        heap = self._heap
        while heap and heap[0][0] < horizon_s:
            event_s, _, uid = heapq.heappop(heap)
            vehicle = self._vehicles.get(uid)
            if vehicle is None or vehicle.event_s != event_s:
                continue  # stale entry
            self._handle(vehicle, event_s)

    def _handle(self, vehicle: Vehicle, t: float) -> None:
        network = self.network
        if vehicle.on_edge:
            edge = network.edges[vehicle.cur_edge]
            self.link_count[edge.edge_id] -= 1
            record = self._tt[(edge.edge_id, int(vehicle.edge_enter_s) // self.record_bin_s)]
            record[0] += 1
            record[1] += t - vehicle.edge_enter_s
            vehicle.node_id = edge.to_node_id
            vehicle.cur_edge = NO_EDGE
            vehicle.event_s = t
            owner = self.node_rank[vehicle.node_id]
            if owner != self.rank:
                self._outbound.append((vehicle.uid, owner))
                return

        next_edge = vehicle.next_edge()
        if next_edge is None:
            self._arrive(vehicle, t)
            return

        if vehicle.strategic and self._wants_reroute(vehicle, next_edge, t):
            new_path = network.route(
                vehicle.node_id,
                vehicle.dest_node_id,
                t,
                avoid_edge_id=next_edge,
                live_counts=self.link_count if self.reroute_with_live_loads else None,
            )
            if new_path:
                vehicle.replace_path(new_path)
                vehicle.n_reroutes += 1
                self.stats.reroutes += 1
                next_edge = vehicle.next_edge()

        vehicle.path_pos += 1
        self.link_count[next_edge] += 1
        vehicle.cur_edge = next_edge
        vehicle.edge_enter_s = t
        vehicle.ff_s += network.edges[next_edge].free_flow_s
        vehicle.event_s = t + network.bpr_time_s(next_edge, self.link_count[next_edge])
        self.stats.edge_entries += 1
        self._schedule(vehicle)

    def _wants_reroute(self, vehicle: Vehicle, next_edge: int, t: float) -> bool:
        if len(self.network.out_edges.get(vehicle.node_id, ())) < 2:
            return False
        x2 = self.network.saturation(next_edge, self.link_count[next_edge])
        if x2 <= 0.0:
            return False
        x1 = (t - vehicle.depart_s) / vehicle.ff_s if vehicle.ff_s > 0 else 0.0
        return x1 * self.cos_alpha + x2 * self.sin_alpha - self.theta > 0.0

    def _arrive(self, vehicle: Vehicle, t: float) -> None:
        self._arrivals.append(
            Arrival(
                person_id=vehicle.person_id,
                trip_id=vehicle.trip_id,
                dest_node_id=vehicle.dest_node_id,
                depart_s=vehicle.depart_s,
                arrive_s=t,
                free_flow_s=vehicle.ff_s,
                n_reroutes=vehicle.n_reroutes,
                n_migrations=vehicle.n_migrations,
                rank=self.rank,
            )
        )
        self.stats.arrived += 1
        del self._vehicles[vehicle.uid]
        self.context.remove(vehicle)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------
    def _exchange(self) -> None:
        outbound, self._outbound = self._outbound, []
        for uid, _ in outbound:
            vehicle = self._vehicles.pop(uid)
            vehicle.n_migrations += 1
        self.stats.migrations += len(outbound)
        self.stats.exchange_rounds += 1
        self.context.move_agents(outbound, self._restore)

    def _restore(self, saved: tuple) -> Vehicle:
        vehicle = Vehicle.restore(saved)
        self._vehicles[vehicle.uid] = vehicle
        if self.node_rank[vehicle.node_id] != self.rank:
            # Spawned here for a node owned elsewhere: forward it on.
            self._outbound.append((vehicle.uid, self.node_rank[vehicle.node_id]))
        else:
            self._schedule(vehicle)
        return vehicle

    def _schedule(self, vehicle: Vehicle) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (vehicle.event_s, self._seq, vehicle.uid))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @property
    def vehicles_in_network(self) -> int:
        return len(self._vehicles)

    def drain_arrivals(self) -> list[Arrival]:
        """Return and clear trips that finished on this rank."""
        arrivals, self._arrivals = self._arrivals, []
        return arrivals

    def local_link_loads(self) -> dict[int, int]:
        return {edge_id: count for edge_id, count in self.link_count.items() if count > 0}

    def travel_time_rows(self) -> list[dict]:
        """Per-edge, per-bin mean traversal times recorded on this rank."""
        return [
            {
                "edge_id": edge_id,
                "bin_start_s": bin_index * self.record_bin_s,
                "n_traversals": int(count),
                "mean_travel_time_s": total / count,
            }
            for (edge_id, bin_index), (count, total) in sorted(self._tt.items())
            if count
        ]

    def gather_travel_time_profile(self, root: int = 0) -> list[dict] | None:
        """Collect all ranks' rows on ``root`` (collective). Edges are owned by
        one rank each, so rows never need merging."""
        gathered = self.comm.gather(self.travel_time_rows(), root=root)
        if self.rank != root:
            return None
        return [row for rows in gathered for row in rows]

    def global_stats(self) -> dict[str, int]:
        """Totals over ranks (collective). Round counters are the same
        collective rounds on every rank, so they take the max, not the sum."""
        local = asdict(self.stats)
        keys = sorted(local)
        values = np.array([local[k] for k in keys], dtype=np.int64)
        totals = np.empty(len(keys), dtype=np.int64)
        maxima = np.empty(len(keys), dtype=np.int64)
        self.comm.Allreduce(values, totals, op=MPI.SUM)
        self.comm.Allreduce(values, maxima, op=MPI.MAX)
        result = {key: int(maxima[i] if key in _ROUND_COUNTERS else totals[i]) for i, key in enumerate(keys)}
        result["in_network"] = int(self.comm.allreduce(len(self._vehicles), op=MPI.SUM))
        return result

    # ------------------------------------------------------------------
    def _is_strategic(self, person_id: int, trip_id: int) -> bool:
        if self.strategic_share <= 0.0:
            return False
        if self.strategic_share >= 1.0:
            return True
        return _unit_hash(self.seed, person_id, trip_id) < self.strategic_share


_ROUND_COUNTERS = frozenset({"sync_rounds", "exchange_rounds"})
_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _unit_hash(seed: int, person_id: int, trip_id: int) -> float:
    """Rank-count-independent uniform draw in [0, 1) for a (person, trip)."""
    mixed = _splitmix64(_splitmix64(_splitmix64(seed & _MASK64) ^ (person_id & _MASK64)) ^ (trip_id & _MASK64))
    return (mixed >> 11) / float(1 << 53)
