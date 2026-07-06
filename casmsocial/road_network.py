from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoadNode:
    node_id: int
    x: float
    y: float


@dataclass(frozen=True)
class RoadEdge:
    edge_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    travel_time_min: float
    mode: str = "drive"
    road_type: str | None = None


@dataclass(frozen=True)
class RoadRoute:
    origin_place_id: int
    destination_place_id: int
    origin_node_id: int
    destination_node_id: int
    distance_m: float
    travel_time_min: int
    node_ids: tuple[int, ...] = ()


@dataclass
class RoadNetwork:
    nodes: dict[int, RoadNode]
    adjacency: dict[int, list[RoadEdge]]
    place_to_node: dict[int, int]
    route_cache: dict[tuple[int, int, str], RoadRoute | None] = field(default_factory=dict)

    @classmethod
    def from_tables(
        cls,
        nodes: list[dict],
        edges: list[dict],
        place_snaps: list[dict],
    ) -> RoadNetwork:
        node_map = {
            int(node["node_id"]): RoadNode(
                node_id=int(node["node_id"]),
                x=float(node["x"]),
                y=float(node["y"]),
            )
            for node in nodes
        }

        adjacency: dict[int, list[RoadEdge]] = {node_id: [] for node_id in node_map}
        for edge in edges:
            road_edge = RoadEdge(
                edge_id=int(edge["edge_id"]),
                from_node_id=int(edge["from_node_id"]),
                to_node_id=int(edge["to_node_id"]),
                length_m=float(edge["length_m"]),
                travel_time_min=float(edge["travel_time_min"]),
                mode=str(edge.get("mode", "drive")),
                road_type=edge.get("road_type"),
            )
            adjacency.setdefault(road_edge.from_node_id, []).append(road_edge)

        place_to_node = {int(place_snap["place_id"]): int(place_snap["road_node_id"]) for place_snap in place_snaps}

        return cls(nodes=node_map, adjacency=adjacency, place_to_node=place_to_node)

    def nearest_node_for_place(self, place_id: int) -> int | None:
        return self.place_to_node.get(place_id)

    def route_between_places(
        self,
        origin_place_id: int,
        destination_place_id: int,
        mode: str = "drive",
    ) -> RoadRoute | None:
        cache_key = (origin_place_id, destination_place_id, mode)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        origin_node_id = self.nearest_node_for_place(origin_place_id)
        destination_node_id = self.nearest_node_for_place(destination_place_id)
        if origin_node_id is None or destination_node_id is None:
            self.route_cache[cache_key] = None
            return None

        shortest_path = self.shortest_path(origin_node_id, destination_node_id, mode)
        if shortest_path is None:
            self.route_cache[cache_key] = None
            return None

        distance_m, travel_time_min, node_ids = shortest_path
        route = RoadRoute(
            origin_place_id=origin_place_id,
            destination_place_id=destination_place_id,
            origin_node_id=origin_node_id,
            destination_node_id=destination_node_id,
            distance_m=distance_m,
            travel_time_min=max(1, math.ceil(travel_time_min)),
            node_ids=node_ids,
        )
        self.route_cache[cache_key] = route
        return route

    def shortest_path(
        self,
        origin_node_id: int,
        destination_node_id: int,
        mode: str = "drive",
    ) -> tuple[float, float, tuple[int, ...]] | None:
        if origin_node_id == destination_node_id:
            return (0.0, 0.0, (origin_node_id,))
        if origin_node_id not in self.nodes or destination_node_id not in self.nodes:
            return None

        queue: list[tuple[float, float, int]] = [(0.0, 0.0, origin_node_id)]
        best_time: dict[int, float] = {origin_node_id: 0.0}
        best_distance: dict[int, float] = {origin_node_id: 0.0}
        previous: dict[int, int] = {}

        while queue:
            current_time, current_distance, node_id = heapq.heappop(queue)
            if current_time > best_time.get(node_id, math.inf):
                continue

            if node_id == destination_node_id:
                return (
                    current_distance,
                    current_time,
                    self._reconstruct_path(previous, origin_node_id, destination_node_id),
                )

            for edge in self.adjacency.get(node_id, []):
                if edge.mode != mode:
                    continue

                next_node_id = edge.to_node_id
                next_time = current_time + edge.travel_time_min
                next_distance = current_distance + edge.length_m
                previous_best = best_time.get(next_node_id, math.inf)

                if next_time >= previous_best:
                    continue

                best_time[next_node_id] = next_time
                best_distance[next_node_id] = next_distance
                previous[next_node_id] = node_id
                heapq.heappush(queue, (next_time, next_distance, next_node_id))

        return None

    def _reconstruct_path(
        self,
        previous: dict[int, int],
        origin_node_id: int,
        destination_node_id: int,
    ) -> tuple[int, ...]:
        path = [destination_node_id]
        current_node_id = destination_node_id

        while current_node_id != origin_node_id:
            current_node_id = previous[current_node_id]
            path.append(current_node_id)

        path.reverse()
        return tuple(path)
