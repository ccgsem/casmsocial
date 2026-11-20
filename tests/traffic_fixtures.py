"""Small road networks for the vehicle mobsim tests."""

from __future__ import annotations

import math

from casmsocial.traffic.network import TrafficNetwork

SPEED_MPS = 13.4  # ~30 mph
ORIGIN_LON, ORIGIN_LAT = -78.64, 35.78  # Raleigh
STEP_DEG = 0.01


def _haversine_m(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(math.sqrt(a))


def grid_tables(width: int, height: int, capacity_vph: float | None = None, road_type: str = "residential"):
    """Two-way grid of ``width x height`` nodes; node id = row * width + col.

    One place is snapped to every node with place id ``1000 + node id``.
    """
    nodes = []
    for row in range(height):
        for col in range(width):
            nodes.append({
                "node_id": row * width + col,
                "x": ORIGIN_LON + col * STEP_DEG,
                "y": ORIGIN_LAT + row * STEP_DEG,
            })
    by_id = {node["node_id"]: node for node in nodes}

    edges = []

    def add(a, b):
        length = _haversine_m(by_id[a]["x"], by_id[a]["y"], by_id[b]["x"], by_id[b]["y"]) * 1.05
        edge = {
            "edge_id": len(edges),
            "from_node_id": a,
            "to_node_id": b,
            "length_m": length,
            "travel_time_min": length / SPEED_MPS / 60.0,
            "mode": "drive",
            "road_type": road_type,
        }
        if capacity_vph is not None:
            edge["capacity_vph"] = capacity_vph
        edges.append(edge)

    for row in range(height):
        for col in range(width):
            here = row * width + col
            if col + 1 < width:
                add(here, here + 1)
                add(here + 1, here)
            if row + 1 < height:
                add(here, here + width)
                add(here + width, here)

    snaps = [{"place_id": 1000 + node["node_id"], "road_node_id": node["node_id"]} for node in nodes]
    return nodes, edges, snaps


def grid_network(width: int, height: int, capacity_vph: float | None = None, **kwargs) -> TrafficNetwork:
    nodes, edges, snaps = grid_tables(width, height, capacity_vph)
    return TrafficNetwork.from_tables(nodes, edges, snaps, **kwargs)


def split_columns(network: TrafficNetwork, width: int, n_ranks: int) -> dict[int, int]:
    """Place-to-rank map that cuts the grid into vertical bands."""
    band = math.ceil(width / n_ranks)
    return {
        place_id: min(n_ranks - 1, (node_id % width) // band)
        for place_id, node_id in network.road_network.place_to_node.items()
    }
