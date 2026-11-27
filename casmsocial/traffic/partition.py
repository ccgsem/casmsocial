"""Assign road nodes to MPI ranks so they line up with the place partition.

BATSim splits intersections over a geographic grid. casmsocial already
partitions places with METIS, and persons live on the rank of their current
place, so road nodes follow the places snapped to them. A trip then starts on
the rank that already holds the traveller.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from casmsocial.traffic.network import TrafficNetwork


def assign_node_ranks(
    network: TrafficNetwork,
    place_rank: Mapping[int, int],
    default_rank: int = 0,
) -> dict[int, int]:
    """Return ``node_id -> rank`` for every node in ``network``.

    1. A node with snapped places takes the most common rank among them
       (ties go to the lowest rank, so every rank computes the same map).
    2. Other nodes take the rank of the nearest labelled node by hop count
       over the undirected road graph, again breaking ties by lowest rank.
    3. Nodes not connected to any labelled node get ``default_rank``.
    """
    votes: dict[int, Counter] = {}
    for place_id, node_id in network.road_network.place_to_node.items():
        rank = place_rank.get(place_id)
        if rank is None or node_id not in network.nodes:
            continue
        votes.setdefault(node_id, Counter())[int(rank)] += 1

    node_rank: dict[int, int] = {
        node_id: min(counter.items(), key=lambda item: (-item[1], item[0]))[0] for node_id, counter in votes.items()
    }

    neighbours: dict[int, set[int]] = {node_id: set() for node_id in network.nodes}
    for edge in network.edges.values():
        neighbours[edge.from_node_id].add(edge.to_node_id)
        neighbours[edge.to_node_id].add(edge.from_node_id)

    # Level-synchronous BFS from all labelled nodes; on each level a node takes
    # the lowest rank among the neighbours that reached it, so the result does
    # not depend on dict iteration order.
    frontier = sorted(node_rank)
    while frontier:
        claims: dict[int, int] = {}
        for node_id in frontier:
            rank = node_rank[node_id]
            for other in neighbours[node_id]:
                if other in node_rank:
                    continue
                if other not in claims or rank < claims[other]:
                    claims[other] = rank
        node_rank.update(claims)
        frontier = sorted(claims)

    for node_id in network.nodes:
        node_rank.setdefault(node_id, default_rank)
    return node_rank


def edges_per_rank(network: TrafficNetwork, node_rank: Mapping[int, int]) -> Counter:
    """Count edges owned by each rank (an edge belongs to its start node's rank)."""
    return Counter(node_rank[edge.from_node_id] for edge in network.edges.values())


def boundary_edge_share(network: TrafficNetwork, node_rank: Mapping[int, int]) -> float:
    """Share of edges whose end node is on another rank; each crossing costs a migration."""
    if not network.edges:
        return 0.0
    crossing = sum(1 for edge in network.edges.values() if node_rank[edge.from_node_id] != node_rank[edge.to_node_id])
    return crossing / len(network.edges)


__all__ = ["assign_node_ranks", "boundary_edge_share", "edges_per_rank"]
