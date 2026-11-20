"""Single-rank tests for the prototype vehicle mobsim (casmsocial.traffic)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from casmsocial.traffic import (  # noqa: E402
    TrafficNetwork,
    Vehicle,
    VehicleMobsim,
    assign_node_ranks,
    boundary_edge_share,
)
from casmsocial.traffic.mobsim import _unit_hash  # noqa: E402
from tests.traffic_fixtures import grid_network, grid_tables, split_columns  # noqa: E402

COMM = MPI.COMM_SELF


def single_rank(network):
    return {node_id: 0 for node_id in network.nodes}


# ----------------------------------------------------------------------
# TrafficNetwork
# ----------------------------------------------------------------------
def test_batsim_link_model_matches_time_on_link():
    network = grid_network(2, 1, capacity_vph=600, link_model="batsim")
    edge_id = network.out_edges[0][0]
    free = network.edges[edge_id].free_flow_s
    occ = network.edges[edge_id].occupancy_capacity

    assert network.bpr_time_s(edge_id, 0) == pytest.approx(free)
    assert network.bpr_time_s(edge_id, 7) == pytest.approx(free * (1 + 0.15 * (7 / occ) ** 4))


@pytest.mark.parametrize("vehicles", [1, 5, 50, 500, 5000])
def test_implied_flow_bpr_solves_its_fixed_point(vehicles):
    network = grid_network(2, 1, capacity_vph=600)
    edge_id = network.out_edges[0][0]
    free = network.edges[edge_id].free_flow_s
    capacity_vps = 600 / 3600

    t = network.bpr_time_s(edge_id, vehicles)
    flow_ratio = vehicles / t / capacity_vps
    assert t >= free
    assert t == pytest.approx(free * (1 + 0.15 * flow_ratio**4), rel=1e-6)


def test_implied_flow_bpr_grows_with_load_but_far_slower_than_batsim():
    implied = grid_network(2, 1, capacity_vph=600)
    batsim = grid_network(2, 1, capacity_vph=600, link_model="batsim")
    edge_id = implied.out_edges[0][0]

    assert implied.bpr_time_s(edge_id, 50) > implied.bpr_time_s(edge_id, 5) > implied.bpr_time_s(edge_id, 0)
    assert implied.bpr_time_s(edge_id, 50) < batsim.bpr_time_s(edge_id, 50) / 10


def test_unknown_link_model_rejected():
    with pytest.raises(ValueError):
        grid_network(2, 1, link_model="queue")


def test_capacity_defaults_by_road_type_and_column_override():
    nodes, edges, snaps = grid_tables(2, 1)
    edges[0]["road_type"] = "primary"
    edges[1]["capacity_vph"] = 123.0
    network = TrafficNetwork.from_tables(nodes, edges, snaps)

    assert network.edges[0].capacity_vph == 1800.0
    assert network.edges[1].capacity_vph == 123.0


def test_astar_matches_dijkstra_on_geographic_grid():
    network = grid_network(6, 5)
    assert network.uses_astar

    for origin, dest in [(0, 29), (5, 24), (12, 3)]:
        path = network.route(origin, dest)
        expected = network.road_network.shortest_path(origin, dest)
        assert path is not None and expected is not None
        assert network.free_flow_path_time_s(path) == pytest.approx(expected[1] * 60.0)
        assert network.edges[path[0]].from_node_id == origin
        assert network.edges[path[-1]].to_node_id == dest


def test_heuristic_disabled_when_coordinates_disagree_with_lengths():
    nodes, edges, snaps = grid_tables(3, 1)
    for edge in edges:
        edge["length_m"] = 1.0  # far shorter than the coordinates imply
    network = TrafficNetwork.from_tables(nodes, edges, snaps)

    assert not network.uses_astar
    assert network.route(0, 2) is not None


def test_route_avoids_edge_and_uses_profile():
    network = grid_network(3, 3)
    direct = network.route(0, 2)
    avoided = network.route(0, 2, avoid_edge_id=direct[0])
    assert direct[0] not in avoided

    # Make the first leg of the direct route very slow in the 0-900 s bin.
    network.set_travel_time_profile([{"edge_id": direct[0], "bin_start_s": 0, "mean_travel_time_s": 10_000.0}])
    assert direct[0] not in network.route(0, 2, depart_s=10.0)
    assert network.route(0, 2, depart_s=1000.0) == direct


# ----------------------------------------------------------------------
# Partition
# ----------------------------------------------------------------------
def test_assign_node_ranks_votes_then_spreads_deterministically():
    network = grid_network(4, 1)
    # Only the end nodes have places with ranks.
    place_rank = {1000: 0, 1003: 1}
    node_rank = assign_node_ranks(network, place_rank)

    assert node_rank == {0: 0, 1: 0, 2: 1, 3: 1}


def test_assign_node_ranks_breaks_vote_ties_to_lowest_rank():
    network = grid_network(2, 1)
    network.road_network.place_to_node[5000] = 0  # second place on node 0
    node_rank = assign_node_ranks(network, {1000: 1, 5000: 0, 1001: 1})
    assert node_rank[0] == 0


def test_boundary_share_counts_crossing_edges():
    network = grid_network(4, 1)
    node_rank = assign_node_ranks(network, split_columns(network, 4, 2))
    # 6 directed edges, 2 cross the middle.
    assert boundary_edge_share(network, node_rank) == pytest.approx(2 / 6)


# ----------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------
def test_vehicle_save_restore_round_trip():
    vehicle = Vehicle(7, 1, 42, 3, 10, 20, 900.0, [1, 2, 3], path_pos=1, cur_edge=1, ff_s=12.5, strategic=True)
    clone = Vehicle.restore(vehicle.save())

    assert clone.uid == vehicle.uid
    assert clone.remaining_path == [2, 3]
    assert (clone.person_id, clone.trip_id, clone.cur_edge, clone.ff_s, clone.strategic) == (42, 3, 1, 12.5, True)


# ----------------------------------------------------------------------
# Mobsim
# ----------------------------------------------------------------------
def test_single_vehicle_travels_at_near_free_flow():
    network = grid_network(5, 1, capacity_vph=100_000)
    mobsim = VehicleMobsim(COMM, network, single_rank(network))
    assert mobsim.spawn(person_id=1, trip_id=0, origin_node_id=0, dest_node_id=4, depart_s=60.0)

    mobsim.advance_to(3600.0)
    (arrival,) = mobsim.drain_arrivals()

    expected = network.free_flow_path_time_s(network.route(0, 4))
    assert arrival.depart_s == 60.0
    assert arrival.travel_time_s == pytest.approx(expected, rel=1e-6)
    assert arrival.fitness == pytest.approx(1.0, rel=1e-6)
    assert mobsim.vehicles_in_network == 0


def test_vehicles_wait_until_departure_across_ticks():
    network = grid_network(3, 1)
    mobsim = VehicleMobsim(COMM, network, single_rank(network))
    mobsim.spawn(1, 0, 0, 2, depart_s=2000.0)

    mobsim.advance_to(900.0)
    mobsim.advance_to(1800.0)
    assert mobsim.drain_arrivals() == []
    assert mobsim.stats.edge_entries == 0

    mobsim.advance_to(5400.0)
    assert len(mobsim.drain_arrivals()) == 1


def test_chunked_advance_matches_single_advance():
    def run(chunks):
        network = grid_network(4, 4, capacity_vph=300)
        mobsim = VehicleMobsim(COMM, network, single_rank(network))
        for i in range(60):
            mobsim.spawn(i, 0, i % 4, 15 - (i % 3), depart_s=i * 5.0)
        for t in chunks:
            mobsim.advance_to(t)
        return sorted((a.person_id, a.arrive_s) for a in mobsim.drain_arrivals())

    one = run([7200.0])
    many = run([900.0 * k for k in range(1, 9)])
    assert len(one) == 60
    assert one == pytest.approx(many)


def test_congestion_slows_a_bottleneck():
    def mean_time(n):
        network = grid_network(4, 1, capacity_vph=200)
        mobsim = VehicleMobsim(COMM, network, single_rank(network))
        for i in range(n):
            mobsim.spawn(i, 0, 0, 3, depart_s=float(i))
        mobsim.advance_to(36_000.0)
        arrivals = mobsim.drain_arrivals()
        assert len(arrivals) == n
        return sum(a.travel_time_s for a in arrivals) / n

    assert mean_time(80) > 1.5 * mean_time(1)


def _bottom_row_rush(share):
    network = grid_network(4, 3, capacity_vph=150)
    mobsim = VehicleMobsim(COMM, network, single_rank(network), strategic_share=share)
    # Everyone heads along the bottom row; rows above offer detours.
    for i in range(120):
        mobsim.spawn(i, 0, 0, 3, depart_s=float(i))
    mobsim.advance_to(36_000.0)
    arrivals = mobsim.drain_arrivals()
    assert len(arrivals) == 120
    return mobsim.stats, sum(a.travel_time_s for a in arrivals) / len(arrivals)


def test_some_strategic_agents_reroute_around_congestion():
    stats_none, mean_none = _bottom_row_rush(0.0)
    stats_some, mean_some = _bottom_row_rush(0.3)

    assert stats_none.reroutes == 0
    assert stats_some.reroutes > 0
    assert mean_some < 0.95 * mean_none


def test_all_strategic_agents_herd_onto_the_same_detours():
    """With everyone strategic, drivers chase the same detours (as BATSim's
    model predicts), so trips get longer and cover many more links."""
    stats_some, mean_some = _bottom_row_rush(0.3)
    stats_all, mean_all = _bottom_row_rush(1.0)

    assert mean_all > mean_some
    assert stats_all.edge_entries > 2 * stats_some.edge_entries


def test_travel_time_profile_feeds_next_run():
    network = grid_network(4, 1, capacity_vph=200)
    mobsim = VehicleMobsim(COMM, network, single_rank(network), record_bin_s=900)
    for i in range(50):
        mobsim.spawn(i, 0, 0, 3, depart_s=float(i))
    mobsim.advance_to(36_000.0)

    rows = mobsim.gather_travel_time_profile()
    assert rows and all(row["bin_start_s"] % 900 == 0 for row in rows)
    assert sum(row["n_traversals"] for row in rows) == 50 * 3
    first_edge = network.route(0, 3)[0]
    first_row = next(row for row in rows if row["edge_id"] == first_edge and row["bin_start_s"] == 0)
    assert first_row["mean_travel_time_s"] > network.edges[first_edge].free_flow_s

    network.set_travel_time_profile(rows)
    assert network.expected_time_s(first_edge, 10.0) == pytest.approx(first_row["mean_travel_time_s"])


def test_unroutable_trip_is_counted_not_spawned():
    nodes, edges, snaps = grid_tables(3, 1)
    edges = [edge for edge in edges if edge["from_node_id"] < edge["to_node_id"]]  # one-way eastbound
    network = TrafficNetwork.from_tables(nodes, edges, snaps)
    mobsim = VehicleMobsim(COMM, network, single_rank(network))

    assert not mobsim.spawn(1, 0, 2, 0, depart_s=0.0)
    assert mobsim.stats.unroutable == 1
    assert mobsim.vehicles_in_network == 0


def test_spawn_between_places_uses_snaps():
    network = grid_network(3, 1)
    mobsim = VehicleMobsim(COMM, network, single_rank(network))
    assert mobsim.spawn_between_places(1, 0, 1000, 1002, depart_s=0.0)
    assert not mobsim.spawn_between_places(1, 1, 1000, 9999, depart_s=0.0)
    mobsim.advance_to(3600.0)
    assert [a.dest_node_id for a in mobsim.drain_arrivals()] == [2]


def test_global_stats_on_one_rank():
    network = grid_network(3, 1)
    mobsim = VehicleMobsim(COMM, network, single_rank(network))
    mobsim.spawn(1, 0, 0, 2, depart_s=0.0)
    mobsim.advance_to(3600.0)
    stats = mobsim.global_stats()
    assert stats["spawned"] == stats["arrived"] == 1
    assert stats["edge_entries"] == 2
    assert stats["in_network"] == 0


def test_strategic_draw_is_stable_and_roughly_uniform():
    draws = [_unit_hash(7, person, 0) for person in range(4000)]
    assert draws == [_unit_hash(7, person, 0) for person in range(4000)]
    assert all(0.0 <= d < 1.0 for d in draws)
    assert math.isclose(sum(d < 0.1 for d in draws) / len(draws), 0.1, abs_tol=0.02)


def test_sync_window_must_be_positive():
    network = grid_network(2, 1)
    with pytest.raises(ValueError):
        VehicleMobsim(COMM, network, single_rank(network), sync_window_s=0.0)
