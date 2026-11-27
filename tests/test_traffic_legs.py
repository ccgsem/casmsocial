"""Routed legs ride the vehicle mobsim: spawn on leg start, hold until arrival."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import pytest
from mpi4py import MPI
from repast4py.parameters import init_params

from casmsocial.activities import Act, Leg
from casmsocial.casmpop import CasmPop
from casmsocial.model import Model
from casmsocial.person import Person, get_transit_coupler, set_transit_coupler
from casmsocial.traffic import LegTrafficCoupler, VehicleMobsim, assign_node_ranks
from scripts.create_mvp_ducklake import create_mvp_tables
from tests.traffic_fixtures import grid_network

# ----------------------------------------------------------------------
# Coupler on its own
# ----------------------------------------------------------------------


def _coupler(capacity_vph: float | None = None) -> LegTrafficCoupler:
    network = grid_network(4, 1, capacity_vph=capacity_vph)  # nodes 0-1-2-3 in a row
    node_rank = assign_node_ranks(network, {})
    return LegTrafficCoupler(VehicleMobsim(MPI.COMM_SELF, network, node_rank, sync_window_s=1.0))


def _act(start: int, end: int, place_id: int) -> Act:
    return Act(person_id=1, activity_id=0, activity_sequence=0, starttime_min=start, endtime_min=end, place_id=place_id)


def _leg(origin_node: int = 0, dest_node: int = 3, mode: str = "drive") -> Leg:
    return Leg(
        mode=mode,
        origin_place_id=1000 + origin_node,
        destination_place_id=1000 + dest_node,
        origin_node_id=origin_node,
        destination_node_id=dest_node,
        travel_time_min=5,
    )


def test_leg_start_spawns_vehicle_at_previous_activity_end_and_holds_person():
    coupler = _coupler()
    coupler.advance(tick_min=480, minute_of_day=480)  # 08:00 on day one

    # Seen at 08:00; the previous activity ended at 07:50.
    assert coupler.on_leg_start(7, _leg(), _act(0, 470, 1000), _act(540, 600, 1003), minute_of_day=480)
    assert coupler.is_held(7)
    pending = coupler.held[7]
    assert pending.depart_s == 470 * 60
    assert pending.scheduled_arrival_s == 540 * 60
    assert pending.trip_id == 470

    done = coupler.advance(tick_min=495, minute_of_day=495)
    assert [record.person_id for record in done] == [7]
    assert not coupler.is_held(7)
    record = done[0]
    assert record.depart_s == 470 * 60
    assert record.arrive_s == pytest.approx(record.depart_s + record.free_flow_s)
    assert record.delay_s == 0.0
    assert (record.origin_place_id, record.destination_place_id) == (1000, 1003)


def test_departure_before_the_previous_tick_is_clamped_to_it():
    coupler = _coupler()
    coupler.advance(tick_min=60, minute_of_day=60)
    coupler.advance(tick_min=75, minute_of_day=75)
    # The previous activity ended at 00:10, before the run's previous tick (01:00).
    coupler.on_leg_start(1, _leg(), _act(0, 10, 1000), _act(120, 200, 1003), minute_of_day=75)
    assert coupler.held[1].depart_s == 60 * 60


def test_previous_activity_ending_yesterday_departs_yesterday():
    coupler = _coupler()
    coupler.advance(tick_min=1440 + 15, minute_of_day=15)  # 00:15 on day two
    coupler.advance(tick_min=1440 + 30, minute_of_day=30)
    coupler.on_leg_start(1, _leg(), _act(1300, 1435, 1000), _act(60, 120, 1003), minute_of_day=30)
    # 23:55 yesterday is before the previous tick (00:15), so it is clamped there.
    assert coupler.held[1].depart_s == (1440 + 15) * 60
    assert coupler.held[1].scheduled_arrival_s == (1440 + 15 + 65) * 60


@pytest.mark.parametrize(
    "leg, previous, following",
    [
        (Leg(mode="travel"), _act(0, 470, 1000), _act(540, 600, 1003)),  # unrouted
        (_leg(mode="walk"), _act(0, 470, 1000), _act(540, 600, 1003)),  # other mode
        (_leg(), None, _act(540, 600, 1003)),
        (_leg(), _act(0, 470, 1000), None),
    ],
)
def test_legs_the_mobsim_cannot_carry_are_skipped(leg, previous, following):
    coupler = _coupler()
    assert not coupler.on_leg_start(1, leg, previous, following, minute_of_day=480)
    assert not coupler.held
    assert coupler.skipped_legs == 1


def test_congestion_holds_travellers_past_their_scheduled_start():
    coupler = _coupler(capacity_vph=60)
    coupler.advance(tick_min=480, minute_of_day=480)
    for person in range(40):
        coupler.on_leg_start(person, _leg(), _act(0, 480, 1000), _act(500, 600, 1003), minute_of_day=480)

    released = []
    for tick in range(495, 2000, 15):
        released.extend(coupler.advance(tick_min=tick, minute_of_day=tick % 1440))
        if not coupler.held:
            break

    assert len(released) == 40
    assert any(record.delay_s > 0 for record in released)
    assert max(record.travel_time_s for record in released) > 5 * min(r.free_flow_s for r in released)
    summary = coupler.global_summary()
    assert summary["trips_completed"] == 40
    assert summary["trips_delayed"] == sum(1 for record in released if record.delay_s > 0)
    assert summary["held_now"] == 0
    rows = coupler.trip_rows()
    assert {row["person_id"] for row in rows} == set(range(40))
    assert all("delay_s" in row for row in rows)


def test_unmatched_arrival_is_ignored():
    coupler = _coupler()
    coupler.advance(tick_min=0, minute_of_day=0)
    coupler.mobsim.spawn(99, 5, 0, 3, 0.0)  # not a leg the coupler started
    assert coupler.advance(tick_min=60, minute_of_day=60) == []
    assert coupler.trip_records == []


# ----------------------------------------------------------------------
# CasmPop with the MVP fixture
# ----------------------------------------------------------------------


@pytest.fixture
def mvp_model_factory(tmp_path, monkeypatch):
    conn = duckdb.connect(":memory:")
    create_mvp_tables(conn)

    def use_test_data_resources(model: CasmPop) -> None:
        model.data_path = tmp_path
        model.conn = conn

    previous_model = Model.get_model()
    previous_behavior_engine = Person.getBehaviorEngine()
    previous_activity_names = CasmPop.get_activity_names()
    previous_planned_activity_names = CasmPop.get_planned_activity_names()
    previous_activities_data_type = CasmPop._CasmPop__activities_data_type

    monkeypatch.setattr(CasmPop, "_set_data_resources", use_test_data_resources)
    CasmPop.register_activity_names([])
    CasmPop.register_planned_activity_names([])
    CasmPop._CasmPop__activities_data_type = None

    def build(**overrides) -> CasmPop:
        params = init_params(str(Path("config/mvp.yaml")), "{}")
        params.update({
            "roads.enabled": True,
            "roads.nodes.file": "casmsocial_mvp.road_nodes",
            "roads.edges.file": "casmsocial_mvp.road_edges",
            "roads.place_snap.file": "casmsocial_mvp.place_road_snap",
            "behavior.engine": "schedule",
            "behavior.llm.enabled": False,
            "communication.enabled": False,
            "observers.agent_log.enabled": False,
            "observers.behavior_log.enabled": False,
            "time.step.minutes": 15,
            "traffic.enabled": True,
        })
        params.update(overrides)
        model = CasmPop(MPI.COMM_SELF, params)
        model.build_context()
        return model

    try:
        yield build
    finally:
        set_transit_coupler(None)
        CasmPop.register_activity_names(previous_activity_names)
        CasmPop.register_planned_activity_names(previous_planned_activity_names)
        CasmPop._CasmPop__activities_data_type = previous_activities_data_type
        Person.registerBehaviorEngine(previous_behavior_engine)
        Model.theModel = previous_model
        conn.close()


def _persons(model: CasmPop) -> dict[int, Person]:
    return {person.id: person for person in model.context.agents(agent_type=Person.TYPE)}


def _run_day(model: CasmPop) -> dict[int, dict[int, int]]:
    """Step one day; return {minute_of_day: {person_id: place_id}} after each tick."""
    places_by_minute = {}
    for _ in range(24 * 60 // model.time_step_minutes):
        model.step()
        places_by_minute[model.cal.minute_of_day] = {pid: p.place_id for pid, p in _persons(model).items()}
    return places_by_minute


def test_traffic_is_off_by_default(mvp_model_factory):
    model = mvp_model_factory(**{"traffic.enabled": False})
    assert model.traffic is None
    assert get_transit_coupler() is None


def test_traffic_requires_roads(mvp_model_factory):
    with pytest.raises(ValueError, match="roads.enabled"):
        mvp_model_factory(**{"roads.enabled": False})


def test_free_flow_legs_keep_the_schedule(mvp_model_factory):
    model = mvp_model_factory()
    assert get_transit_coupler() is model.traffic

    places = _run_day(model)

    records = sorted(model.traffic.trip_records, key=lambda r: (r.depart_s, r.person_id))
    assert [(r.person_id, r.origin_place_id, r.destination_place_id) for r in records] == [
        (1, 100, 300),
        (2, 200, 300),
        (1, 300, 100),
        (2, 300, 200),
    ]
    assert all(record.delay_s == 0 for record in records)
    # Departs when the previous activity ends (08:00, 17:00), arrives after free-flow time.
    assert [r.depart_s for r in records] == [480 * 60, 480 * 60, 1020 * 60, 1020 * 60]
    assert {r.person_id: r.free_flow_s for r in records} == {1: 12 * 60, 2: 10 * 60}
    # Arrived early, so people still enter at the scheduled start, as without traffic.
    assert places[525] == {1: 0, 2: 0}
    assert places[540] == {1: 300, 2: 300}
    assert places[1080] == {1: 100, 2: 200}
    assert not model.traffic.held

    model._log_traffic_summary()
    assert get_transit_coupler() is None


def test_congested_leg_delays_entering_the_activity(mvp_model_factory):
    model = mvp_model_factory()
    network = model.traffic.mobsim.network
    # One car on edge 1 (node 1 -> 3, person 1's morning route) is enough to jam it.
    network.edges[1] = dataclasses.replace(network.edges[1], capacity_vph=0.05)

    places = _run_day(model)

    morning = {r.person_id: r for r in model.traffic.trip_records if r.destination_place_id == 300}
    assert morning[2].delay_s == 0
    late = morning[1]
    assert late.delay_s > 60 * 60
    arrive_min = late.arrive_s / 60
    first_tick_after = min(minute for minute in places if minute >= arrive_min)
    # Held in transit through the scheduled 09:00 start until the tick after arrival.
    for minute in range(540, first_tick_after, 15):
        assert places[minute][1] == 0, minute
    assert places[first_tick_after][1] == 300
    assert places[540][2] == 300
