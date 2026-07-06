from __future__ import annotations

from casmsocial.activities import (
    Act,
    Leg,
    make_routed_plan,
    restore_plan_element,
    serialize_plan_element,
    validate_leg_against_schedule,
)
from casmsocial.road_network import RoadRoute


class FakeRoadNetwork:
    def route_between_places(self, origin_place_id, destination_place_id, mode="drive"):
        return RoadRoute(
            origin_place_id=origin_place_id,
            destination_place_id=destination_place_id,
            origin_node_id=1,
            destination_node_id=2,
            distance_m=1200.0,
            travel_time_min=8,
            node_ids=(1, 2),
        )


def test_make_routed_plan_inserts_route_metadata():
    activities = [
        Act(1, 0, 0, 0, 60, 100),
        Act(1, 1, 1, 90, 180, 200),
    ]
    places = [100, 200]

    plan = make_routed_plan(activities, places, FakeRoadNetwork())

    assert isinstance(plan[1], Leg)
    assert plan[1].origin_place_id == 100
    assert plan[1].destination_place_id == 200
    assert plan[1].origin_node_id == 1
    assert plan[1].destination_node_id == 2
    assert plan[1].distance_m == 1200.0
    assert plan[1].travel_time_min == 8


def test_leg_serialization_round_trip():
    leg = Leg("drive", 100, 200, 1, 2, 1200.0, 8)

    restored = restore_plan_element(serialize_plan_element(leg))

    assert restored == leg


def test_validate_leg_against_schedule():
    prev_act = Act(1, 0, 0, 0, 60, 100)
    next_act = Act(1, 1, 1, 90, 180, 200)

    assert validate_leg_against_schedule(prev_act, Leg("drive", travel_time_min=20), next_act) is True
    assert validate_leg_against_schedule(prev_act, Leg("drive", travel_time_min=40), next_act) is False
