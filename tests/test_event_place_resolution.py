from collections import namedtuple

from mpi4py import MPI

from casmsocial.activities import Act, make_plan, make_routed_plan, resolve_activity_place
from casmsocial.person import Person
from casmsocial.place import EnhancedPlacesProjection, Place


def test_event_place_id_overrides_legacy_activity_anchor():
    activity = Act(
        person_id=1,
        activity_id=4,
        activity_sequence=2,
        starttime_min=600,
        endtime_min=660,
        place_id=9002,
    )

    assert resolve_activity_place(activity, [10, 11, 12, 13, 9001]) == 9002


def test_event_place_resolution_falls_back_to_legacy_activity_anchor():
    activity = Act(
        person_id=1,
        activity_id=1,
        activity_sequence=1,
        starttime_min=480,
        endtime_min=540,
        place_id=0,
    )

    assert resolve_activity_place(activity, [10, 11]) == 11


def test_schedule_movement_uses_event_place_over_repeated_activity_anchor():
    places = namedtuple("Places", ["home", "shopping"])(100, 9001)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 4, 1, 120, 180, 9002),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    shopping = Place({"sp_id": 9002, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add_place(shopping)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    person.move_at(90, True, projection)
    assert person.move_at(150, True, projection) is True
    assert person.place_id == 9002
    assert projection.get_place_for_agent(person) == shopping


def test_routed_leg_uses_event_places_over_activity_type_anchors():
    activities = [Act(1, 0, 0, 0, 60, 100), Act(1, 4, 1, 120, 180, 9002)]

    class NoRouteNetwork:
        def route_between_places(self, *_args):
            return None

    plan = make_routed_plan(activities, [100, 9001], NoRouteNetwork())

    assert plan[1].origin_place_id == 100
    assert plan[1].destination_place_id == 9002
