from __future__ import annotations

from collections import namedtuple
from datetime import datetime

from mpi4py import MPI

from casmsocial.activities import DEFAULT_TRAVEL_LEG, Act, activity_semantics_for, default_activity_semantics, make_plan
from casmsocial.communication.types import CommMessage, MessageKind, build_message_payload
from casmsocial.person import BehaviorEngineV2, LLMBehaviorEngine, Person, ScheduleBehaviorEngine
from casmsocial.place import EnhancedPlacesProjection, Place


def test_behavior_engine_v2_import():
    assert BehaviorEngineV2.__name__ == "BehaviorEngineV2"
    assert ScheduleBehaviorEngine.__name__ == "ScheduleBehaviorEngine"
    assert LLMBehaviorEngine.__name__ == "LLMBehaviorEngine"


def test_person_default_field_constructor_initializes_default_state():
    places = (100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    person = Person.from_default_fields(
        1,
        0,
        [plan],
        places,
        x=float("inf"),
        y=None,
        minute_last_moved=30,
        network={"home": 100},
        behavior_engine=ScheduleBehaviorEngine,
    )

    assert person.uid == (1, Person.TYPE, 0)
    assert tuple(person.location.coordinates) == (0, 0, 0)
    assert person.state.person_id == 1
    assert person.state.place_id == 100
    assert person.state.rank_place_id == 100
    assert person.state.places is places
    assert person.state.minute_last_moved == 30
    assert person.network == {"home": 100}
    assert isinstance(person.behavior_engine, ScheduleBehaviorEngine)


def test_person_zero_location_constructor_initializes_default_state():
    places = (100, 200)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])

    person = Person.from_default_zero_location(1, 0, [plan], places, ScheduleBehaviorEngine)

    assert person.uid == (1, Person.TYPE, 0)
    assert tuple(person.location.coordinates) == (0, 0, 0)
    assert person.state.place_id == 100
    assert person.state.rank_place_id == 100
    assert person.state.places is places
    assert person.state.minute_last_moved == 0
    assert person.network is None
    assert person.__dict__["_behavior_engine"] is None
    assert isinstance(person.behavior_engine, ScheduleBehaviorEngine)
    assert person.__dict__["_behavior_engine"] is person.behavior_engine


def test_person_schedule_zero_location_constructor_initializes_hot_path_state():
    places = (100, 200, None)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])

    person = Person.from_default_schedule_zero_location(1, 0, plan, places)

    assert person.uid == (1, Person.TYPE, 0)
    assert person.plans == [plan, []]
    assert tuple(person.location.coordinates) == (0, 0, 0)
    assert person.state.place_id == 100
    assert person.state.rank_place_id == 100
    assert person.state.places is places
    assert "inbox" not in person.__dict__
    assert person.__dict__["_behavior_engine"] is None
    assert isinstance(person.behavior_engine, ScheduleBehaviorEngine)


def test_person_default_constructor_defers_schedule_behavior_engine_until_access():
    places = (100,)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])

    person = Person.from_default_fields(
        1,
        0,
        [plan],
        places,
        behavior_engine=ScheduleBehaviorEngine,
    )

    assert person.__dict__["_behavior_engine"] is None
    behavior_engine = person.behavior_engine
    assert isinstance(behavior_engine, ScheduleBehaviorEngine)
    assert person.behavior_engine is behavior_engine


def test_person_set_behavior_engine_defers_schedule_engine_until_access():
    places = (100,)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])
    person = Person.from_default_fields(1, 0, [plan], places, behavior_engine=LLMBehaviorEngine)

    assert isinstance(person.behavior_engine, LLMBehaviorEngine)

    person.setBehaviorEngine(ScheduleBehaviorEngine)

    assert person.__dict__["_behavior_engine"] is None
    assert isinstance(person.behavior_engine, ScheduleBehaviorEngine)


def test_person_zero_location_constructor_can_defer_communication_state():
    places = (100,)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])

    person = Person.from_default_zero_location(
        1,
        0,
        [plan],
        places,
        ScheduleBehaviorEngine,
        initialize_communication_state=False,
    )

    assert "inbox" not in person.__dict__
    assert "recent_messages" not in person.__dict__
    assert "outbound_message_intents" not in person.__dict__
    assert "pending_acks" not in person.__dict__
    assert person.decide_messages(None) == []
    assert person.save()[4] == {}

    message = CommMessage(
        msg_id="msg-1",
        sender_uid=(2, Person.TYPE, 0),
        sender_place_id=100,
        receiver_uid=person.uid,
        receiver_place_id=100,
        mode="local",
        payload=build_message_payload(MessageKind.CHECK_IN),
        tick=1,
    )
    person.receive(message)
    person.process_inbox(None)

    assert person.inbox == []
    assert person.recent_messages == [message]


def test_person_move_transitions_across_leg():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    person = Person(1, 0, [plan], places, {"sp_id": 1})

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add_place(work)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    person.move(_sim_time_at(30), projection)
    assert projection.get_place_for_agent(person) == home
    assert person.place_id == 100
    assert person.rank_place_id == 100

    changed = person.move(_sim_time_at(90), projection)
    assert changed is True
    assert projection.get_place_for_agent(person) is None
    assert person.place_id == 0
    assert person.rank_place_id == 100

    changed = person.move(_sim_time_at(150), projection)
    assert changed is True
    assert projection.get_place_for_agent(person) == work
    assert person.place_id == 200
    assert person.rank_place_id == 200


def test_person_move_at_skips_projection_lookup_when_place_is_unchanged():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    class TrackingProjection(EnhancedPlacesProjection):
        def __init__(self):
            super().__init__("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
            self.lookup_calls = 0

        def lookup_place(self, place_id):
            self.lookup_calls += 1
            return super().lookup_place(place_id)

    person = Person(1, 0, [plan], places, {"sp_id": 1})
    projection = TrackingProjection()
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    changed = person.move_at(30, True, projection)

    assert changed is False
    assert projection.lookup_calls == 0
    assert projection.get_place_for_agent(person) == home


def test_person_can_target_known_remote_place_without_local_place_object():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    person = Person(1, 0, [plan], places, {"sp_id": 1})

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    projection.set_place_rank_map({100: 0, 200: 1})
    projection.add_place(home)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    person.move(_sim_time_at(90), projection)

    changed = person.move(_sim_time_at(150), projection)

    assert changed is True
    assert projection.get_place_for_agent(person) is None
    assert person.place_id == 200
    assert person.rank_place_id == 200
    assert person.uid[2] == 0


def test_current_leg_returns_active_leg_only_during_transit():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    person = Person(1, 0, [plan], places, {"sp_id": 1})

    assert person.current_leg(_sim_time_at(30)) is None

    leg = person.current_leg(_sim_time_at(90))
    assert leg is not None
    assert leg.mode == "travel"

    assert person.current_leg(_sim_time_at(150)) is None


def test_plan_state_resolves_after_minute_wraps():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})

    later_state = person.get_plan_state(_sim_time_at(150))
    wrapped_state = person.get_plan_state(_sim_time_at(30))

    assert later_state.element.activity_id == 1
    assert wrapped_state.element.activity_id == 0


def test_make_plan_reuses_default_travel_leg():
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )

    assert plan[1] is DEFAULT_TRAVEL_LEG


def test_float_activity_id_resolves_place_without_crashing():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = [
        Act(1, 0.0, 0, 0, 60, 100),
        Act(1, 1.0, 1, 120, 180, 200),
    ]

    person = Person(1, 0, [plan], places, {"sp_id": 1})

    assert person._place_id_for_activity(1.0) == 200


def test_process_inbox_caches_recent_messages_for_behavior():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])
    person = Person(1, 0, [plan], places, {"sp_id": 1})

    person.receive(
        CommMessage(
            msg_id="msg-1",
            sender_uid=(2, 0, 0),
            sender_place_id=100,
            receiver_uid=person.uid,
            receiver_place_id=100,
            mode="two_way",
            payload=build_message_payload(
                MessageKind.CHECK_IN,
                metadata={"sender_place_id": 100},
            ),
            tick=1,
        )
    )
    person.process_inbox(None)

    assert len(person.inbox) == 0
    assert len(person.recent_messages) == 1
    assert person.recent_messages[0].msg_id == "msg-1"


def test_default_activity_semantics_uses_conservative_social_defaults():
    home_semantics = default_activity_semantics(0)
    work_semantics = default_activity_semantics(1)
    other_semantics = default_activity_semantics(3)

    assert home_semantics.is_home is True
    assert home_semantics.is_social is False
    assert home_semantics.is_flexible is False
    assert home_semantics.is_mandatory is False
    assert home_semantics.is_travel_sensitive is False
    assert work_semantics.is_home is False
    assert work_semantics.is_social is False
    assert work_semantics.is_flexible is True
    assert work_semantics.is_mandatory is False
    assert work_semantics.is_travel_sensitive is True
    assert other_semantics.is_home is False
    assert other_semantics.is_social is False
    assert other_semantics.is_flexible is True
    assert other_semantics.is_mandatory is False
    assert other_semantics.is_travel_sensitive is True


def test_activity_semantics_for_applies_social_override():
    overridden = activity_semantics_for(3, {"social_ids": [3]})
    baseline = activity_semantics_for(3)

    assert baseline.is_social is False
    assert overridden.is_social is True
    assert overridden.is_flexible is True
    assert overridden.is_mandatory is False


def test_llm_behavior_engine_queues_reply_intent_from_recent_message():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.recent_messages = [
        CommMessage(
            msg_id="msg-1",
            sender_uid=(2, 0, 0),
            sender_place_id=100,
            receiver_uid=person.uid,
            receiver_place_id=100,
            mode="two_way",
            payload=build_message_payload(
                MessageKind.CHECK_IN,
                metadata={"sender_place_id": 100},
            ),
            tick=1,
        )
    ]

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    class StubContext:
        def get_projection(self, name):
            assert name == "places_projection"
            return projection

    person.behavior_engine.decide(StubContext(), _sim_time_at(30))
    intents = person.decide_messages(None)

    assert len(intents) == 1
    assert intents[0].receiver_uid == (2, 0, 0)
    assert intents[0].payload["kind"] == MessageKind.ACKNOWLEDGMENT
    assert intents[0].payload["metadata"]["auto_reply"] is True
    assert person.behavior_engine.cognition.last_llm_summary != ""


def test_llm_behavior_engine_warning_message_favors_stay_home():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.WARNING,
            text="Storm warning",
            urgency=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A warning")
    assert person.place_id == 100
    assert person.behavior_engine.cognition.needs["safety"] >= 0.6
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_invitation_message_seeks_social_contact_and_replies():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.INVITATION,
            text="Join us for lunch",
            trust_weight=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)

    assert person.behavior_engine.cognition.last_llm_summary.startswith("An invitation")
    assert person.behavior_engine.cognition.needs["social"] >= 0.5
    assert len(intents) == 1
    assert intents[0].payload["kind"] == MessageKind.ACKNOWLEDGMENT


def test_llm_behavior_engine_request_help_sends_reply():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.REQUEST_HELP,
            text="Need help now",
            urgency=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A request for help")
    assert len(intents) == 1
    assert intents[0].payload["text"] == "Help request acknowledged"


def test_llm_behavior_engine_reminder_reinforces_schedule_without_reply():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.REMINDER,
            text="Don't forget your appointment",
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A reminder")
    assert person.place_id == 100
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_urgent_announcement_favors_stay_home():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.ANNOUNCEMENT,
            text="Facility closure",
            topic="closure",
            urgency=0.9,
            trust_weight=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A high-urgency announcement")
    assert person.place_id == 100
    assert person.behavior_engine.cognition.needs["safety"] >= 0.4
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_trusted_recommendation_can_favor_stay_home():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.RECOMMENDATION,
            text="Stay home today",
            topic="stay_home",
            urgency=0.8,
            trust_weight=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A trusted urgent recommendation")
    assert person.place_id == 100
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_coordination_request_prompts_reply():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Meet at the pickup point",
            topic="pickup",
            urgency=0.7,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A coordination request")
    assert len(intents) == 1
    assert intents[0].payload["text"] == "Coordination confirmed"


def test_llm_behavior_engine_warning_overrides_invitation_in_recent_window():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.INVITATION,
            text="Join us for lunch",
            trust_weight=0.9,
        ),
        _comm_message(
            person,
            MessageKind.WARNING,
            text="Severe storm warning",
            urgency=0.9,
        ),
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A warning")
    assert person.place_id == 100
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_request_help_overrides_invitation_with_reply():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.INVITATION,
            text="Come to lunch",
            trust_weight=0.8,
        ),
        _comm_message(
            person,
            MessageKind.REQUEST_HELP,
            text="Need help immediately",
            urgency=0.9,
        ),
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A request for help")
    assert len(intents) == 1
    assert intents[0].payload["text"] == "Help request acknowledged"


def test_llm_behavior_engine_multiple_reminders_keep_schedule_without_reply():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(person, MessageKind.REMINDER, text="Morning appointment", urgency=0.5),
        _comm_message(person, MessageKind.REMINDER, text="Leave on time", urgency=0.4),
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("A reminder")
    assert person.place_id == 100
    assert person.decide_messages(None) == []


def test_llm_behavior_engine_warning_memory_persists_after_messages_clear():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.WARNING,
            text="Storm warning",
            urgency=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    person.recent_messages = []
    person.behavior_engine.deliberation_interval = 0

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(90))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("Recent caution in memory")
    assert person.place_id == 100


def test_llm_behavior_engine_reminder_memory_reinforces_schedule_after_messages_clear():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(person, MessageKind.REMINDER, text="Morning appointment", urgency=0.6),
        _comm_message(person, MessageKind.REMINDER, text="Leave on time", urgency=0.5),
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    person.recent_messages = []
    person.behavior_engine.deliberation_interval = 0

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(90))

    assert person.behavior_engine.cognition.last_llm_summary.startswith("Recent reminders in memory")
    assert person.place_id == 100
    assert person.decide_messages(None) == []


def test_warning_stay_home_adjustment_rewrites_future_plan_to_home():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [_comm_message(person, MessageKind.WARNING, text="Storm warning", urgency=0.9)]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    plan = person.plans[0]
    assert len(plan) == 1
    assert isinstance(plan[0], Act)
    assert plan[0].activity_id == 0
    assert plan[0].place_id == 100
    assert plan[0].starttime_min == 30
    assert person.current_leg(_sim_time_at(90)) is None


def test_urgent_recommendation_stay_home_adjustment_rewrites_future_plan_to_home():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.RECOMMENDATION,
            text="Stay home today",
            topic="stay_home",
            urgency=0.9,
            trust_weight=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    plan = person.plans[0]
    assert len(plan) == 1
    assert isinstance(plan[0], Act)
    assert plan[0].activity_id == 0
    assert plan[0].place_id == 100
    assert plan[0].starttime_min == 30


def test_coordination_adjustment_defers_next_activity_and_replies():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Pickup is delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)
    plan = person.plans[0]

    assert len(intents) == 1
    assert intents[0].payload["text"] == "Coordination confirmed"
    assert isinstance(plan[2], Act)
    assert plan[2].starttime_min == 150
    assert plan[2].endtime_min == 210
    assert person.behavior_engine.cognition.episodic_memory[-1]["data"]["plan_adjustment_kind"] == "defer_next_activity"


def test_travel_recommendation_adjustment_defers_next_activity_without_reply():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.RECOMMENDATION,
            text="Delay travel due to congestion",
            topic="avoid_travel",
            urgency=0.5,
            trust_weight=0.6,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    plan = person.plans[0]

    assert person.decide_messages(None) == []
    assert isinstance(plan[2], Act)
    assert plan[2].starttime_min == 150
    assert plan[2].endtime_min == 210
    assert person.behavior_engine.cognition.episodic_memory[-1]["data"]["plan_adjustment_kind"] == "defer_next_activity"


def test_coordination_cancel_adjustment_skips_next_flexible_activity_and_replies():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Cancel pickup",
            topic="cancel",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    intents = person.decide_messages(None)
    plan = person.plans[0]

    assert len(intents) == 1
    assert intents[0].payload["text"] == "Coordination confirmed"
    assert len(plan) == 1
    assert isinstance(plan[0], Act)
    assert plan[0].activity_id == 0
    memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
    assert memory_data["plan_adjustment_kind"] == "skip_flexible_activity"


def test_recommendation_skip_adjustment_skips_next_flexible_activity_without_reply():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.RECOMMENDATION,
            text="Skip this trip",
            topic="skip",
            urgency=0.5,
            trust_weight=0.6,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    plan = person.plans[0]

    assert person.decide_messages(None) == []
    assert len(plan) == 1
    assert isinstance(plan[0], Act)
    assert plan[0].activity_id == 0
    memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
    assert memory_data["plan_adjustment_kind"] == "skip_flexible_activity"


def test_social_cancel_adjustment_cancels_next_social_activity_with_override():
    original_config = dict(LLMBehaviorEngine._config)
    LLMBehaviorEngine.configure(activity_semantics_overrides={"social_ids": [1]})
    try:
        person, projection = _make_llm_person_with_commute_plan()
        person.recent_messages = [
            _comm_message(
                person,
                MessageKind.COORDINATION,
                text="Cancel lunch",
                topic="cancel_social",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]

        person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
        intents = person.decide_messages(None)
        plan = person.plans[0]

        assert len(intents) == 1
        assert intents[0].payload["text"] == "Coordination confirmed"
        assert len(plan) == 1
        assert isinstance(plan[0], Act)
        assert plan[0].activity_id == 0
        memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
        assert memory_data["plan_adjustment_kind"] == "cancel_social_activity"
    finally:
        LLMBehaviorEngine._config = original_config


def test_social_cancel_adjustment_is_skipped_without_social_override():
    original_config = dict(LLMBehaviorEngine._config)
    LLMBehaviorEngine.configure(activity_semantics_overrides={})
    try:
        person, projection = _make_llm_person_with_commute_plan()
        original_plan = list(person.plans[0])
        person.recent_messages = [
            _comm_message(
                person,
                MessageKind.COORDINATION,
                text="Cancel lunch",
                topic="cancel_social",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]

        person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

        assert person.plans[0] == original_plan
        memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
        assert memory_data["plan_adjustment_requested_kind"] == ""
        assert memory_data["plan_adjustment_applied"] is False
        assert memory_data["plan_adjustment_skip_reason"] == ""
    finally:
        LLMBehaviorEngine._config = original_config


def test_defer_next_activity_leaves_plan_unchanged_when_no_slack_available():
    places_type = namedtuple("TestPlaces", ["home", "work", "school"])
    places = places_type(100, 200, 300)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
            Act(1, 2, 2, 180, 240, 300),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
    school = Place({"sp_id": 300, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add_place(work)
    projection.add_place(school)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    original_plan = list(person.plans[0])
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Pickup delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    adjusted_plan = person.plans[0]
    assert len(adjusted_plan) == len(original_plan)
    assert isinstance(adjusted_plan[2], Act)
    assert adjusted_plan[2].starttime_min == 120
    assert adjusted_plan[2].endtime_min == 180


def test_defer_next_activity_skips_future_home_activity_and_defers_work():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 0, 1, 120, 180, 100),
            Act(1, 1, 2, 240, 300, 200),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add_place(work)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Pickup delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    adjusted_plan = person.plans[0]
    assert isinstance(adjusted_plan[2], Act)
    assert adjusted_plan[2].activity_id == 0
    assert adjusted_plan[2].starttime_min == 120
    assert adjusted_plan[2].endtime_min == 180
    assert isinstance(adjusted_plan[4], Act)
    assert adjusted_plan[4].activity_id == 1
    assert adjusted_plan[4].starttime_min == 270
    assert adjusted_plan[4].endtime_min == 330


def test_defer_next_activity_ignores_non_travel_sensitive_only_future_home():
    places_type = namedtuple("TestPlaces", ["home"])
    places = places_type(100)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 0, 1, 120, 180, 100),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add(person)
    projection.assign_agent_to_place(person, home)

    original_plan = list(person.plans[0])
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.COORDINATION,
            text="Pickup delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    adjusted_plan = person.plans[0]
    assert adjusted_plan == original_plan


def test_warning_preserve_home_activity_prevents_later_departure():
    person, projection = _make_llm_person_with_commute_plan()
    person.recent_messages = [
        _comm_message(
            person,
            MessageKind.WARNING,
            text="Storm warning",
            urgency=0.9,
            trust_weight=0.9,
        )
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    changed = person.move(_sim_time_at(150), projection)

    assert changed in (False, True)
    assert person.current_leg(_sim_time_at(150)) is None
    assert projection.get_place_for_agent(person) is not None
    assert person.place_id == 100
    assert person.rank_place_id == 100


def test_coordination_defer_next_activity_delays_arrival_against_baseline():
    baseline_person, baseline_projection = _make_llm_person_with_commute_plan()
    delayed_person, delayed_projection = _make_llm_person_with_commute_plan()
    delayed_person.recent_messages = [
        _comm_message(
            delayed_person,
            MessageKind.COORDINATION,
            text="Pickup delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]

    delayed_person.behavior_engine.decide(_StubContext(delayed_projection), _sim_time_at(30))

    baseline_changed = baseline_person.move(_sim_time_at(135), baseline_projection)
    delayed_changed = delayed_person.move(_sim_time_at(135), delayed_projection)

    assert baseline_changed is True
    assert baseline_person.current_leg(_sim_time_at(135)) is None
    assert baseline_projection.get_place_for_agent(baseline_person) is not None
    assert baseline_person.place_id == 200

    assert delayed_changed is True
    assert delayed_person.current_leg(_sim_time_at(135)) is not None
    assert delayed_projection.get_place_for_agent(delayed_person) is None
    assert delayed_person.place_id == 0


def test_home_only_future_plan_adjustment_leaves_movement_unchanged():
    places_type = namedtuple("TestPlaces", ["home"])
    places = places_type(100)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 0, 1, 120, 180, 100),
        ]
    )
    baseline_person = Person(1, 0, [list(plan)], places, {"sp_id": 1})
    baseline_person.behavior_engine = LLMBehaviorEngine(baseline_person)
    adjusted_person = Person(2, 0, [list(plan)], places, {"sp_id": 2})
    adjusted_person.behavior_engine = LLMBehaviorEngine(adjusted_person)

    baseline_projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    adjusted_projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    baseline_home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    adjusted_home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    baseline_projection.add_place(baseline_home)
    adjusted_projection.add_place(adjusted_home)
    baseline_projection.add(baseline_person)
    adjusted_projection.add(adjusted_person)
    baseline_projection.assign_agent_to_place(baseline_person, baseline_home)
    adjusted_projection.assign_agent_to_place(adjusted_person, adjusted_home)

    adjusted_person.recent_messages = [
        _comm_message(
            adjusted_person,
            MessageKind.COORDINATION,
            text="Pickup delayed",
            topic="pickup",
            urgency=0.8,
            trust_weight=0.8,
        )
    ]
    adjusted_person.behavior_engine.decide(_StubContext(adjusted_projection), _sim_time_at(30))

    baseline_changed = baseline_person.move(_sim_time_at(150), baseline_projection)
    adjusted_changed = adjusted_person.move(_sim_time_at(150), adjusted_projection)

    assert baseline_changed is False
    assert adjusted_changed is False
    assert baseline_projection.get_place_for_agent(baseline_person) == baseline_home
    assert adjusted_projection.get_place_for_agent(adjusted_person) == adjusted_home
    assert baseline_person.place_id == adjusted_person.place_id == 100
    assert baseline_person.current_leg(_sim_time_at(150)) is None
    assert adjusted_person.current_leg(_sim_time_at(150)) is None


def test_skip_flexible_activity_prevents_later_departure():
    baseline_person, baseline_projection = _make_llm_person_with_commute_plan()
    skipped_person, skipped_projection = _make_llm_person_with_commute_plan()
    skipped_person.recent_messages = [
        _comm_message(
            skipped_person,
            MessageKind.RECOMMENDATION,
            text="Skip this trip",
            topic="skip",
            urgency=0.5,
            trust_weight=0.6,
        )
    ]

    skipped_person.behavior_engine.decide(_StubContext(skipped_projection), _sim_time_at(30))

    baseline_changed = baseline_person.move(_sim_time_at(135), baseline_projection)
    skipped_changed = skipped_person.move(_sim_time_at(135), skipped_projection)

    assert baseline_changed is True
    assert baseline_person.current_leg(_sim_time_at(135)) is None
    assert baseline_projection.get_place_for_agent(baseline_person) is not None
    assert baseline_person.place_id == 200

    assert skipped_changed is False
    assert skipped_person.current_leg(_sim_time_at(135)) is None
    assert skipped_projection.get_place_for_agent(skipped_person) is not None
    assert skipped_person.place_id == 100
    assert skipped_person.rank_place_id == 100


def test_cancel_social_activity_prevents_later_departure_with_social_override():
    original_config = dict(LLMBehaviorEngine._config)
    LLMBehaviorEngine.configure(activity_semantics_overrides={"social_ids": [1]})
    try:
        baseline_person, baseline_projection = _make_llm_person_with_commute_plan()
        canceled_person, canceled_projection = _make_llm_person_with_commute_plan()
        canceled_person.recent_messages = [
            _comm_message(
                canceled_person,
                MessageKind.COORDINATION,
                text="Cancel lunch",
                topic="cancel_social",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]

        canceled_person.behavior_engine.decide(_StubContext(canceled_projection), _sim_time_at(30))

        baseline_changed = baseline_person.move(_sim_time_at(135), baseline_projection)
        canceled_changed = canceled_person.move(_sim_time_at(135), canceled_projection)

        assert baseline_changed is True
        assert baseline_person.current_leg(_sim_time_at(135)) is None
        assert baseline_projection.get_place_for_agent(baseline_person) is not None
        assert baseline_person.place_id == 200

        assert canceled_changed is False
        assert canceled_person.current_leg(_sim_time_at(135)) is None
        assert canceled_projection.get_place_for_agent(canceled_person) is not None
        assert canceled_person.place_id == 100
        assert canceled_person.rank_place_id == 100
    finally:
        LLMBehaviorEngine._config = original_config


def test_llm_behavior_engine_repeated_warnings_saturate_memory_signal():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [
        _comm_message(person, MessageKind.WARNING, text="Warning 1", urgency=0.1, trust_weight=0.1),
        _comm_message(person, MessageKind.WARNING, text="Warning 2", urgency=0.1, trust_weight=0.1),
        _comm_message(person, MessageKind.WARNING, text="Warning 3", urgency=0.1, trust_weight=0.1),
    ]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

    memory_trace = person.behavior_engine.cognition.episodic_memory[-1]["data"]
    assert memory_trace["safety_signal"] == person.behavior_engine.adapter.signal_cap


def test_llm_behavior_engine_memory_decay_reduces_signal_over_successive_ticks():
    person, projection = _make_llm_person_with_home()
    person.recent_messages = [_comm_message(person, MessageKind.WARNING, text="Storm warning", urgency=0.9)]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))
    first_signal = person.behavior_engine.cognition.episodic_memory[-1]["data"]["safety_signal"]

    person.recent_messages = []
    person.behavior_engine.deliberation_interval = 0
    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(90))
    second_signal = person.behavior_engine.cognition.episodic_memory[-1]["data"]["safety_signal"]

    person.behavior_engine.decide(_StubContext(projection), _sim_time_at(150))
    third_signal = person.behavior_engine.cognition.episodic_memory[-1]["data"]["safety_signal"]

    assert first_signal >= second_signal >= third_signal
    assert second_signal < first_signal


class _StubContext:
    def __init__(self, projection):
        self.projection = projection

    def get_projection(self, name):
        assert name == "places_projection"
        return self.projection


def _make_llm_person_with_home():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan([Act(1, 0, 0, 0, 60, 100)])
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add(person)
    projection.assign_agent_to_place(person, home)
    return person, projection


def _make_llm_person_with_commute_plan():
    places_type = namedtuple("TestPlaces", ["home", "work"])
    places = places_type(100, 200)
    plan = make_plan(
        [
            Act(1, 0, 0, 0, 60, 100),
            Act(1, 1, 1, 120, 180, 200),
        ]
    )
    person = Person(1, 0, [plan], places, {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)

    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
    projection.add_place(home)
    projection.add_place(work)
    projection.add(person)
    projection.assign_agent_to_place(person, home)
    return person, projection


def _comm_message(
    person,
    kind: MessageKind,
    *,
    text: str,
    topic: str = "",
    urgency: float = 0.0,
    trust_weight: float = 0.5,
):
    return CommMessage(
        msg_id="msg-1",
        sender_uid=(2, 0, 0),
        sender_place_id=100,
        receiver_uid=person.uid,
        receiver_place_id=100,
        mode="two_way",
        payload=build_message_payload(
            kind,
            topic=topic,
            text=text,
            urgency=urgency,
            trust_weight=trust_weight,
            metadata={"sender_place_id": 100},
        ),
        tick=1,
    )


def _sim_time_at(minute_of_day: int):
    from casmsocial.sim_time import SimTime

    cal = SimTime(datetime(2025, 1, 1))
    cal.set_datetime(datetime(2025, 1, 1, minute_of_day // 60, minute_of_day % 60))
    return cal
