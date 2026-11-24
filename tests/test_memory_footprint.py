"""Per-person memory savings that must not change behaviour: a shared local adapter and compact memory traces."""

from __future__ import annotations

import gc
import pickle
import weakref
from collections import namedtuple

import pytest

from casmsocial.activities import DEFAULT_TRAVEL_LEG, Act, Leg, make_plan, restore_plans, serialize_plans
from casmsocial.household import Household, HouseholdData
from casmsocial.llm_adapter import LocalBehaviorLLMAdapter
from casmsocial.person import (
    LLMBehaviorEngine,
    MemoryRecord,
    Person,
    _shared_value,
    memory_event_dict,
    reset_shared_values,
)
from casmsocial.sim_time import SimTime

_Places = namedtuple("_Places", ["home", "work"])
_Cal = namedtuple("_Cal", ["tick"])


@pytest.fixture(autouse=True)
def _restore_engine_config():
    previous_engine = Person.getBehaviorEngine()
    previous_config = LLMBehaviorEngine._config
    yield
    LLMBehaviorEngine._config = previous_config
    Person.registerBehaviorEngine(previous_engine)


def _engine(person_id: int = 1) -> LLMBehaviorEngine:
    Person.registerBehaviorEngine(LLMBehaviorEngine)
    plan = make_plan([Act(person_id, 0, 0, 0, 60, 100)])
    person = Person(person_id, 0, [plan], _Places(100, 200), {"sp_id": person_id})
    return person.behavior_engine


def test_engines_share_one_local_adapter_until_the_config_changes():
    a, b = _engine(1), _engine(2)
    assert isinstance(a.adapter, LocalBehaviorLLMAdapter)
    assert a.adapter is b.adapter

    LLMBehaviorEngine.configure(signal_cap=2.5)
    c = _engine(3)
    assert c.adapter is not a.adapter and c.adapter.signal_cap == 2.5
    assert _engine(4).adapter is c.adapter


def test_a_config_changed_in_place_builds_a_new_adapter():
    a = _engine(1)
    LLMBehaviorEngine._config = dict(LLMBehaviorEngine._config)
    LLMBehaviorEngine._config["memory_decay"] = 0.5
    b = _engine(2)
    assert b.adapter is not a.adapter and b.adapter.memory_decay == 0.5
    LLMBehaviorEngine._config["memory_decay"] = 0.4  # same dict object, new value
    assert _engine(3).adapter.memory_decay == 0.4


def test_an_injected_adapter_is_used_as_given():
    sentinel = object()
    LLMBehaviorEngine.configure(adapter_instance=sentinel)
    assert _engine(1).adapter is sentinel and _engine(2).adapter is sentinel


def test_equal_consecutive_traces_share_their_data_and_keep_their_values():
    engine = _engine()
    data = {"action": "stay_home", "threat": round(0.123456, 4), "policy_as_of": "2026-10-15"}
    for tick in (300, 360, 420):
        engine._record_memory_event(_Cal(tick), "message_appraisal", "reason", dict(data))
    engine._record_memory_event(_Cal(480), "message_appraisal", "reason", {**data, "threat": 0.2})

    memory = engine.cognition.episodic_memory
    assert [event["tick"] for event in memory] == [300, 360, 420, 480]
    assert [event["data"] for event in memory[:3]] == [data] * 3
    assert memory[0]["data"] is memory[1]["data"] is memory[2]["data"]
    assert memory[3]["data"] == {**data, "threat": 0.2} and memory[3]["data"] is not memory[2]["data"]


def test_shared_values_are_equal_objects_of_the_same_type():
    x = float("0.1234")
    y = float("0.1234")
    assert x is not y
    assert _shared_value(x) is _shared_value(y)
    assert _shared_value(1.0) == 1.0 and isinstance(_shared_value(1.0), float)
    assert isinstance(_shared_value(1), int) and not isinstance(_shared_value(1), bool) and _shared_value(True) is True
    negative_zero = _shared_value(-0.0)
    assert str(negative_zero) == "-0.0"
    nan = float("nan")
    assert _shared_value(nan) is nan
    big = 10**6 + 1
    assert _shared_value(big) == big


def _duckdb_model(params):
    import duckdb

    from casmsocial.casmpop import CasmPop

    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.params = params
    model.conn = duckdb.connect(":memory:")
    return model


def _setting(model, name):
    return model.conn.execute("SELECT value FROM duckdb_settings() WHERE name = ?", [name]).fetchone()[0]


def test_duckdb_threads_default_to_the_cpus_this_process_may_use(monkeypatch):
    import casmsocial.casmpop as casmpop

    monkeypatch.setattr(casmpop, "_usable_cpu_count", lambda ranks_on_node=1: 3)
    model = _duckdb_model({})
    model._configure_duckdb_resources()
    assert _setting(model, "threads") == "3"


def test_duckdb_threads_and_memory_limit_can_be_set():
    model = _duckdb_model({"duckdb.threads": 2, "duckdb.memory_limit": "1GB"})
    model._configure_duckdb_resources()
    assert _setting(model, "threads") == "2"
    assert _setting(model, "memory_limit") in {"1.0 GiB", "953.6 MiB", "1.0 GB"}


def test_duckdb_threads_must_be_positive():
    model = _duckdb_model({"duckdb.threads": 0})
    with pytest.raises(ValueError, match="duckdb.threads"):
        model._configure_duckdb_resources()


def test_usable_cpu_count_follows_the_affinity_mask(monkeypatch):
    import casmsocial.casmpop as casmpop

    monkeypatch.setattr(casmpop.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert casmpop._usable_cpu_count() == 4
    monkeypatch.delattr(casmpop.os, "sched_getaffinity")
    monkeypatch.setattr(casmpop.os, "cpu_count", lambda: 10)
    assert casmpop._usable_cpu_count() == 10


def test_a_household_does_not_keep_a_member_who_left_alive():
    household = Household({"household_id": 7, "place_id": 100}, HouseholdData)
    stays, leaves = _engine(1).agent, _engine(2).agent
    household.addMember(stays)
    household.addMember(leaves)
    household.addMember(stays)
    assert household.getHouseholdMembers() == [stays, leaves]

    gone = weakref.ref(leaves)
    del leaves
    gc.collect()  # the person and its engine refer to each other
    assert gone() is None
    assert household.getHouseholdMembers() == [stays]


def test_migrated_traces_are_shared_again_and_keep_their_values():
    engine = _engine()
    for tick in (1, 2):
        engine._record_memory_event(_Cal(tick), "decision", "follow_schedule", {"action": "stay_home", "score": 0.25})
    arrived = _engine(2)
    # Migration pickles the state, so every trace and value arrives as a new object.
    arrived.restore_state(pickle.loads(pickle.dumps(engine.save_state())))
    events = arrived.cognition.episodic_memory
    assert events == engine.cognition.episodic_memory
    assert events[0]["data"] is events[1]["data"]
    assert events[0]["summary"] is _shared_value("follow_schedule")


def test_trace_data_is_reused_only_for_values_of_the_same_type():
    engine = _engine()
    for data in ({"score": 1.0}, {"score": 1}, {"score": True}, {"x": 0.0}, {"x": -0.0}, {"x": -0.0}):
        engine._record_memory_event(_Cal(1), "decision", "follow_schedule", data)
    stored = [event["data"] for event in engine.cognition.episodic_memory]
    assert [type(d.get("score", d.get("x"))) for d in stored] == [float, int, bool, float, float, float]
    assert str(stored[4]["x"]) == "-0.0"
    assert stored[5] is stored[4]  # same values, same type: shared
    assert stored[1] is not stored[0] and stored[2] is not stored[1] and stored[4] is not stored[3]


def test_adapters_get_copies_of_the_shared_trace_data():
    engine = _engine()
    for tick in (1, 2):
        engine._record_memory_event(_Cal(tick), "decision", "follow_schedule", {"action": "stay_home"})
    context = engine._build_llm_context({}, SimTime())
    context["recent_memory"][0]["data"]["action"] = "edited"
    assert [event["data"]["action"] for event in engine.cognition.episodic_memory] == ["stay_home", "stay_home"]


def test_copies_of_shared_trace_data_are_independent_of_each_other():
    engine = _engine()
    recorded = ((1, "stay_home"), (2, "stay_home"), (3, "go_out"), (4, "stay_home"), (5, "stay_home"))
    for tick, action in recorded:
        engine._record_memory_event(_Cal(tick), "decision", "follow_schedule", {"action": action})
    stored = engine.cognition.episodic_memory
    assert stored[0]["data"] is stored[1]["data"] and stored[3]["data"] is stored[4]["data"]

    recent = engine._build_llm_context({}, SimTime())["recent_memory"]
    assert recent == [memory_event_dict(event) for event in stored]
    assert all(isinstance(event["data"], dict) for event in recent)
    assert len({id(event["data"]) for event in recent}) == len(recent)
    recent[0]["data"]["action"] = recent[3]["data"]["action"] = "edited"
    assert [event["data"]["action"] for event in recent] == ["edited", "stay_home", "go_out", "edited", "stay_home"]
    assert [event["data"]["action"] for event in stored] == [action for _, action in recorded]


def test_unrounded_floats_are_not_shared_and_the_table_resets(monkeypatch):
    import casmsocial.person as person

    monkeypatch.setattr(person, "_SHARED_VALUES", {})
    assert _shared_value(float("0.123456789")) is not None and person._SHARED_VALUES == {}
    _shared_value(float("0.25"))
    _shared_value("stay_home")
    assert len(person._SHARED_VALUES) == 2
    person.reset_shared_values()
    assert person._SHARED_VALUES == {}


def test_a_nested_config_change_builds_a_new_shared_adapter():
    LLMBehaviorEngine.configure(activity_semantics_overrides={"social_ids": [1]})
    a = _engine(1)
    LLMBehaviorEngine._config["activity_semantics_overrides"]["social_ids"].append(2)
    b = _engine(2)
    assert b.adapter is not a.adapter


def test_usable_cpu_count_splits_the_machine_between_unbound_ranks(monkeypatch):
    import casmsocial.casmpop as casmpop

    monkeypatch.setattr(casmpop.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(casmpop.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    assert casmpop._usable_cpu_count(16) == 4  # plain mpirun: every rank may use the whole node
    monkeypatch.setattr(casmpop.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert casmpop._usable_cpu_count(16) == 4  # bound by Slurm: the mask is the answer
    assert casmpop._ranks_on_this_node(None) == 1


def test_a_person_coming_home_rejoins_their_household():
    from casmsocial.casmpop import CasmPop

    person = _engine(1).agent  # home place 100
    household = Household({"household_id": 7, "place_id": 100}, HouseholdData)
    model = CasmPop.__new__(CasmPop)
    model.households_by_place_id = {100: household}
    data = person.save()
    del person
    gc.collect()
    assert household.getHouseholdMembers() == []

    back = model.restore_person(data)
    assert household.getHouseholdMembers() == [back]


def test_restored_plans_share_the_travel_leg_and_equal_integers():
    person_id, home = 70_000_001, 900_001_234
    plan = [
        Act(person_id, 1, 0, 0, 480, home),
        DEFAULT_TRAVEL_LEG,
        Act(person_id, 2, 1, 510, 1020, 900_005_678),
        Leg("drive", home, 900_005_678, 1, 2, 1234.5, 17),
        Act(person_id, 3, 2, 1050, 1440, home),
    ]

    restored = restore_plans(pickle.loads(pickle.dumps(serialize_plans([plan]))))[0]

    assert restored == plan
    assert restored[1] is DEFAULT_TRAVEL_LEG
    assert restored[3] is not DEFAULT_TRAVEL_LEG and isinstance(restored[3], Leg)
    assert restored[0].person_id is restored[2].person_id is restored[4].person_id
    assert restored[0].place_id is restored[4].place_id


def test_stored_traces_read_like_dicts_and_cannot_be_changed():
    engine = _engine()
    data = {"action": "stay_home", "threat": 0.25, "pathway": "none"}
    engine._record_memory_event(_Cal(3), "message_appraisal", "reason", dict(data))
    event = engine.cognition.episodic_memory[-1]

    assert isinstance(event, MemoryRecord) and isinstance(event["data"], MemoryRecord)
    assert event == {"tick": 3, "event_type": "message_appraisal", "summary": "reason", "data": data}
    assert {"tick": 3, "event_type": "message_appraisal", "summary": "reason", "data": data} == event
    assert list(event) == ["tick", "event_type", "summary", "data"] and len(event["data"]) == 3
    assert event["data"].get("action") == "stay_home" and event["data"].get("missing", 7) == 7
    assert "threat" in event["data"] and "missing" not in event["data"]
    assert dict(event["data"]) == data and {**event}["tick"] == 3
    with pytest.raises(KeyError):
        event["data"]["missing"]
    with pytest.raises(TypeError):
        event["data"]["action"] = "edited"  # type: ignore[index]


def test_traces_with_the_same_keys_share_one_layout():
    first, second = _engine(1), _engine(2)
    first._record_memory_event(_Cal(1), "decision", "a", {"action": "stay_home", "score": 0.1})
    second._record_memory_event(_Cal(2), "decision", "b", {"action": "go", "score": 0.9})
    a, b = first.cognition.episodic_memory[-1], second.cognition.episodic_memory[-1]
    assert a._layout is b._layout and a["data"]._layout is b["data"]._layout


def test_stored_traces_pickle_as_compact_records():
    engine = _engine()
    for tick in (1, 2):
        engine._record_memory_event(
            _Cal(tick), "decision", "follow_schedule", {"action": "stay_home", "nested": [1, 2]}
        )
    events = engine.cognition.episodic_memory

    arrived = pickle.loads(pickle.dumps(events))
    assert all(isinstance(e, MemoryRecord) and isinstance(e["data"], MemoryRecord) for e in arrived)
    assert arrived == events
    assert arrived[1] == {
        "tick": 2,
        "event_type": "decision",
        "summary": "follow_schedule",
        "data": {"action": "stay_home", "nested": [1, 2]},
    }
    assert arrived[0]._layout is events[0]._layout  # the receiving rank's layout
    assert arrived[0]["data"] is arrived[1]["data"]  # data shared between traces stays shared
    assert arrived[0]["summary"] is _shared_value("follow_schedule")  # values shared with this rank's


def test_traces_pickled_as_plain_dicts_are_still_compacted_on_arrival():
    engine = _engine()
    for tick in (1, 2):
        engine._record_memory_event(_Cal(tick), "decision", "follow_schedule", {"action": "stay_home"})
    state = list(engine.save_state())
    plain = [memory_event_dict(e) for e in engine.cognition.episodic_memory]
    state[[i for i, v in enumerate(state) if v is engine.cognition.episodic_memory][0]] = plain
    arrived = _engine(2)
    arrived.restore_state(tuple(state))
    events = arrived.cognition.episodic_memory
    assert all(isinstance(e, MemoryRecord) for e in events) and events == plain
    assert events[0]["data"] is events[1]["data"]


def test_resetting_shared_values_also_forgets_trace_layouts():
    engine = _engine()
    engine._record_memory_event(_Cal(1), "decision", "a", {"action": "stay_home"})
    before = engine.cognition.episodic_memory[-1]
    reset_shared_values()
    engine._record_memory_event(_Cal(2), "decision", "a", {"action": "stay_home"})
    after = engine.cognition.episodic_memory[-1]
    assert after == {**before, "tick": 2} and after._layout is not before._layout


def test_trace_copies_are_plain_dicts_equal_to_the_stored_traces():
    engine = _engine()
    for tick in (1, 2):
        engine._record_memory_event(_Cal(tick), "decision", "follow_schedule", {"action": "stay_home", "nested": [1]})
    engine.cognition.episodic_memory.append({"tick": 3, "event_type": "x", "summary": "", "data": {"a": 1}})
    stored = engine.cognition.episodic_memory

    copies = [memory_event_dict(event) for event in stored]
    assert copies == stored and stored == copies
    assert all(isinstance(c, dict) and isinstance(c["data"], dict) for c in copies)
    copies[0]["data"]["action"] = "edited"
    assert stored[0]["data"]["action"] == "stay_home" and stored[1]["data"]["action"] == "stay_home"
    assert engine._build_llm_context({}, SimTime())["recent_memory"] == [memory_event_dict(e) for e in stored]


def test_records_compare_by_keys_and_values():
    first, second = _engine(1), _engine(2)
    first._record_memory_event(_Cal(1), "decision", "a", {"action": "stay_home", "score": 0.5})
    second._record_memory_event(_Cal(1), "decision", "a", {"score": 0.5, "action": "stay_home"})
    a, b = first.cognition.episodic_memory[-1], second.cognition.episodic_memory[-1]
    assert a == b  # same keys in another order, as dicts compare
    assert a["data"] != {"action": "stay_home", "score": 0.25} and a != 3
    assert a["data"].to_dict() == {"action": "stay_home", "score": 0.5}
