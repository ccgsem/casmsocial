"""Per-person memory savings that must not change behaviour: a shared local adapter and compact memory traces."""

from __future__ import annotations

import gc
import pickle
import weakref
from collections import namedtuple

import pytest

from casmsocial.activities import Act, make_plan
from casmsocial.household import Household, HouseholdData
from casmsocial.llm_adapter import LocalBehaviorLLMAdapter
from casmsocial.person import LLMBehaviorEngine, Person, _shared_value

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

    monkeypatch.setattr(casmpop, "_usable_cpu_count", lambda: 3)
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
