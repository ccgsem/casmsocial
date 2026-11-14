"""``person_profile_priors`` loading and hand-off to the LLM adapter.

Covers the opt-in ``person_profile_priors.table`` key: rows reach each local
person's cognition, then the adapter context as ``profile_priors``; people
without rows are counted, and ``behavior.profile_priors.required`` stops
start-up. Migration of the priors between ranks is covered by
``test_person_migration_mpi``.
"""

from __future__ import annotations

from collections import namedtuple
from datetime import datetime

import duckdb
import pytest

from casmsocial.activities import Act, make_plan
from casmsocial.casmpop import CasmPop, MissingProfilePriorsError, MissingRequiredTableError
from casmsocial.person import BehaviorEngineV2, LLMBehaviorEngine, Person, ScheduleBehaviorEngine
from casmsocial.sim_time import SimTime

_Places = namedtuple("_Places", ["home", "work"])


class _SumComm:
    """Single-rank stand-in for ``allreduce(op=SUM)``."""

    def allreduce(self, value, op=None):
        return value


class _Context:
    def __init__(self, people):
        self.people = people

    def agents(self, agent_type=None):
        return iter(self.people)


def _person(person_id: int, engine=LLMBehaviorEngine) -> Person:
    plan = make_plan([Act(person_id, 0, 0, 0, 60, 100)])
    person = Person(person_id, 0, [plan], _Places(100, 200), {"sp_id": person_id})
    person.behavior_engine = engine(person)
    return person


def _model(people, params=None, priors_rows=None, engine=LLMBehaviorEngine):
    Person.registerBehaviorEngine(engine)
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.comm = _SumComm()
    model.context = _Context(people)
    model.params = {"person_profile_priors.table": "pkg.person_profile_priors", **(params or {})}
    model.conn = duckdb.connect(":memory:")
    model.conn.execute("CREATE TEMPORARY TABLE persons AS SELECT * FROM range(1, 4) t(sp_id)")
    model.conn.execute("CREATE SCHEMA pkg")
    model.conn.execute("""
        CREATE TABLE pkg.person_profile_priors (
            sp_persons_id BIGINT, profile_id VARCHAR, prior_name VARCHAR, prior_value DOUBLE,
            prior_payload JSON, uncertainty DOUBLE, provenance_refs VARCHAR
        )
    """)
    for row in priors_rows or []:
        model.conn.execute("INSERT INTO pkg.person_profile_priors VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    return model


@pytest.fixture(autouse=True)
def _restore_registered_engine():
    previous = Person.getBehaviorEngine()
    yield
    Person.registerBehaviorEngine(previous)


ROWS = [
    (1, "pod-a", "risk_sensitivity", 0.8, '{"persona_value": 0.6}', 0.1, "ref"),
    (1, "pod-a", "authority_trust", 0.3, '{"persona_value": 0.5}', None, "ref"),
    (2, "pod-b", "risk_sensitivity", 0.2, '{"persona_value": 0.4}', None, "ref"),
    (99, "pod-z", "risk_sensitivity", 0.5, "{}", None, "ref"),  # not in persons: never read
]


def test_priors_attach_to_local_people_in_adapter_shape():
    people = [_person(1), _person(2)]
    model = _model(people, priors_rows=ROWS)

    model.load_profile_priors()

    priors = people[0].behavior_engine.cognition.profile_priors
    assert [row["prior_name"] for row in priors] == ["authority_trust", "risk_sensitivity"]
    assert priors[1] == {
        "profile_id": "pod-a",
        "prior_name": "risk_sensitivity",
        "prior_value": 0.8,
        "prior_payload": '{"persona_value": 0.6}',
        "uncertainty": 0.1,
    }
    assert people[1].behavior_engine.cognition.profile_priors[0]["profile_id"] == "pod-b"
    assert model.profile_priors_missing == 0


def test_people_without_priors_are_counted():
    people = [_person(1), _person(3)]
    model = _model(people, priors_rows=ROWS)

    model.load_profile_priors()

    assert people[1].behavior_engine.cognition.profile_priors == ()
    assert model.profile_priors_missing == 1


def test_required_priors_stop_start_up_when_any_person_lacks_them():
    model = _model([_person(1), _person(3)], params={"behavior.profile_priors.required": True}, priors_rows=ROWS)

    with pytest.raises(MissingProfilePriorsError, match="1 people have no rows"):
        model.load_profile_priors()


def test_no_table_key_leaves_priors_empty():
    people = [_person(1)]
    model = _model(people, priors_rows=ROWS)
    del model.params["person_profile_priors.table"]

    model.load_profile_priors()

    assert people[0].behavior_engine.cognition.profile_priors == ()
    assert model.profile_priors_missing == 0


def test_missing_table_is_an_error():
    model = _model([_person(1)], params={"person_profile_priors.table": "pkg.nope"})

    with pytest.raises(MissingRequiredTableError):
        model.load_profile_priors()


def test_schedule_engine_skips_priors_unless_required():
    model = _model([], priors_rows=ROWS, engine=ScheduleBehaviorEngine)
    model.load_profile_priors()
    assert model.profile_priors_missing == 0

    model.params["behavior.profile_priors.required"] = True
    with pytest.raises(MissingProfilePriorsError, match="no cognition"):
        model.load_profile_priors()


def test_v2_engine_holds_priors_too():
    people = [_person(2, BehaviorEngineV2)]
    model = _model(people, priors_rows=ROWS, engine=BehaviorEngineV2)

    model.load_profile_priors()

    assert people[0].behavior_engine.cognition.profile_priors[0]["profile_id"] == "pod-b"


def test_adapter_context_carries_profile_priors():
    person = _person(1)
    priors = ({"profile_id": "pod-a", "prior_name": "risk_sensitivity", "prior_value": 0.8},)
    person.behavior_engine.cognition.profile_priors = priors
    engine = person.behavior_engine

    context = engine._build_llm_context(OBSERVATION, _sim_time())

    assert context["profile_priors"] is priors


def test_adapter_context_priors_default_to_empty():
    engine = _person(1).behavior_engine

    assert engine._build_llm_context(OBSERVATION, _sim_time())["profile_priors"] == ()


def test_behavior_summary_totals_failures_and_missing_priors():
    people = [_person(1), _person(2)]
    people[0].behavior_engine.cognition.llm_failures = 3
    model = _model(people)
    model.profile_priors_missing = 2

    model._log_behavior_summary()

    assert model.behavior_summary == {"llm_failures": 3, "profile_priors_missing": 2}


OBSERVATION = {"incoming_messages": 0.0, "hour_of_day": 0.0, "is_weekday": 1.0, "is_transit": 0.0, "pending_acks": 0.0}


def _sim_time() -> SimTime:
    cal = SimTime(datetime(2025, 1, 1))
    cal.set_datetime(datetime(2025, 1, 1, 0, 30))
    return cal
