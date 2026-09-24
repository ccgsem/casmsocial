"""Stable agent-log types and non-replayed terminal observer snapshots."""

from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from casmsocial.adapters.runner import _ObservationBridge
from casmsocial.casmpop import AgentLogger

SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("random_seed", pa.int64(), nullable=False),
    pa.field("tick", pa.int32(), nullable=False),
    pa.field("rank", pa.int32(), nullable=False),
    pa.field("agent_id", pa.int64(), nullable=False),
    pa.field("x", pa.float64(), nullable=False),
    pa.field("y", pa.float64(), nullable=False),
    pa.field("place_id", pa.int64(), nullable=False),
])


def make_model(tmp_path):
    person = SimpleNamespace(id=1, pt=SimpleNamespace(x=-87.0, y=41.0), state=SimpleNamespace(place_id=10))
    people = [person]
    model = SimpleNamespace(
        params={
            "simulation.run_id": "test-run",
            "random.seed": 42,
            "observers.output_dir": str(tmp_path),
            "observers.agent_log_file": "agent_log.parquet",
        },
        context=SimpleNamespace(agents=lambda **_: iter(people)),
        comm=SimpleNamespace(Get_rank=lambda: 0),
        cal=SimpleNamespace(tick=60.0),
    )
    return model, people


def test_agent_log_live_and_durable_types_agree(tmp_path):
    model, _ = make_model(tmp_path)
    observer = AgentLogger("test", model)
    observer.on_step(model)
    live = observer.get_output_tables(model)["agent_log"]
    assert live.schema == SCHEMA
    durable = ds.dataset(tmp_path / "agent_log.parquet", format="parquet", partitioning="hive").to_table()
    assert durable.to_pylist() == live.to_pylist()
    for field in SCHEMA:
        assert durable.schema.field(field.name).type == field.type


def test_empty_snapshots_keep_schema_and_do_not_replay_old_people(tmp_path):
    model, people = make_model(tmp_path)
    observer = AgentLogger("test", model)
    person = people.pop()
    observer.on_step(model)
    empty = observer.get_output_tables(model)["agent_log"]
    assert empty.schema == SCHEMA and empty.num_rows == 0
    assert not (tmp_path / "agent_log.parquet").exists()
    people.append(person)
    observer.on_step(model)
    people.clear()
    model.cal.tick = 120.0
    observer.on_step(model)
    empty = observer.get_output_tables(model)["agent_log"]
    assert empty.schema == SCHEMA and empty.num_rows == 0


@pytest.mark.parametrize("tick", [60.5, float("nan"), float("inf"), 2**31, None])
def test_invalid_tick_rejected_before_publication_or_write(tmp_path, tick):
    model, _ = make_model(tmp_path)
    model.cal.tick = tick
    observer = AgentLogger("test", model)
    with pytest.raises((pa.ArrowInvalid, ValueError)):
        observer.on_step(model)
    assert observer.get_output_tables(model) == {}
    assert not (tmp_path / "agent_log.parquet").exists()


@pytest.mark.parametrize("attribute,value", [("id", None), ("id", 1.5), ("place_id", None), ("place_id", 10.5)])
def test_invalid_agent_or_place_identity_is_not_silently_coerced(tmp_path, attribute, value):
    model, people = make_model(tmp_path)
    target = people[0] if attribute == "id" else people[0].state
    setattr(target, attribute, value)
    with pytest.raises((pa.ArrowInvalid, ValueError)):
        AgentLogger("test", model).on_step(model)
    assert not (tmp_path / "agent_log.parquet").exists()


def test_custom_state_columns_are_preserved_and_empty_schema_is_not_guessed(tmp_path):
    model, people = make_model(tmp_path)
    people[0].state.energy = 12.5
    model.params["observers.agent_log_columns"] = ["energy"]
    observer = AgentLogger("test", model)
    person = people.pop()
    observer.on_step(model)
    assert observer.get_output_tables(model) == {}
    people.append(person)
    observer.on_step(model)
    table = observer.get_output_tables(model)["agent_log"]
    assert table.column_names == [*AgentLogger.IDENTITY_SCHEMA.names, "energy"]
    assert table["energy"].to_pylist() == [12.5]
    people.clear()
    observer.on_step(model)
    assert observer.get_output_tables(model)["agent_log"].schema == table.schema
    assert observer.get_output_tables(model)["agent_log"].num_rows == 0


@pytest.mark.parametrize("copy_table", [False, True])
def test_bridge_does_not_replay_last_snapshot_at_end(copy_table):
    sink = Mock()
    bridge = _ObservationBridge(sink)
    table = pa.table({"tick": [60], "value": [1]})
    model = Mock()
    model.get_observer_output_tables.return_value = {"agent_log": table}
    bridge.on_step(model)
    if copy_table:
        model.get_observer_output_tables.return_value = {"agent_log": pa.table(table.to_pydict())}
    bridge.on_end(model)
    bridge.on_end(model)
    sink.publish.assert_called_once_with("agent_log", table)
    sink.flush.assert_called_once()


def test_bridge_preserves_new_terminal_data_and_new_channels():
    sink = Mock()
    bridge = _ObservationBridge(sink)
    first, last = pa.table({"tick": [60]}), pa.table({"tick": [120]})
    final_only = pa.table({"total": [2]})
    model = Mock()
    model.get_observer_output_tables.return_value = {"agent_log": first}
    bridge.on_step(model)
    model.get_observer_output_tables.return_value = {"agent_log": last, "summary": final_only}
    bridge.on_end(model)
    assert [(call.args[0], call.args[1].to_pydict()) for call in sink.publish.call_args_list] == [
        ("agent_log", first.to_pydict()),
        ("agent_log", last.to_pydict()),
        ("summary", final_only.to_pydict()),
    ]


def test_bridge_does_not_deduplicate_identical_batches_from_different_steps():
    sink = Mock()
    bridge = _ObservationBridge(sink)
    model = Mock()
    model.get_observer_output_tables.return_value = {"counts": pa.table({"count": [100]})}
    bridge.on_step(model)
    bridge.on_step(model)
    bridge.on_end(model)
    assert sink.publish.call_count == 2
    assert bridge.tick == 2
    sink.flush.assert_called_once()
