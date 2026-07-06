from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from pyarrow.dataset import HivePartitioning

from scripts.reconstruct_agent_state import (
    DELTA_SCHEMA,
    AgentStateReconstructionError,
    reconstruct_agent_state,
    summarize_reconstruction,
    write_reconstructed_agent_state,
)

AUDIT_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("random_seed", pa.int64()),
        pa.field("tick", pa.int32()),
        pa.field("rank", pa.int32()),
        pa.field("agents_evaluated", pa.int64()),
        pa.field("agents_changed", pa.int64()),
    ]
)


def _delta_row(
    *,
    run_id: str = "seed_42",
    random_seed: int = 42,
    tick: int,
    rank: int,
    agent_id: int,
    state_hash: str,
    change_mask: str,
    place_id: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "random_seed": random_seed,
        "tick": tick,
        "rank": rank,
        "agent_id": agent_id,
        "state_hash": state_hash,
        "change_mask": change_mask,
        "x": float(place_id),
        "y": float(place_id + 1),
        "place_id": place_id,
        "rank_place_id": place_id,
        "last_decision": "follow_schedule",
        "last_memory_event_type": "llm_proposal",
        "last_plan_adjustment_kind": "",
        "safety_signal": 0.0,
        "social_signal": 0.0,
        "obligation_signal": 0.0,
        "schedule_signal": 1.0,
        "reply_signal": 0.0,
    }


def _audit_row(
    *,
    run_id: str = "seed_42",
    random_seed: int = 42,
    tick: int,
    rank: int,
    agents_evaluated: int,
    agents_changed: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "random_seed": random_seed,
        "tick": tick,
        "rank": rank,
        "agents_evaluated": agents_evaluated,
        "agents_changed": agents_changed,
    }


def _write_hive_dataset(path, rows, schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    partition_schema = pa.schema(
        [pa.field("run_id", pa.string()), pa.field("tick", pa.int32()), pa.field("rank", pa.int32())]
    )
    ds.write_dataset(
        data=table,
        base_dir=path,
        format="parquet",
        partitioning=HivePartitioning(partition_schema),
    )


def test_reconstruct_agent_state_forward_fills_changed_rows(tmp_path):
    delta_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=10, rank=0, agent_id=1, state_hash="a" * 64, change_mask="__initial__", place_id=100),
            _delta_row(tick=10, rank=1, agent_id=2, state_hash="b" * 64, change_mask="__initial__", place_id=200),
            _delta_row(tick=20, rank=0, agent_id=1, state_hash="c" * 64, change_mask="place_id", place_id=300),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(
        audit_path,
        [
            _audit_row(tick=10, rank=0, agents_evaluated=1, agents_changed=1),
            _audit_row(tick=10, rank=1, agents_evaluated=1, agents_changed=1),
            _audit_row(tick=20, rank=0, agents_evaluated=1, agents_changed=1),
            _audit_row(tick=20, rank=1, agents_evaluated=1, agents_changed=0),
            _audit_row(tick=30, rank=0, agents_evaluated=1, agents_changed=0),
            _audit_row(tick=30, rank=1, agents_evaluated=1, agents_changed=0),
        ],
        AUDIT_SCHEMA,
    )

    table = reconstruct_agent_state(delta_path, audit_path)
    rows = sorted(table.to_pylist(), key=lambda row: (row["tick"], row["agent_id"]))

    assert summarize_reconstruction(table) == {"rows": 6, "runs": 1, "agents": 2, "ticks": 3, "ranks": 2}
    assert [(row["run_id"], row["tick"], row["agent_id"], row["place_id"], row["source_tick"]) for row in rows] == [
        ("seed_42", 10, 1, 100, 10),
        ("seed_42", 10, 2, 200, 10),
        ("seed_42", 20, 1, 300, 20),
        ("seed_42", 20, 2, 200, 10),
        ("seed_42", 30, 1, 300, 20),
        ("seed_42", 30, 2, 200, 10),
    ]


def test_reconstruct_agent_state_accepts_empty_audit_ticks_without_delta_log(tmp_path):
    delta_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    _write_hive_dataset(
        audit_path,
        [
            _audit_row(tick=10, rank=0, agents_evaluated=0, agents_changed=0),
            _audit_row(tick=20, rank=0, agents_evaluated=0, agents_changed=0),
        ],
        AUDIT_SCHEMA,
    )

    table = reconstruct_agent_state(delta_path, audit_path)

    assert table.num_rows == 0
    assert summarize_reconstruction(table) == {"rows": 0, "runs": 0, "agents": 0, "ticks": 0, "ranks": 0}


def test_reconstruct_agent_state_rejects_audit_change_count_mismatch(tmp_path):
    delta_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=10, rank=0, agent_id=1, state_hash="a" * 64, change_mask="__initial__", place_id=100),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(
        audit_path,
        [
            _audit_row(tick=10, rank=0, agents_evaluated=1, agents_changed=0),
        ],
        AUDIT_SCHEMA,
    )

    with pytest.raises(AgentStateReconstructionError, match="Audit changed count mismatch"):
        reconstruct_agent_state(delta_path, audit_path)


def test_reconstruct_agent_state_rejects_audit_agent_count_mismatch(tmp_path):
    delta_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=10, rank=0, agent_id=1, state_hash="a" * 64, change_mask="__initial__", place_id=100),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(
        audit_path,
        [
            _audit_row(tick=10, rank=0, agents_evaluated=2, agents_changed=1),
        ],
        AUDIT_SCHEMA,
    )

    with pytest.raises(AgentStateReconstructionError, match="Audit evaluated count mismatch"):
        reconstruct_agent_state(delta_path, audit_path)


def test_write_reconstructed_agent_state_writes_hive_dataset(tmp_path):
    delta_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    output_path = tmp_path / "agent_state_reconstructed.parquet"
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=10, rank=0, agent_id=1, state_hash="a" * 64, change_mask="__initial__", place_id=100),
            _delta_row(tick=20, rank=0, agent_id=1, state_hash="b" * 64, change_mask="place_id", place_id=200),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(
        audit_path,
        [
            _audit_row(tick=10, rank=0, agents_evaluated=1, agents_changed=1),
            _audit_row(tick=20, rank=0, agents_evaluated=1, agents_changed=1),
        ],
        AUDIT_SCHEMA,
    )

    table = reconstruct_agent_state(delta_path, audit_path)
    write_reconstructed_agent_state(table, output_path)

    rows = sorted(
        ds.dataset(output_path, format="parquet", partitioning="hive").to_table().to_pylist(),
        key=lambda row: row["tick"],
    )
    assert [(row["run_id"], row["tick"], row["rank"], row["agent_id"], row["place_id"]) for row in rows] == [
        ("seed_42", 10, 0, 1, 100),
        ("seed_42", 20, 0, 1, 200),
    ]
