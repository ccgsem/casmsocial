from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from pyarrow.dataset import HivePartitioning

from scripts.load_agent_state_delta_ducklake import (
    AgentStateDeltaDuckLakeLoadError,
    load_agent_state_delta_outputs,
)
from scripts.reconstruct_agent_state import DELTA_SCHEMA, RECONSTRUCTED_SCHEMA
from tests.test_agent_state_reconstruction import AUDIT_SCHEMA


def _write_hive_dataset(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
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


def _delta_row(*, tick: int, agent_id: int, change_mask: str = "__initial__") -> dict[str, object]:
    return {
        "run_id": "seed_42",
        "random_seed": 42,
        "tick": tick,
        "rank": 0,
        "agent_id": agent_id,
        "state_hash": f"{agent_id}" * 64,
        "change_mask": change_mask,
        "x": float(agent_id),
        "y": 0.0,
        "place_id": 100,
        "rank_place_id": 100,
        "last_decision": "follow_schedule",
        "last_memory_event_type": "llm_proposal",
        "last_plan_adjustment_kind": "",
        "safety_signal": 0.0,
        "social_signal": 0.0,
        "obligation_signal": 0.0,
        "schedule_signal": 0.0,
        "reply_signal": 0.0,
    }


def _reconstructed_row(*, tick: int, agent_id: int, source_tick: int) -> dict[str, object]:
    row = _delta_row(tick=source_tick, agent_id=agent_id)
    row["tick"] = tick
    row["source_tick"] = source_tick
    return row


def _write_validation_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": "2026-06-01T00:00:00+00:00",
                "paths": {
                    "agent_log": "output/mvp_delta_state_agent_log.parquet",
                    "behavior_log": "output/mvp_delta_state_behavior_log.parquet",
                    "delta_log": "output/mvp_agent_state_delta.parquet",
                    "audit_log": "output/mvp_agent_state_delta_audit.parquet",
                    "reconstructed_log": "output/mvp_agent_state_reconstructed.parquet",
                },
                "validation": {
                    "delta_rows": 2,
                    "audit_rows": 2,
                    "matched_rows": 4,
                    "reconstructed": {"rows": 4, "runs": 1, "agents": 2, "ticks": 2, "ranks": 1},
                },
                "efficiency": {
                    "rows": {
                        "full_state_rows": 4,
                        "delta_rows": 2,
                        "audit_rows": 2,
                        "delta_plus_audit_rows": 4,
                        "reconstructed_rows": 4,
                        "delta_to_full_ratio": 0.5,
                        "delta_plus_audit_to_full_ratio": 1.0,
                        "full_to_delta_ratio": 2.0,
                        "delta_row_reduction_percent": 50.0,
                    },
                    "storage": {
                        "full_reference_bytes": 1000,
                        "delta_log_bytes": 200,
                        "audit_log_bytes": 120,
                        "delta_plus_audit_bytes": 320,
                        "reconstructed_log_bytes": 800,
                        "delta_plus_audit_to_reconstructed_ratio": 0.4,
                        "delta_plus_audit_file_reduction_percent": 60.0,
                    },
                    "changes": {
                        "total_agents_evaluated": 4,
                        "total_agents_changed": 2,
                        "ticks_with_changes": 1,
                        "max_agents_changed_per_tick": 2,
                        "changed_agents_per_tick": [
                            {"run_id": "seed_42", "tick": 60, "agents_evaluated": 2, "agents_changed": 2, "ranks": 1},
                            {
                                "run_id": "seed_42",
                                "tick": 120,
                                "agents_evaluated": 2,
                                "agents_changed": 0,
                                "ranks": 1,
                            },
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_agent_state_delta_outputs_creates_query_tables(tmp_path: Path) -> None:
    delta_path = tmp_path / "mvp_agent_state_delta.parquet"
    audit_path = tmp_path / "mvp_agent_state_delta_audit.parquet"
    reconstructed_path = tmp_path / "mvp_agent_state_reconstructed.parquet"
    report_path = tmp_path / "mvp_delta_state_validation.json"
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=60, agent_id=1),
            _delta_row(tick=60, agent_id=2),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(
        audit_path,
        [
            {"tick": 60, "rank": 0, "agents_evaluated": 2, "agents_changed": 2},
            {"tick": 120, "rank": 0, "agents_evaluated": 2, "agents_changed": 0},
        ],
        AUDIT_SCHEMA,
    )
    _write_hive_dataset(
        reconstructed_path,
        [
            _reconstructed_row(tick=60, agent_id=1, source_tick=60),
            _reconstructed_row(tick=60, agent_id=2, source_tick=60),
            _reconstructed_row(tick=120, agent_id=1, source_tick=60),
            _reconstructed_row(tick=120, agent_id=2, source_tick=60),
        ],
        RECONSTRUCTED_SCHEMA,
    )
    _write_validation_report(report_path)

    conn = duckdb.connect(":memory:")
    try:
        result = load_agent_state_delta_outputs(
            conn,
            schema_name="mvp_observability",
            delta_log_path=delta_path,
            audit_log_path=audit_path,
            reconstructed_log_path=reconstructed_path,
            validation_report_path=report_path,
        )

        assert result["tables"] == {
            "agent_state_delta": 2,
            "agent_state_delta_audit": 2,
            "agent_state_reconstructed": 4,
            "agent_state_delta_validation": 1,
            "agent_state_delta_changes_by_tick": 2,
        }
        assert conn.execute("""
            SELECT tick, agent_id, change_mask
            FROM mvp_observability.agent_state_delta
            ORDER BY tick, agent_id
            """).fetchall() == [(60, 1, "__initial__"), (60, 2, "__initial__")]
        assert conn.execute("""
            SELECT runs, full_state_rows, delta_rows, delta_row_reduction_percent, total_agents_changed
            FROM mvp_observability.agent_state_delta_validation
            """).fetchone() == (1, 4, 2, 50.0, 2)
        assert conn.execute("""
            SELECT run_id, tick, agents_changed
            FROM mvp_observability.agent_state_delta_changes_by_tick
            ORDER BY run_id, tick
            """).fetchall() == [("seed_42", 60, 2), ("seed_42", 120, 0)]
    finally:
        conn.close()


def test_load_agent_state_delta_outputs_rejects_bad_schema_identifier(tmp_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    try:
        with pytest.raises(AgentStateDeltaDuckLakeLoadError, match="Invalid SQL identifier"):
            load_agent_state_delta_outputs(
                conn,
                schema_name="bad-schema",
                delta_log_path=tmp_path / "missing_delta.parquet",
                audit_log_path=tmp_path / "missing_audit.parquet",
                validation_report_path=tmp_path / "missing_validation.json",
            )
    finally:
        conn.close()
