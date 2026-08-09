from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from pyarrow.dataset import HivePartitioning

from scripts.reconstruct_agent_state import DELTA_SCHEMA
from scripts.validate_agent_state_delta import AgentStateDeltaValidationError, validate_agent_state_delta
from tests.test_agent_state_reconstruction import AUDIT_SCHEMA
from tests.test_mvp_output_validation import _write_agent_log, _write_behavior_log


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


def _delta_row(*, tick: int, agent_id: int, place_id: int = 100) -> dict[str, object]:
    return {
        "run_id": "seed_42",
        "random_seed": 42,
        "tick": tick,
        "rank": 0,
        "agent_id": agent_id,
        "state_hash": f"{agent_id}" * 64,
        "change_mask": "__initial__",
        "x": 0.0,
        "y": 0.0,
        "place_id": place_id,
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


def _audit_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(1, 25):
        rows.append(
            {
                "run_id": "seed_42",
                "random_seed": 42,
                "tick": 60 * index,
                "rank": 0,
                "agents_evaluated": 2,
                "agents_changed": 2 if index == 1 else 0,
            }
        )
    return rows


def _write_delta_logs(delta_path, audit_path, *, mismatched_place_id: int | None = None) -> None:
    _write_hive_dataset(
        delta_path,
        [
            _delta_row(tick=60, agent_id=1, place_id=mismatched_place_id or 100),
            _delta_row(tick=60, agent_id=2),
        ],
        DELTA_SCHEMA,
    )
    _write_hive_dataset(audit_path, _audit_rows(), AUDIT_SCHEMA)


def test_validate_agent_state_delta_accepts_reconstructed_state(tmp_path):
    agent_log_path = tmp_path / "mvp_delta_state_agent_log.parquet"
    behavior_log_path = tmp_path / "mvp_delta_state_behavior_log.parquet"
    delta_path = tmp_path / "mvp_agent_state_delta.parquet"
    audit_path = tmp_path / "mvp_agent_state_delta_audit.parquet"
    reconstructed_path = tmp_path / "mvp_agent_state_reconstructed.parquet"
    report_path = tmp_path / "mvp_delta_state_validation.json"
    _write_agent_log(agent_log_path)
    _write_behavior_log(behavior_log_path)
    _write_delta_logs(delta_path, audit_path)

    report = validate_agent_state_delta(
        agent_log_path=agent_log_path,
        behavior_log_path=behavior_log_path,
        delta_log_path=delta_path,
        audit_log_path=audit_path,
        reconstructed_output_path=reconstructed_path,
        report_path=report_path,
    )

    assert report["validation"]["full_logs"]["agent"] == {
        "rows": 48,
        "runs": 1,
        "agents": 2,
        "ticks": 24,
        "ranks": 1,
    }
    assert report["validation"]["reconstructed"] == {"rows": 48, "runs": 1, "agents": 2, "ticks": 24, "ranks": 1}
    assert report["validation"]["delta_rows"] == 2
    assert report["validation"]["audit_rows"] == 24
    assert report["efficiency"]["rows"]["full_state_rows"] == 48
    assert report["efficiency"]["rows"]["delta_rows"] == 2
    assert report["efficiency"]["rows"]["delta_to_full_ratio"] == 0.041667
    assert report["efficiency"]["rows"]["delta_row_reduction_percent"] == 95.833
    assert report["efficiency"]["changes"]["total_agents_evaluated"] == 48
    assert report["efficiency"]["changes"]["total_agents_changed"] == 2
    assert report["efficiency"]["changes"]["ticks_with_changes"] == 1
    assert report["efficiency"]["changes"]["changed_agents_per_tick"][0] == {
        "run_id": "seed_42",
        "agents_changed": 2,
        "agents_evaluated": 2,
        "ranks": 1,
        "tick": 60,
    }
    assert report["efficiency"]["storage"]["delta_plus_audit_bytes"] > 0
    assert reconstructed_path.exists()
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert written_report["validation"]["matched_rows"] == 48
    assert written_report["efficiency"]["rows"]["delta_plus_audit_rows"] == 26


def test_validate_agent_state_delta_rejects_state_mismatch(tmp_path):
    agent_log_path = tmp_path / "mvp_delta_state_agent_log.parquet"
    behavior_log_path = tmp_path / "mvp_delta_state_behavior_log.parquet"
    delta_path = tmp_path / "mvp_agent_state_delta.parquet"
    audit_path = tmp_path / "mvp_agent_state_delta_audit.parquet"
    _write_agent_log(agent_log_path)
    _write_behavior_log(behavior_log_path)
    _write_delta_logs(delta_path, audit_path, mismatched_place_id=999)

    with pytest.raises(AgentStateDeltaValidationError, match="State mismatch"):
        validate_agent_state_delta(
            agent_log_path=agent_log_path,
            behavior_log_path=behavior_log_path,
            delta_log_path=delta_path,
            audit_log_path=audit_path,
            reconstructed_output_path=None,
            report_path=None,
        )
