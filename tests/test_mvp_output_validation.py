from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from pyarrow.dataset import HivePartitioning

from scripts.validate_mvp_output import (
    MvpValidationError,
    validate_agent_log,
    validate_behavior_log,
    validate_mvp_output,
)


def _write_agent_log(
    path,
    *,
    omit_column: str | None = None,
    rows_per_agent: int = 24,
    rank_by_agent: dict[int, int] | None = None,
    run_ids: tuple[str, ...] = ("seed_42",),
    random_seed_by_run: dict[str, int] | None = None,
) -> None:
    rows = []
    ticks = [60 * index for index in range(1, rows_per_agent + 1)]
    for run_id in run_ids:
        random_seed = random_seed_by_run.get(run_id, 42) if random_seed_by_run is not None else 42
        for tick in ticks:
            for agent_id in (1, 2):
                rows.append(
                    {
                        "run_id": run_id,
                        "random_seed": random_seed,
                        "tick": tick,
                        "rank": rank_by_agent.get(agent_id, 0) if rank_by_agent is not None else 0,
                        "agent_id": agent_id,
                        "x": 0.0,
                        "y": 0.0,
                        "place_id": 100,
                    }
                )

    if omit_column is not None:
        for row in rows:
            del row[omit_column]

    table = pa.Table.from_pylist(rows)
    partition_schema = pa.schema(
        [pa.field("run_id", pa.string()), pa.field("tick", pa.int32()), pa.field("rank", pa.int32())]
    )
    ds.write_dataset(
        data=table,
        base_dir=path,
        format="parquet",
        partitioning=HivePartitioning(partition_schema),
    )


def _write_behavior_log(
    path,
    *,
    omit_column: str | None = None,
    rows_per_agent: int = 24,
    rank_by_agent: dict[int, int] | None = None,
    run_ids: tuple[str, ...] = ("seed_42",),
    random_seed_by_run: dict[str, int] | None = None,
) -> None:
    rows = []
    ticks = [60 * index for index in range(1, rows_per_agent + 1)]
    for run_id in run_ids:
        random_seed = random_seed_by_run.get(run_id, 42) if random_seed_by_run is not None else 42
        for tick in ticks:
            for agent_id in (1, 2):
                rows.append(
                    {
                        "run_id": run_id,
                        "random_seed": random_seed,
                        "tick": tick,
                        "rank": rank_by_agent.get(agent_id, 0) if rank_by_agent is not None else 0,
                        "agent_id": agent_id,
                        "place_id": 100,
                        "rank_place_id": 100,
                        "last_decision": "follow_schedule",
                        "last_llm_summary": "No recent messages; following schedule.",
                        "last_memory_event_type": "llm_proposal",
                        "last_plan_adjustment_requested_kind": "",
                        "last_plan_adjustment_applied": False,
                        "last_plan_adjustment_skip_reason": "",
                        "last_plan_adjustment_kind": "",
                        "last_plan_adjustment_delay_minutes": 0,
                        "last_plan_adjustment_target_activity_id": -1,
                        "last_plan_adjustment_target_place_id": 0,
                        "safety_signal": 0.0,
                        "social_signal": 0.0,
                        "obligation_signal": 0.0,
                        "schedule_signal": 0.0,
                        "reply_signal": 0.0,
                    }
                )

    if omit_column is not None:
        for row in rows:
            del row[omit_column]

    table = pa.Table.from_pylist(rows)
    partition_schema = pa.schema(
        [pa.field("run_id", pa.string()), pa.field("tick", pa.int32()), pa.field("rank", pa.int32())]
    )
    ds.write_dataset(
        data=table,
        base_dir=path,
        format="parquet",
        partitioning=HivePartitioning(partition_schema),
    )


def test_validate_behavior_log_accepts_mvp_contract(tmp_path):
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    _write_behavior_log(behavior_log_path)

    summary = validate_behavior_log(behavior_log_path)

    assert summary == {"rows": 48, "runs": 1, "agents": 2, "ticks": 24, "ranks": 1}


def test_validate_behavior_log_rejects_missing_required_column(tmp_path):
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    _write_behavior_log(behavior_log_path, omit_column="reply_signal")

    with pytest.raises(MvpValidationError, match="missing required columns"):
        validate_behavior_log(behavior_log_path)


def test_validate_behavior_log_rejects_unexpected_row_count(tmp_path):
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    _write_behavior_log(behavior_log_path, rows_per_agent=23)

    with pytest.raises(MvpValidationError, match="Expected 48 behavior rows"):
        validate_behavior_log(behavior_log_path)


def test_validate_agent_log_accepts_mvp_contract(tmp_path):
    agent_log_path = tmp_path / "mvp_agent_log.parquet"
    _write_agent_log(agent_log_path)

    summary = validate_agent_log(agent_log_path)

    assert summary == {"rows": 48, "runs": 1, "agents": 2, "ticks": 24, "ranks": 1}


def test_validate_agent_log_rejects_missing_required_column(tmp_path):
    agent_log_path = tmp_path / "mvp_agent_log.parquet"
    _write_agent_log(agent_log_path, omit_column="x")

    with pytest.raises(MvpValidationError, match="missing required columns"):
        validate_agent_log(agent_log_path)


def test_validate_agent_log_rejects_unexpected_row_count(tmp_path):
    agent_log_path = tmp_path / "mvp_agent_log.parquet"
    _write_agent_log(agent_log_path, rows_per_agent=23)

    with pytest.raises(MvpValidationError, match="Expected 48 agent rows"):
        validate_agent_log(agent_log_path)


def test_validate_mvp_output_accepts_both_logs(tmp_path):
    agent_log_path = tmp_path / "mvp_agent_log.parquet"
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    _write_agent_log(agent_log_path)
    _write_behavior_log(behavior_log_path)

    summary = validate_mvp_output(agent_log_path, behavior_log_path)

    assert summary == {
        "agent": {"rows": 48, "runs": 1, "agents": 2, "ticks": 24, "ranks": 1},
        "behavior": {"rows": 48, "runs": 1, "agents": 2, "ticks": 24, "ranks": 1},
    }


def test_validate_mvp_output_accepts_multiple_run_partitions(tmp_path):
    agent_log_path = tmp_path / "mvp_agent_log.parquet"
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    run_ids = ("seed_42", "seed_99")
    random_seed_by_run = {"seed_42": 42, "seed_99": 99}
    _write_agent_log(agent_log_path, run_ids=run_ids, random_seed_by_run=random_seed_by_run)
    _write_behavior_log(behavior_log_path, run_ids=run_ids, random_seed_by_run=random_seed_by_run)

    summary = validate_mvp_output(agent_log_path, behavior_log_path, expected_runs=2)

    assert summary == {
        "agent": {"rows": 96, "runs": 2, "agents": 2, "ticks": 24, "ranks": 1},
        "behavior": {"rows": 96, "runs": 2, "agents": 2, "ticks": 24, "ranks": 1},
    }
