from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from scripts import validate_wake_county_heat_deployment as deployment
from scripts.materialize_wake_county_heat_fixture import WakeCountyHeatFixtureError


def _write_agent_log(path: Path) -> None:
    table = pa.Table.from_pylist([
        {"run_id": deployment.DEFAULT_RUN_ID, "tick": 60, "rank": 0, "agent_id": 1},
        {"run_id": deployment.DEFAULT_RUN_ID, "tick": 60, "rank": 0, "agent_id": 2},
        {"run_id": "other_run", "tick": 60, "rank": 0, "agent_id": 99},
    ])
    ds.write_dataset(
        table,
        base_dir=path,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([
                ("run_id", pa.string()),
                ("tick", pa.int64()),
                ("rank", pa.int64()),
            ]),
            flavor="hive",
        ),
    )


def test_read_agent_log_counts_filters_to_smoke_run_id(tmp_path: Path) -> None:
    agent_log_path = tmp_path / "agent_log.parquet"
    _write_agent_log(agent_log_path)

    counts = deployment.read_agent_log_counts(agent_log_path)

    assert counts == deployment.AgentLogCounts(row_count=2, agent_count=2, tick_count=1)


def test_assert_expected_agent_log_counts_rejects_unexpected_rows() -> None:
    counts = deployment.AgentLogCounts(row_count=3, agent_count=2, tick_count=1)

    with pytest.raises(WakeCountyHeatFixtureError, match="Agent log counts do not match"):
        deployment.assert_expected_agent_log_counts(counts, expected_agents=2)


def test_assert_expected_table_counts_rejects_manifest_mismatch() -> None:
    with pytest.raises(WakeCountyHeatFixtureError, match="Loaded table counts do not match"):
        deployment.assert_expected_table_counts({"persons_1000_households": 1}, {"persons_1000_households": 2})
