"""Validate the Wake County Heat local DuckLake deployment path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from scripts.materialize_wake_county_heat_fixture import (
    DEFAULT_DATABASE_NAME,
    DEFAULT_DUCKLAKE_PATH,
    DEFAULT_FIXTURE_PATH,
    MaterializationResult,
    WakeCountyHeatFixtureError,
    materialize_fixture,
    read_fixture_manifest,
)

DEFAULT_CONFIG_PATH = Path("config/casmsocial.yaml")
DEFAULT_OUTPUT_DIR = Path("data/output/wake_county_heat_deployment_smoke")
DEFAULT_AGENT_LOG_FILE = "agent_log.parquet"
DEFAULT_DURATION_HOURS = 1
DEFAULT_RUN_ID = "wake_county_heat_deployment_smoke"


@dataclass(frozen=True)
class AgentLogCounts:
    """Observed row counts from the deployment smoke agent log."""

    row_count: int
    agent_count: int
    tick_count: int


@dataclass(frozen=True)
class DeploymentSmokeResult:
    """Summary of the Wake County Heat deployment smoke validation."""

    materialization: MaterializationResult
    output_dir: Path
    agent_log_path: Path
    agent_log_counts: AgentLogCounts


def expected_table_counts(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> dict[str, int]:
    """Return expected fixture table counts from the manifest."""
    manifest = read_fixture_manifest(fixture_path)
    return {table.name: table.rows for table in manifest.tables}


def assert_expected_table_counts(actual: dict[str, int], expected: dict[str, int]) -> None:
    """Raise when loaded fixture counts differ from manifest counts."""
    if actual != expected:
        raise WakeCountyHeatFixtureError(f"Loaded table counts do not match manifest counts: {actual} != {expected}")


def read_agent_log_counts(agent_log_path: Path, run_id: str = DEFAULT_RUN_ID) -> AgentLogCounts:
    """Read the smoke-run agent log and return rows, agents, and tick counts."""
    if not agent_log_path.exists():
        raise WakeCountyHeatFixtureError(f"Agent log path does not exist: {agent_log_path}")

    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT agent_id) AS agent_count,
                COUNT(DISTINCT tick) AS tick_count
            FROM read_parquet(?, hive_partitioning = true)
            WHERE run_id = ?
            """,
            [str(agent_log_path), run_id],
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise WakeCountyHeatFixtureError(f"Agent log has no readable rows: {agent_log_path}")

    return AgentLogCounts(row_count=int(row[0]), agent_count=int(row[1]), tick_count=int(row[2]))


def assert_expected_agent_log_counts(counts: AgentLogCounts, expected_agents: int) -> None:
    """Raise when the smoke-run agent log does not match the one-hour expectation."""
    expected = AgentLogCounts(row_count=expected_agents, agent_count=expected_agents, tick_count=1)
    if counts != expected:
        raise WakeCountyHeatFixtureError(f"Agent log counts do not match expected smoke output: {counts} != {expected}")


def run_model_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    agent_log_file: str = DEFAULT_AGENT_LOG_FILE,
    duration_hours: int = DEFAULT_DURATION_HOURS,
    run_id: str = DEFAULT_RUN_ID,
    python_executable: str = sys.executable,
) -> Path:
    """Run the shipped config for a short Wake County Heat smoke simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_log_path = output_dir / agent_log_file
    if agent_log_path.is_dir():
        shutil.rmtree(agent_log_path)
    elif agent_log_path.exists():
        agent_log_path.unlink()

    overrides = {
        "duration.hours": duration_hours,
        "simulation.run_id": run_id,
        "observers.output_dir": str(output_dir),
        "observers.agent_log_file": agent_log_file,
        "logging.rank0_only": True,
    }
    env = os.environ.copy()
    env["CASMSOCIAL_DATA_PATH"] = str(ducklake_path.parent)
    env["CASMSOCIAL_DUCKLAKE_PATH"] = str(ducklake_path)

    subprocess.run(
        [python_executable, "-m", "casmsocial", str(config_path), json.dumps(overrides)],
        check=True,
        env=env,
    )
    return agent_log_path


def validate_local_deployment(
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    database_name: str = DEFAULT_DATABASE_NAME,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    agent_log_file: str = DEFAULT_AGENT_LOG_FILE,
    duration_hours: int = DEFAULT_DURATION_HOURS,
    run_id: str = DEFAULT_RUN_ID,
    python_executable: str = sys.executable,
) -> DeploymentSmokeResult:
    """Materialize the fixture, run the default config, and verify smoke output."""
    materialization = materialize_fixture(
        fixture_path=fixture_path,
        ducklake_path=ducklake_path,
        database_name=database_name,
    )
    expected_counts = expected_table_counts(fixture_path)
    assert_expected_table_counts(dict(materialization.tables), expected_counts)

    agent_log_path = run_model_smoke(
        config_path=config_path,
        ducklake_path=ducklake_path,
        output_dir=output_dir,
        agent_log_file=agent_log_file,
        duration_hours=duration_hours,
        run_id=run_id,
        python_executable=python_executable,
    )
    agent_counts = read_agent_log_counts(agent_log_path, run_id=run_id)
    assert_expected_agent_log_counts(agent_counts, expected_agents=expected_counts["persons_1000_households"])

    return DeploymentSmokeResult(
        materialization=materialization,
        output_dir=output_dir,
        agent_log_path=agent_log_path,
        agent_log_counts=agent_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--ducklake-path", type=Path, default=DEFAULT_DUCKLAKE_PATH)
    parser.add_argument("--database-name", default=DEFAULT_DATABASE_NAME)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--agent-log-file", default=DEFAULT_AGENT_LOG_FILE)
    parser.add_argument("--duration-hours", type=int, default=DEFAULT_DURATION_HOURS)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()

    try:
        result = validate_local_deployment(
            fixture_path=args.fixture_path,
            ducklake_path=args.ducklake_path,
            database_name=args.database_name,
            config_path=args.config_path,
            output_dir=args.output_dir,
            agent_log_file=args.agent_log_file,
            duration_hours=args.duration_hours,
            run_id=args.run_id,
            python_executable=args.python_executable,
        )
    except (WakeCountyHeatFixtureError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    table_counts = ", ".join(f"{name}={rows}" for name, rows in result.materialization.tables.items())
    print(
        "Wake County Heat deployment smoke passed: "
        f"ducklake_path={result.materialization.ducklake_path} "
        f"schema={result.materialization.schema_name} ({table_counts}); "
        f"agent_log={result.agent_log_path} "
        f"rows={result.agent_log_counts.row_count} "
        f"agents={result.agent_log_counts.agent_count} "
        f"ticks={result.agent_log_counts.tick_count}"
    )


if __name__ == "__main__":
    main()
