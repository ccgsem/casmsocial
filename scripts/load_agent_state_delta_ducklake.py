"""Load MVP agent-state delta outputs into DuckLake for analysis."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa

from casmsocial.ducklake_utils import get_ducklake_connection

DEFAULT_DUCKLAKE_PATH = Path("examples/mvp/mvp.ducklake")
DEFAULT_DATABASE_NAME = "insights_ducklake"
DEFAULT_SCHEMA_NAME = "mvp_observability"
DEFAULT_DELTA_LOG_PATH = Path("data/output/mvp_agent_state_delta.parquet")
DEFAULT_AUDIT_LOG_PATH = Path("data/output/mvp_agent_state_delta_audit.parquet")
DEFAULT_RECONSTRUCTED_LOG_PATH = Path("data/output/mvp_agent_state_reconstructed.parquet")
DEFAULT_VALIDATION_REPORT_PATH = Path("data/output/mvp_delta_state_validation.json")

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

VALIDATION_SCHEMA = pa.schema(
    [
        pa.field("report_path", pa.string()),
        pa.field("version", pa.int32()),
        pa.field("generated_at", pa.string()),
        pa.field("agent_log_path", pa.string()),
        pa.field("behavior_log_path", pa.string()),
        pa.field("delta_log_path", pa.string()),
        pa.field("audit_log_path", pa.string()),
        pa.field("reconstructed_log_path", pa.string()),
        pa.field("full_state_rows", pa.int64()),
        pa.field("delta_rows", pa.int64()),
        pa.field("audit_rows", pa.int64()),
        pa.field("delta_plus_audit_rows", pa.int64()),
        pa.field("reconstructed_rows", pa.int64()),
        pa.field("runs", pa.int64()),
        pa.field("matched_rows", pa.int64()),
        pa.field("delta_to_full_ratio", pa.float64()),
        pa.field("delta_plus_audit_to_full_ratio", pa.float64()),
        pa.field("full_to_delta_ratio", pa.float64()),
        pa.field("delta_row_reduction_percent", pa.float64()),
        pa.field("full_reference_bytes", pa.int64()),
        pa.field("delta_log_bytes", pa.int64()),
        pa.field("audit_log_bytes", pa.int64()),
        pa.field("delta_plus_audit_bytes", pa.int64()),
        pa.field("reconstructed_log_bytes", pa.int64()),
        pa.field("delta_plus_audit_to_reconstructed_ratio", pa.float64()),
        pa.field("delta_plus_audit_file_reduction_percent", pa.float64()),
        pa.field("total_agents_evaluated", pa.int64()),
        pa.field("total_agents_changed", pa.int64()),
        pa.field("ticks_with_changes", pa.int64()),
        pa.field("max_agents_changed_per_tick", pa.int64()),
    ]
)

CHANGES_BY_TICK_SCHEMA = pa.schema(
    [
        pa.field("report_path", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("tick", pa.int32()),
        pa.field("agents_evaluated", pa.int64()),
        pa.field("agents_changed", pa.int64()),
        pa.field("ranks", pa.int64()),
    ]
)


class AgentStateDeltaDuckLakeLoadError(ValueError):
    """Raised when delta-state outputs cannot be loaded into DuckLake."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AgentStateDeltaDuckLakeLoadError(message)


def _quote_identifier(identifier: str) -> str:
    _require(IDENTIFIER_PATTERN.match(identifier) is not None, f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _qualified_name(schema_name: str, table_name: str) -> str:
    return f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"


def _has_parquet_files(path: Path) -> bool:
    return path.is_file() and path.suffix == ".parquet" or path.is_dir() and any(path.rglob("*.parquet"))


def _parquet_source(path: Path, label: str) -> str:
    expanded_path = path.expanduser()
    _require(expanded_path.exists(), f"{label} path does not exist: {expanded_path}")
    _require(_has_parquet_files(expanded_path), f"{label} path contains no parquet files: {expanded_path}")
    if expanded_path.is_file():
        return str(expanded_path)
    return str(expanded_path / "**" / "*.parquet")


def _row_count(conn: duckdb.DuckDBPyConnection, qualified_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {qualified_name}").fetchone()
    return int(row[0]) if row is not None else 0


def _create_schema(conn: duckdb.DuckDBPyConnection, schema_name: str) -> None:
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(schema_name)}")


def _load_parquet_table(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema_name: str,
    table_name: str,
    path: Path,
    label: str,
) -> int:
    source = _parquet_source(path, label)
    qualified_name = _qualified_name(schema_name, table_name)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {qualified_name} AS
        SELECT *
        FROM read_parquet(?, hive_partitioning = true, union_by_name = true)
        """,
        [source],
    )
    return _row_count(conn, qualified_name)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _read_validation_report(report_path: Path) -> Mapping[str, Any]:
    expanded_path = report_path.expanduser()
    _require(expanded_path.exists(), f"Validation report path does not exist: {expanded_path}")
    return _mapping(json.loads(expanded_path.read_text(encoding="utf-8")), "Validation report")


def _validation_row(report_path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    paths = _mapping(report.get("paths"), "Validation report paths")
    validation = _mapping(report.get("validation"), "Validation report validation")
    reconstructed = _mapping(validation.get("reconstructed"), "Validation report validation.reconstructed")
    efficiency = _mapping(report.get("efficiency"), "Validation report efficiency")
    rows = _mapping(efficiency.get("rows"), "Validation report efficiency.rows")
    storage = _mapping(efficiency.get("storage"), "Validation report efficiency.storage")
    changes = _mapping(efficiency.get("changes"), "Validation report efficiency.changes")

    return {
        "report_path": str(report_path),
        "version": _int_value(report.get("version")),
        "generated_at": report.get("generated_at"),
        "agent_log_path": paths.get("agent_log"),
        "behavior_log_path": paths.get("behavior_log"),
        "delta_log_path": paths.get("delta_log"),
        "audit_log_path": paths.get("audit_log"),
        "reconstructed_log_path": paths.get("reconstructed_log"),
        "full_state_rows": _int_value(rows.get("full_state_rows")),
        "delta_rows": _int_value(rows.get("delta_rows")),
        "audit_rows": _int_value(rows.get("audit_rows")),
        "delta_plus_audit_rows": _int_value(rows.get("delta_plus_audit_rows")),
        "reconstructed_rows": _int_value(rows.get("reconstructed_rows")),
        "runs": _int_value(reconstructed.get("runs")),
        "matched_rows": _int_value(validation.get("matched_rows")),
        "delta_to_full_ratio": _float_value(rows.get("delta_to_full_ratio")),
        "delta_plus_audit_to_full_ratio": _float_value(rows.get("delta_plus_audit_to_full_ratio")),
        "full_to_delta_ratio": _float_value(rows.get("full_to_delta_ratio")),
        "delta_row_reduction_percent": _float_value(rows.get("delta_row_reduction_percent")),
        "full_reference_bytes": _int_value(storage.get("full_reference_bytes")),
        "delta_log_bytes": _int_value(storage.get("delta_log_bytes")),
        "audit_log_bytes": _int_value(storage.get("audit_log_bytes")),
        "delta_plus_audit_bytes": _int_value(storage.get("delta_plus_audit_bytes")),
        "reconstructed_log_bytes": _int_value(storage.get("reconstructed_log_bytes")),
        "delta_plus_audit_to_reconstructed_ratio": _float_value(storage.get("delta_plus_audit_to_reconstructed_ratio")),
        "delta_plus_audit_file_reduction_percent": _float_value(storage.get("delta_plus_audit_file_reduction_percent")),
        "total_agents_evaluated": _int_value(changes.get("total_agents_evaluated")),
        "total_agents_changed": _int_value(changes.get("total_agents_changed")),
        "ticks_with_changes": _int_value(changes.get("ticks_with_changes")),
        "max_agents_changed_per_tick": _int_value(changes.get("max_agents_changed_per_tick")),
    }


def _changes_by_tick_rows(report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    efficiency = _mapping(report.get("efficiency"), "Validation report efficiency")
    changes = _mapping(efficiency.get("changes"), "Validation report efficiency.changes")
    changed_agents_per_tick = changes.get("changed_agents_per_tick", [])
    _require(isinstance(changed_agents_per_tick, list), "changed_agents_per_tick must be a JSON array")

    rows = []
    for row in changed_agents_per_tick:
        tick_row = _mapping(row, "changed_agents_per_tick row")
        rows.append(
            {
                "report_path": str(report_path),
                "run_id": tick_row.get("run_id"),
                "tick": _int_value(tick_row.get("tick")),
                "agents_evaluated": _int_value(tick_row.get("agents_evaluated")),
                "agents_changed": _int_value(tick_row.get("agents_changed")),
                "ranks": _int_value(tick_row.get("ranks")),
            }
        )
    return rows


def _replace_table_from_arrow(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema_name: str,
    table_name: str,
    table: Any,
) -> int:
    temp_view_name = f"_{schema_name}_{table_name}_input"
    conn.register(temp_view_name, table)
    try:
        qualified_name = _qualified_name(schema_name, table_name)
        conn.execute(f"CREATE OR REPLACE TABLE {qualified_name} AS SELECT * FROM {_quote_identifier(temp_view_name)}")
        return _row_count(conn, qualified_name)
    finally:
        conn.unregister(temp_view_name)


def _load_validation_tables(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema_name: str,
    validation_report_path: Path,
) -> dict[str, int]:
    report = _read_validation_report(validation_report_path)
    validation_table = pa.Table.from_pylist([_validation_row(validation_report_path, report)], schema=VALIDATION_SCHEMA)
    changes_table = pa.Table.from_pylist(
        _changes_by_tick_rows(validation_report_path, report),
        schema=CHANGES_BY_TICK_SCHEMA,
    )

    return {
        "agent_state_delta_validation": _replace_table_from_arrow(
            conn,
            schema_name=schema_name,
            table_name="agent_state_delta_validation",
            table=validation_table,
        ),
        "agent_state_delta_changes_by_tick": _replace_table_from_arrow(
            conn,
            schema_name=schema_name,
            table_name="agent_state_delta_changes_by_tick",
            table=changes_table,
        ),
    }


def load_agent_state_delta_outputs(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    delta_log_path: Path = DEFAULT_DELTA_LOG_PATH,
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
    reconstructed_log_path: Path | None = DEFAULT_RECONSTRUCTED_LOG_PATH,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    """Load delta-state MVP artifacts into queryable DuckDB/DuckLake tables."""
    _create_schema(conn, schema_name)

    table_counts = {
        "agent_state_delta": _load_parquet_table(
            conn,
            schema_name=schema_name,
            table_name="agent_state_delta",
            path=delta_log_path,
            label="Delta agent-state log",
        ),
        "agent_state_delta_audit": _load_parquet_table(
            conn,
            schema_name=schema_name,
            table_name="agent_state_delta_audit",
            path=audit_log_path,
            label="Delta agent-state audit log",
        ),
    }
    if reconstructed_log_path is not None:
        table_counts["agent_state_reconstructed"] = _load_parquet_table(
            conn,
            schema_name=schema_name,
            table_name="agent_state_reconstructed",
            path=reconstructed_log_path,
            label="Reconstructed agent-state log",
        )

    table_counts.update(
        _load_validation_tables(
            conn,
            schema_name=schema_name,
            validation_report_path=validation_report_path,
        )
    )

    return {
        "schema": schema_name,
        "tables": table_counts,
        "paths": {
            "delta_log": str(delta_log_path),
            "audit_log": str(audit_log_path),
            "reconstructed_log": str(reconstructed_log_path) if reconstructed_log_path is not None else None,
            "validation_report": str(validation_report_path),
        },
    }


def load_agent_state_delta_ducklake(
    *,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    database_name: str = DEFAULT_DATABASE_NAME,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    delta_log_path: Path = DEFAULT_DELTA_LOG_PATH,
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
    reconstructed_log_path: Path | None = DEFAULT_RECONSTRUCTED_LOG_PATH,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    """Open the DuckLake catalog and load delta-state MVP artifacts into it."""
    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    try:
        result = load_agent_state_delta_outputs(
            conn,
            schema_name=schema_name,
            delta_log_path=delta_log_path,
            audit_log_path=audit_log_path,
            reconstructed_log_path=reconstructed_log_path,
            validation_report_path=validation_report_path,
        )
    finally:
        conn.close()

    return {"ducklake_path": str(ducklake_path), "database_name": database_name, **result}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ducklake-path", type=Path, default=DEFAULT_DUCKLAKE_PATH)
    parser.add_argument("--database-name", default=DEFAULT_DATABASE_NAME)
    parser.add_argument("--schema", dest="schema_name", default=DEFAULT_SCHEMA_NAME)
    parser.add_argument("--delta-log", type=Path, default=DEFAULT_DELTA_LOG_PATH)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG_PATH)
    parser.add_argument("--reconstructed-log", type=Path, default=DEFAULT_RECONSTRUCTED_LOG_PATH)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT_PATH)
    parser.add_argument(
        "--skip-reconstructed",
        action="store_true",
        help="Load only the changed-state and audit logs, plus the validation summary tables.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reconstructed_log_path = None if args.skip_reconstructed else args.reconstructed_log
    try:
        result = load_agent_state_delta_ducklake(
            ducklake_path=args.ducklake_path,
            database_name=args.database_name,
            schema_name=args.schema_name,
            delta_log_path=args.delta_log,
            audit_log_path=args.audit_log,
            reconstructed_log_path=reconstructed_log_path,
            validation_report_path=args.validation_report,
        )
    except (AgentStateDeltaDuckLakeLoadError, duckdb.Error, json.JSONDecodeError) as exc:
        print(f"Agent state delta DuckLake load failed: {exc}", file=sys.stderr)
        return 1

    counts = ", ".join(f"{table}={rows}" for table, rows in sorted(result["tables"].items()))
    print("Agent state delta DuckLake load complete: " f"{args.ducklake_path} schema={result['schema']} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
