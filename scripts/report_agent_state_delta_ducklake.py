"""Write query examples and a Markdown report for delta-state DuckLake tables."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import duckdb

from casmsocial.ducklake_utils import get_ducklake_connection
from scripts.load_agent_state_delta_ducklake import (
    DEFAULT_DATABASE_NAME,
    DEFAULT_DUCKLAKE_PATH,
    DEFAULT_SCHEMA_NAME,
)

DEFAULT_REPORT_PATH = Path("output/mvp_agent_state_delta_ducklake_report.md")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_TABLES: tuple[str, ...] = (
    "agent_state_delta",
    "agent_state_delta_audit",
    "agent_state_reconstructed",
    "agent_state_delta_validation",
    "agent_state_delta_changes_by_tick",
)


class AgentStateDeltaDuckLakeReportError(ValueError):
    """Raised when delta-state observability tables cannot be reported."""


@dataclass(frozen=True)
class QueryExample:
    title: str
    description: str
    sql: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AgentStateDeltaDuckLakeReportError(message)


def _quote_identifier(identifier: str) -> str:
    _require(IDENTIFIER_PATTERN.match(identifier) is not None, f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _table_name(schema_name: str, table_name: str) -> str:
    return f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"


def _required_tables_present(conn: duckdb.DuckDBPyConnection, schema_name: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT table_name
        FROM duckdb_tables
        WHERE schema_name = ?
          AND table_name IN (SELECT UNNEST(?))
        """,
        [schema_name, list(REQUIRED_TABLES)],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _validate_required_tables(conn: duckdb.DuckDBPyConnection, schema_name: str) -> None:
    present_tables = _required_tables_present(conn, schema_name)
    missing_tables = sorted(set(REQUIRED_TABLES) - present_tables)
    _require(
        not missing_tables,
        f"Missing delta-state observability tables in schema {schema_name!r}: {missing_tables}",
    )


def _fetch_query(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    cursor = conn.execute(sql)
    description = cursor.description or []
    columns = [str(column[0]) for column in description]
    rows = cast(list[tuple[Any, ...]], cursor.fetchall())
    return columns, rows


def _fetch_one_dict(conn: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    columns, rows = _fetch_query(conn, sql)
    _require(len(rows) == 1, f"Expected exactly one row from summary query, found {len(rows)}")
    return dict(zip(columns, rows[0]))


def _query_examples(schema_name: str, sample_limit: int) -> tuple[QueryExample, ...]:
    delta = _table_name(schema_name, "agent_state_delta")
    audit = _table_name(schema_name, "agent_state_delta_audit")
    reconstructed = _table_name(schema_name, "agent_state_reconstructed")
    validation = _table_name(schema_name, "agent_state_delta_validation")
    changes = _table_name(schema_name, "agent_state_delta_changes_by_tick")

    return (
        QueryExample(
            title="Efficiency Snapshot",
            description="Summarize row reduction, storage reduction, and change totals from the validation table.",
            sql=f"""
SELECT
    runs,
    full_state_rows,
    delta_rows,
    audit_rows,
    reconstructed_rows,
    delta_row_reduction_percent,
    delta_plus_audit_file_reduction_percent,
    total_agents_evaluated,
    total_agents_changed,
    ticks_with_changes
FROM {validation}
""".strip(),
        ),
        QueryExample(
            title="Changed Agents By Tick",
            description="Find the ticks where agent state changed and retain zero-change ticks for continuity.",
            sql=f"""
SELECT
    run_id,
    tick,
    agents_evaluated,
    agents_changed,
    ranks
FROM {changes}
ORDER BY run_id, tick
""".strip(),
        ),
        QueryExample(
            title="Audit Rows By Rank",
            description="Inspect the per-rank audit rows that make unchanged ticks visible.",
            sql=f"""
SELECT
    run_id,
    tick,
    rank,
    agents_evaluated,
    agents_changed
FROM {audit}
ORDER BY run_id, tick, rank
LIMIT {sample_limit}
""".strip(),
        ),
        QueryExample(
            title="Most Frequently Changed Agents",
            description="Rank agents by changed-state rows and list the ticks where those changes were written.",
            sql=f"""
SELECT
    run_id,
    agent_id,
    COUNT(*) AS change_rows,
    MIN(tick) AS first_change_tick,
    MAX(tick) AS last_change_tick,
    STRING_AGG(CAST(tick AS VARCHAR), ', ' ORDER BY tick) AS change_ticks
FROM {delta}
GROUP BY run_id, agent_id
ORDER BY change_rows DESC, run_id, agent_id
LIMIT {sample_limit}
""".strip(),
        ),
        QueryExample(
            title="Change Mask Counts",
            description="Group changed rows by change mask to see which state fields drove output volume.",
            sql=f"""
SELECT
    run_id,
    change_mask,
    COUNT(*) AS rows
FROM {delta}
GROUP BY run_id, change_mask
ORDER BY rows DESC, run_id, change_mask
LIMIT {sample_limit}
""".strip(),
        ),
        QueryExample(
            title="Reconstructed State Freshness",
            description="Compare fresh rows to rows carried forward from an earlier source tick.",
            sql=f"""
SELECT
    run_id,
    tick,
    COUNT(*) AS reconstructed_rows,
    SUM(CASE WHEN source_tick = tick THEN 1 ELSE 0 END) AS fresh_rows,
    SUM(CASE WHEN source_tick < tick THEN 1 ELSE 0 END) AS carried_rows,
    COUNT(DISTINCT source_tick) AS distinct_source_ticks
FROM {reconstructed}
GROUP BY run_id, tick
ORDER BY run_id, tick
""".strip(),
        ),
        QueryExample(
            title="Latest Reconstructed Agent State",
            description="Read the latest dense state rows reconstructed from the changed-only log.",
            sql=f"""
WITH latest_tick AS (
    SELECT
        run_id,
        MAX(tick) AS tick
    FROM {reconstructed}
    GROUP BY run_id
)
SELECT
    reconstructed.run_id,
    reconstructed.agent_id,
    reconstructed.tick,
    reconstructed.source_tick,
    reconstructed.place_id,
    reconstructed.last_decision,
    reconstructed.safety_signal,
    reconstructed.social_signal,
    reconstructed.obligation_signal,
    reconstructed.schedule_signal,
    reconstructed.reply_signal
FROM {reconstructed} AS reconstructed
JOIN latest_tick
  ON reconstructed.run_id = latest_tick.run_id
 AND reconstructed.tick = latest_tick.tick
ORDER BY reconstructed.run_id, reconstructed.agent_id
LIMIT {sample_limit}
""".strip(),
        ),
    )


def build_agent_state_delta_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    sample_limit: int = 10,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run query examples against an open DuckDB/DuckLake connection."""
    _require(sample_limit > 0, "sample_limit must be positive")
    _quote_identifier(schema_name)
    _validate_required_tables(conn, schema_name)

    validation_table = _table_name(schema_name, "agent_state_delta_validation")
    summary = _fetch_one_dict(
        conn,
        f"""
        SELECT
            runs,
            full_state_rows,
            delta_rows,
            audit_rows,
            reconstructed_rows,
            delta_row_reduction_percent,
            delta_plus_audit_file_reduction_percent,
            total_agents_evaluated,
            total_agents_changed,
            ticks_with_changes
        FROM {validation_table}
        """,
    )

    query_results = []
    for query in _query_examples(schema_name, sample_limit):
        columns, rows = _fetch_query(conn, query.sql)
        query_results.append(
            {
                "title": query.title,
                "description": query.description,
                "sql": query.sql,
                "columns": columns,
                "rows": rows,
            }
        )

    return {
        "version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "schema": schema_name,
        "sample_limit": sample_limit,
        "summary": summary,
        "queries": query_results,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not columns:
        return ["_No columns returned._"]
    if not rows:
        return ["_No rows returned._"]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render a DuckLake query report as Markdown."""
    summary = cast(dict[str, Any], report["summary"])
    lines = [
        "# Agent State Delta DuckLake Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Schema: `{report['schema']}`",
        "",
        "## Summary",
        "",
        f"- Runs: {summary['runs']}",
        f"- Full state rows: {summary['full_state_rows']}",
        f"- Delta rows: {summary['delta_rows']}",
        f"- Audit rows: {summary['audit_rows']}",
        f"- Reconstructed rows: {summary['reconstructed_rows']}",
        f"- Delta row reduction: {float(summary['delta_row_reduction_percent']):.3f}%",
        ("- Delta plus audit file reduction: " f"{float(summary['delta_plus_audit_file_reduction_percent']):.3f}%"),
        f"- Agents evaluated: {summary['total_agents_evaluated']}",
        f"- Agents changed: {summary['total_agents_changed']}",
        f"- Ticks with changes: {summary['ticks_with_changes']}",
        "",
    ]

    query_results = cast(list[dict[str, Any]], report["queries"])
    for query_result in query_results:
        lines.extend(
            [
                f"## {query_result['title']}",
                "",
                str(query_result["description"]),
                "",
                "```sql",
                str(query_result["sql"]),
                "```",
                "",
            ]
        )
        columns = cast(list[str], query_result["columns"])
        rows = cast(list[tuple[Any, ...]], query_result["rows"])
        lines.extend(_markdown_table(columns, rows))
        lines.append("")

    return "\n".join(lines)


def write_agent_state_delta_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    sample_limit: int = 10,
) -> dict[str, Any]:
    """Write the Markdown query report for an open DuckDB/DuckLake connection."""
    report = build_agent_state_delta_report(conn, schema_name=schema_name, sample_limit=sample_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def write_agent_state_delta_ducklake_report(
    *,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    database_name: str = DEFAULT_DATABASE_NAME,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    output_path: Path = DEFAULT_REPORT_PATH,
    sample_limit: int = 10,
) -> dict[str, Any]:
    """Open a DuckLake catalog and write the delta-state observability report."""
    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    try:
        report = write_agent_state_delta_report(
            conn,
            output_path=output_path,
            schema_name=schema_name,
            sample_limit=sample_limit,
        )
    finally:
        conn.close()
    return {"ducklake_path": str(ducklake_path), "database_name": database_name, **report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ducklake-path", type=Path, default=DEFAULT_DUCKLAKE_PATH)
    parser.add_argument("--database-name", default=DEFAULT_DATABASE_NAME)
    parser.add_argument("--schema", dest="schema_name", default=DEFAULT_SCHEMA_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = write_agent_state_delta_ducklake_report(
            ducklake_path=args.ducklake_path,
            database_name=args.database_name,
            schema_name=args.schema_name,
            output_path=args.output,
            sample_limit=args.sample_limit,
        )
    except (AgentStateDeltaDuckLakeReportError, duckdb.Error) as exc:
        print(f"Agent state delta DuckLake report failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Agent state delta DuckLake report written: "
        f"{args.output} "
        f"(schema={report['schema']}, queries={len(report['queries'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
