from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from scripts.report_agent_state_delta_ducklake import (
    AgentStateDeltaDuckLakeReportError,
    build_agent_state_delta_report,
    render_markdown,
    write_agent_state_delta_report,
)


def _create_observability_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA mvp_observability")
    conn.execute("""
        CREATE TABLE mvp_observability.agent_state_delta_validation AS
        SELECT
            1::BIGINT AS runs,
            4::BIGINT AS full_state_rows,
            2::BIGINT AS delta_rows,
            2::BIGINT AS audit_rows,
            4::BIGINT AS reconstructed_rows,
            50.0::DOUBLE AS delta_row_reduction_percent,
            60.0::DOUBLE AS delta_plus_audit_file_reduction_percent,
            4::BIGINT AS total_agents_evaluated,
            2::BIGINT AS total_agents_changed,
            1::BIGINT AS ticks_with_changes
        """)
    conn.execute("""
        CREATE TABLE mvp_observability.agent_state_delta_changes_by_tick (
            run_id VARCHAR,
            tick INTEGER,
            agents_evaluated BIGINT,
            agents_changed BIGINT,
            ranks BIGINT
        )
        """)
    conn.execute("""
        INSERT INTO mvp_observability.agent_state_delta_changes_by_tick VALUES
            ('seed_42', 60, 2, 2, 1),
            ('seed_42', 120, 2, 0, 1)
        """)
    conn.execute("""
        CREATE TABLE mvp_observability.agent_state_delta (
            run_id VARCHAR,
            tick INTEGER,
            rank INTEGER,
            agent_id BIGINT,
            change_mask VARCHAR
        )
        """)
    conn.execute("""
        INSERT INTO mvp_observability.agent_state_delta VALUES
            ('seed_42', 60, 0, 1, '__initial__'),
            ('seed_42', 60, 0, 2, '__initial__')
        """)
    conn.execute("""
        CREATE TABLE mvp_observability.agent_state_delta_audit (
            run_id VARCHAR,
            tick INTEGER,
            rank INTEGER,
            agents_evaluated BIGINT,
            agents_changed BIGINT
        )
        """)
    conn.execute("""
        INSERT INTO mvp_observability.agent_state_delta_audit VALUES
            ('seed_42', 60, 0, 2, 2),
            ('seed_42', 120, 0, 2, 0)
        """)
    conn.execute("""
        CREATE TABLE mvp_observability.agent_state_reconstructed (
            run_id VARCHAR,
            tick INTEGER,
            agent_id BIGINT,
            source_tick INTEGER,
            place_id BIGINT,
            last_decision VARCHAR,
            safety_signal DOUBLE,
            social_signal DOUBLE,
            obligation_signal DOUBLE,
            schedule_signal DOUBLE,
            reply_signal DOUBLE
        )
        """)
    conn.execute("""
        INSERT INTO mvp_observability.agent_state_reconstructed VALUES
            ('seed_42', 60, 1, 60, 100, 'follow_schedule', 0.0, 0.0, 0.0, 0.0, 0.0),
            ('seed_42', 60, 2, 60, 100, 'follow_schedule', 0.0, 0.0, 0.0, 0.0, 0.0),
            ('seed_42', 120, 1, 60, 100, 'follow_schedule', 0.0, 0.0, 0.0, 0.0, 0.0),
            ('seed_42', 120, 2, 60, 100, 'follow_schedule', 0.0, 0.0, 0.0, 0.0, 0.0)
        """)


def test_build_agent_state_delta_report_runs_query_examples() -> None:
    conn = duckdb.connect(":memory:")
    try:
        _create_observability_tables(conn)

        report = build_agent_state_delta_report(
            conn,
            schema_name="mvp_observability",
            sample_limit=3,
            generated_at="2026-06-01T00:00:00+00:00",
        )
        markdown = render_markdown(report)

        assert report["summary"]["delta_rows"] == 2
        assert len(report["queries"]) == 7
        assert "## Changed Agents By Tick" in markdown
        assert 'FROM "mvp_observability"."agent_state_delta_changes_by_tick"' in markdown
        assert "| run_id | tick | agents_evaluated | agents_changed | ranks |" in markdown
        assert "| seed_42 | 120 | 2 | 0 | 1 |" in markdown
        assert "Runs: 1" in markdown
        assert "Delta row reduction: 50.000%" in markdown
    finally:
        conn.close()


def test_write_agent_state_delta_report_writes_markdown(tmp_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    try:
        _create_observability_tables(conn)
        output_path = tmp_path / "mvp_agent_state_delta_ducklake_report.md"

        report = write_agent_state_delta_report(
            conn,
            output_path=output_path,
            schema_name="mvp_observability",
            sample_limit=2,
        )

        assert output_path.exists()
        assert report["sample_limit"] == 2
        assert "# Agent State Delta DuckLake Report" in output_path.read_text(encoding="utf-8")
    finally:
        conn.close()


def test_build_agent_state_delta_report_rejects_missing_tables() -> None:
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE SCHEMA mvp_observability")

        with pytest.raises(AgentStateDeltaDuckLakeReportError, match="Missing delta-state observability tables"):
            build_agent_state_delta_report(conn, schema_name="mvp_observability")
    finally:
        conn.close()
