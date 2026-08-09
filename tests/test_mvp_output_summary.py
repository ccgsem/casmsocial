from __future__ import annotations

from scripts.summarize_mvp_output import build_summary, write_summary_report
from tests.test_mvp_output_validation import _write_behavior_log


def test_build_summary_returns_mvp_counts(tmp_path):
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    _write_behavior_log(behavior_log_path)

    summary = build_summary(behavior_log_path)

    assert summary["rows"] == 48
    assert summary["runs"] == 1
    assert summary["run_ids"] == ["seed_42"]
    assert summary["random_seeds"] == [42]
    assert summary["agents"] == 2
    assert summary["ticks"] == 24
    assert summary["ranks"] == 1
    assert summary["tick_min"] == 60
    assert summary["tick_max"] == 1440
    assert summary["decision_counts"] == {"follow_schedule": 48}
    assert summary["memory_event_counts"] == {"llm_proposal": 48}
    assert summary["plan_adjustments_applied"] == 0
    assert summary["signal_averages"]["safety_signal"] == 0.0


def test_build_summary_accepts_expected_two_rank_counts(tmp_path):
    behavior_log_path = tmp_path / "mvp_2rank_behavior_log.parquet"
    _write_behavior_log(behavior_log_path, rank_by_agent={1: 0, 2: 1})

    summary = build_summary(behavior_log_path, expected_ranks=2)

    assert summary["rows"] == 48
    assert summary["runs"] == 1
    assert summary["agents"] == 2
    assert summary["ticks"] == 24
    assert summary["ranks"] == 2


def test_write_summary_report_creates_markdown(tmp_path):
    behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
    summary_path = tmp_path / "mvp_summary.md"
    _write_behavior_log(behavior_log_path)

    write_summary_report(behavior_log_path, summary_path)

    report = summary_path.read_text(encoding="utf-8")
    assert "# MVP Summary" in report
    assert "- Rows: 48" in report
    assert "- Runs: 1 (seed_42)" in report
    assert "- Random seeds: 42" in report
    assert "- `follow_schedule`: 48" in report
    assert "- `safety_signal`: 0.000" in report
