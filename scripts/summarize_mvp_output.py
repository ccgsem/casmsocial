"""Create a compact Markdown summary for the MVP behavior log."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow.dataset as ds

from scripts.validate_mvp_output import (
    DEFAULT_BEHAVIOR_LOG_PATH,
    DEFAULT_EXPECTED_AGENTS,
    DEFAULT_EXPECTED_RANKS,
    DEFAULT_EXPECTED_RUNS,
    DEFAULT_EXPECTED_TICKS,
    SIGNAL_COLUMNS,
    validate_behavior_log,
)

DEFAULT_SUMMARY_PATH = Path("data/output/mvp_summary.md")


def _column_values(table: Any, column_name: str) -> list[Any]:
    return cast(list[Any], table[column_name].to_pylist())


def _counts(table: Any, column_name: str) -> dict[str, int]:
    counter = Counter(str(value) for value in _column_values(table, column_name))
    return dict(sorted(counter.items()))


def _mean(values: list[Any]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def build_summary(
    behavior_log_path: Path = DEFAULT_BEHAVIOR_LOG_PATH,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Return a compact summary for the validated MVP behavior log."""
    validation_summary = validate_behavior_log(
        behavior_log_path,
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )
    table = ds.dataset(behavior_log_path, format="parquet", partitioning="hive").to_table()

    applied_values = _column_values(table, "last_plan_adjustment_applied")
    ticks = [int(value) for value in _column_values(table, "tick")]
    run_ids = sorted(str(value) for value in set(_column_values(table, "run_id")))
    random_seeds = sorted(int(value) for value in set(_column_values(table, "random_seed")))

    return {
        "behavior_log": str(behavior_log_path),
        "rows": validation_summary["rows"],
        "runs": validation_summary["runs"],
        "run_ids": run_ids,
        "random_seeds": random_seeds,
        "agents": validation_summary["agents"],
        "ticks": validation_summary["ticks"],
        "ranks": validation_summary["ranks"],
        "tick_min": min(ticks),
        "tick_max": max(ticks),
        "decision_counts": _counts(table, "last_decision"),
        "memory_event_counts": _counts(table, "last_memory_event_type"),
        "plan_adjustments_applied": sum(1 for value in applied_values if bool(value)),
        "signal_averages": {column_name: _mean(_column_values(table, column_name)) for column_name in SIGNAL_COLUMNS},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render an MVP summary as Markdown."""
    lines = [
        "# MVP Summary",
        "",
        f"Behavior log: `{summary['behavior_log']}`",
        "",
        "## Run Shape",
        "",
        f"- Rows: {summary['rows']}",
        f"- Runs: {summary['runs']} ({', '.join(cast(list[str], summary['run_ids']))})",
        f"- Random seeds: {', '.join(str(seed) for seed in cast(list[int], summary['random_seeds']))}",
        f"- Agents: {summary['agents']}",
        f"- Ticks: {summary['ticks']} ({summary['tick_min']} to {summary['tick_max']})",
        f"- Ranks: {summary['ranks']}",
        f"- Plan adjustments applied: {summary['plan_adjustments_applied']}",
        "",
        "## Decisions",
        "",
    ]

    decision_counts = cast(dict[str, int], summary["decision_counts"])
    lines.extend(f"- `{decision}`: {count}" for decision, count in decision_counts.items())

    lines.extend(["", "## Memory Events", ""])
    memory_event_counts = cast(dict[str, int], summary["memory_event_counts"])
    lines.extend(f"- `{event_type}`: {count}" for event_type, count in memory_event_counts.items())

    lines.extend(["", "## Signal Averages", ""])
    signal_averages = cast(dict[str, float], summary["signal_averages"])
    lines.extend(f"- `{signal_name}`: {average:.3f}" for signal_name, average in signal_averages.items())
    lines.append("")

    return "\n".join(lines)


def write_summary_report(
    behavior_log_path: Path = DEFAULT_BEHAVIOR_LOG_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Validate the behavior log and write a Markdown summary report."""
    summary = build_summary(
        behavior_log_path,
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior-log",
        type=Path,
        default=DEFAULT_BEHAVIOR_LOG_PATH,
        help="Hive-partitioned behavior log directory to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Markdown summary report path.",
    )
    parser.add_argument("--expected-agents", type=int, default=DEFAULT_EXPECTED_AGENTS)
    parser.add_argument("--expected-ticks", type=int, default=DEFAULT_EXPECTED_TICKS)
    parser.add_argument("--expected-ranks", type=int, default=DEFAULT_EXPECTED_RANKS)
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--expected-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_summary_report(
        args.behavior_log,
        args.output,
        expected_agents=args.expected_agents,
        expected_ticks=args.expected_ticks,
        expected_ranks=args.expected_ranks,
        expected_runs=args.expected_runs,
        expected_rows=args.expected_rows,
    )
    print(
        "MVP summary written: "
        f"{args.output} "
        f"(rows={summary['rows']}, runs={summary['runs']}, agents={summary['agents']}, "
        f"ticks={summary['ticks']}, ranks={summary['ranks']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
