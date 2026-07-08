"""Validate the output logs produced by the MVP scenario."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.dataset as ds

DEFAULT_AGENT_LOG_PATH = Path("data/output/mvp_agent_log.parquet")
DEFAULT_BEHAVIOR_LOG_PATH = Path("data/output/mvp_behavior_log.parquet")
DEFAULT_EXPECTED_AGENTS = 2
DEFAULT_EXPECTED_TICKS = 24
DEFAULT_EXPECTED_RANKS = 1
DEFAULT_EXPECTED_RUNS = 1

REQUIRED_AGENT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "random_seed",
    "tick",
    "rank",
    "agent_id",
    "x",
    "y",
    "place_id",
)

REQUIRED_BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "run_id",
    "random_seed",
    "tick",
    "rank",
    "agent_id",
    "place_id",
    "rank_place_id",
    "last_decision",
    "last_llm_summary",
    "last_memory_event_type",
    "last_plan_adjustment_requested_kind",
    "last_plan_adjustment_applied",
    "last_plan_adjustment_skip_reason",
    "last_plan_adjustment_kind",
    "last_plan_adjustment_delay_minutes",
    "last_plan_adjustment_target_activity_id",
    "last_plan_adjustment_target_place_id",
    "safety_signal",
    "social_signal",
    "obligation_signal",
    "schedule_signal",
    "reply_signal",
)

AGENT_INTEGER_COLUMNS: tuple[str, ...] = (
    "random_seed",
    "tick",
    "rank",
    "agent_id",
    "place_id",
)

AGENT_FLOAT_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
)

BEHAVIOR_INTEGER_COLUMNS: tuple[str, ...] = (
    "random_seed",
    "tick",
    "rank",
    "agent_id",
    "place_id",
    "rank_place_id",
    "last_plan_adjustment_delay_minutes",
    "last_plan_adjustment_target_activity_id",
    "last_plan_adjustment_target_place_id",
)

STRING_COLUMNS: tuple[str, ...] = (
    "last_decision",
    "last_llm_summary",
    "last_memory_event_type",
    "last_plan_adjustment_requested_kind",
    "last_plan_adjustment_skip_reason",
    "last_plan_adjustment_kind",
)

SIGNAL_COLUMNS: tuple[str, ...] = (
    "safety_signal",
    "social_signal",
    "obligation_signal",
    "schedule_signal",
    "reply_signal",
)


class MvpValidationError(ValueError):
    """Raised when MVP output does not match the expected artifact contract."""


def _column_values(table: Any, column_name: str) -> list[Any]:
    return cast(list[Any], table[column_name].to_pylist())


def _unique_values(table: Any, column_name: str) -> set[Any]:
    return set(_column_values(table, column_name))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MvpValidationError(message)


def _read_hive_dataset(path: Path, label: str) -> Any:
    _require(path.exists(), f"{label} path does not exist: {path}")
    _require(any(path.rglob("*.parquet")), f"{label} path contains no parquet files: {path}")
    return ds.dataset(path, format="parquet", partitioning="hive").to_table()


def _is_string_like(data_type: Any) -> bool:
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return True
    return bool(
        pa.types.is_dictionary(data_type)
        and (pa.types.is_string(data_type.value_type) or pa.types.is_large_string(data_type.value_type))
    )


def _validate_required_columns(table: Any, required_columns: tuple[str, ...], label: str) -> None:
    missing_columns = sorted(set(required_columns) - set(table.column_names))
    _require(not missing_columns, f"{label} is missing required columns: {missing_columns}")


def _validate_non_null_columns(table: Any, required_columns: tuple[str, ...], label: str) -> None:
    for column_name in required_columns:
        column = table[column_name]
        _require(column.null_count == 0, f"{label} column has null values: {column_name}")


def _validate_agent_schema(table: Any) -> None:
    _validate_required_columns(table, REQUIRED_AGENT_COLUMNS, "Agent log")
    _validate_non_null_columns(table, REQUIRED_AGENT_COLUMNS, "Agent log")
    _require(_is_string_like(table["run_id"].type), "Expected string agent log column: run_id")

    for column_name in AGENT_INTEGER_COLUMNS:
        _require(pa.types.is_integer(table[column_name].type), f"Expected integer agent log column: {column_name}")

    for column_name in AGENT_FLOAT_COLUMNS:
        _require(
            pa.types.is_floating(table[column_name].type), f"Expected floating-point agent log column: {column_name}"
        )


def _validate_behavior_schema(table: Any) -> None:
    _validate_required_columns(table, REQUIRED_BEHAVIOR_COLUMNS, "Behavior log")
    _validate_non_null_columns(table, REQUIRED_BEHAVIOR_COLUMNS, "Behavior log")
    _require(_is_string_like(table["run_id"].type), "Expected string behavior log column: run_id")

    for column_name in BEHAVIOR_INTEGER_COLUMNS:
        _require(pa.types.is_integer(table[column_name].type), f"Expected integer column: {column_name}")

    for column_name in STRING_COLUMNS:
        _require(_is_string_like(table[column_name].type), f"Expected string column: {column_name}")

    _require(
        pa.types.is_boolean(table["last_plan_adjustment_applied"].type),
        "Expected boolean column: last_plan_adjustment_applied",
    )

    for column_name in SIGNAL_COLUMNS:
        _require(pa.types.is_floating(table[column_name].type), f"Expected floating-point signal column: {column_name}")


def _validate_log_shape(
    table: Any,
    label: str,
    *,
    expected_agents: int,
    expected_ticks: int,
    expected_ranks: int,
    expected_runs: int,
    expected_rows: int | None,
) -> dict[str, int]:
    expected_row_count = (
        expected_rows if expected_rows is not None else expected_agents * expected_ticks * expected_runs
    )
    _require(
        table.num_rows == expected_row_count,
        f"Expected {expected_row_count} {label} rows, found {table.num_rows}",
    )

    runs = {str(value) for value in _column_values(table, "run_id")}
    agents = _unique_values(table, "agent_id")
    ticks = _unique_values(table, "tick")
    ranks = _unique_values(table, "rank")

    _require(all(run_id.strip() for run_id in runs), f"{label.title()} log contains empty run_id values")
    _require(len(runs) == expected_runs, f"Expected {expected_runs} {label} runs, found {len(runs)}")
    _require(len(agents) == expected_agents, f"Expected {expected_agents} {label} agents, found {len(agents)}")
    _require(len(ticks) == expected_ticks, f"Expected {expected_ticks} {label} ticks, found {len(ticks)}")
    _require(len(ranks) == expected_ranks, f"Expected {expected_ranks} {label} ranks, found {len(ranks)}")

    rows = cast(list[dict[str, Any]], table.to_pylist())
    tick_counts = Counter((str(row["run_id"]), int(row["tick"])) for row in rows)
    bad_tick_counts = {key: count for key, count in tick_counts.items() if count != expected_agents}
    _require(
        not bad_tick_counts,
        f"Expected {expected_agents} {label} rows per run/tick, found mismatches: {bad_tick_counts}",
    )

    agent_counts = Counter((str(row["run_id"]), int(row["agent_id"])) for row in rows)
    bad_agent_counts = {key: count for key, count in agent_counts.items() if count != expected_ticks}
    _require(
        not bad_agent_counts,
        f"Expected {expected_ticks} {label} rows per run/agent, found mismatches: {bad_agent_counts}",
    )

    return {
        "rows": table.num_rows,
        "runs": len(runs),
        "agents": len(agents),
        "ticks": len(ticks),
        "ranks": len(ranks),
    }


def validate_agent_log(
    agent_log_path: Path = DEFAULT_AGENT_LOG_PATH,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_rows: int | None = None,
) -> dict[str, int]:
    """Validate the MVP agent log and return a compact summary."""
    table = _read_hive_dataset(agent_log_path, "Agent log")
    _validate_agent_schema(table)
    return _validate_log_shape(
        table,
        "agent",
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )


def validate_behavior_log(
    behavior_log_path: Path = DEFAULT_BEHAVIOR_LOG_PATH,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_rows: int | None = None,
) -> dict[str, int]:
    """Validate the MVP behavior log and return a compact summary."""
    table = _read_hive_dataset(behavior_log_path, "Behavior log")
    _validate_behavior_schema(table)
    summary = _validate_log_shape(
        table,
        "behavior",
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )

    empty_decisions = [value for value in _column_values(table, "last_decision") if not str(value).strip()]
    _require(not empty_decisions, "Behavior log contains empty last_decision values")

    return summary


def validate_mvp_output(
    agent_log_path: Path = DEFAULT_AGENT_LOG_PATH,
    behavior_log_path: Path = DEFAULT_BEHAVIOR_LOG_PATH,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_rows: int | None = None,
) -> dict[str, dict[str, int]]:
    """Validate both MVP logs and return compact summaries."""
    agent_summary = validate_agent_log(
        agent_log_path,
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )
    behavior_summary = validate_behavior_log(
        behavior_log_path,
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )
    return {"agent": agent_summary, "behavior": behavior_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-log",
        type=Path,
        default=DEFAULT_AGENT_LOG_PATH,
        help="Hive-partitioned agent log directory to validate.",
    )
    parser.add_argument(
        "--behavior-log",
        type=Path,
        default=DEFAULT_BEHAVIOR_LOG_PATH,
        help="Hive-partitioned behavior log directory to validate.",
    )
    parser.add_argument("--expected-agents", type=int, default=DEFAULT_EXPECTED_AGENTS)
    parser.add_argument("--expected-ticks", type=int, default=DEFAULT_EXPECTED_TICKS)
    parser.add_argument("--expected-ranks", type=int, default=DEFAULT_EXPECTED_RANKS)
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Only print validation failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_mvp_output(
            args.agent_log,
            args.behavior_log,
            expected_agents=args.expected_agents,
            expected_ticks=args.expected_ticks,
            expected_ranks=args.expected_ranks,
            expected_runs=args.expected_runs,
            expected_rows=args.expected_rows,
        )
    except MvpValidationError as exc:
        print(f"MVP output validation failed: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        agent_summary = summary["agent"]
        print(
            "MVP agent log valid: "
            f"{args.agent_log} "
            f"(rows={agent_summary['rows']}, runs={agent_summary['runs']}, agents={agent_summary['agents']}, "
            f"ticks={agent_summary['ticks']}, ranks={agent_summary['ranks']})"
        )
        behavior_summary = summary["behavior"]
        print(
            "MVP behavior log valid: "
            f"{args.behavior_log} "
            f"(rows={behavior_summary['rows']}, runs={behavior_summary['runs']}, "
            f"agents={behavior_summary['agents']}, "
            f"ticks={behavior_summary['ticks']}, ranks={behavior_summary['ranks']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
