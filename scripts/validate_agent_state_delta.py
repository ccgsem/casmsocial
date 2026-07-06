"""Validate reconstructed agent state delta logs against full MVP logs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pyarrow.dataset as ds

from scripts.reconstruct_agent_state import (
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_DELTA_LOG_PATH,
    DEFAULT_RECONSTRUCTED_LOG_PATH,
    STATE_COLUMNS,
    AgentStateReconstructionError,
    reconstruct_agent_state,
    summarize_reconstruction,
    write_reconstructed_agent_state,
)
from scripts.validate_mvp_output import (
    DEFAULT_EXPECTED_AGENTS,
    DEFAULT_EXPECTED_RANKS,
    DEFAULT_EXPECTED_RUNS,
    DEFAULT_EXPECTED_TICKS,
    MvpValidationError,
    validate_mvp_output,
)

DEFAULT_DELTA_AGENT_LOG_PATH = Path("output/mvp_delta_state_agent_log.parquet")
DEFAULT_DELTA_BEHAVIOR_LOG_PATH = Path("output/mvp_delta_state_behavior_log.parquet")
DEFAULT_DELTA_VALIDATION_REPORT_PATH = Path("output/mvp_delta_state_validation.json")

KEY_COLUMNS: tuple[str, ...] = ("run_id", "tick", "rank", "agent_id")
FLOAT_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "safety_signal",
    "social_signal",
    "obligation_signal",
    "schedule_signal",
    "reply_signal",
)
INTEGER_COLUMNS: tuple[str, ...] = ("place_id", "rank_place_id")
STRING_COLUMNS: tuple[str, ...] = ("last_decision", "last_memory_event_type", "last_plan_adjustment_kind")


class AgentStateDeltaValidationError(ValueError):
    """Raised when reconstructed delta state does not match full logs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AgentStateDeltaValidationError(message)


def _read_hive_dataset(path: Path, label: str) -> Any:
    _require(path.exists(), f"{label} path does not exist: {path}")
    _require(any(path.rglob("*.parquet")), f"{label} path contains no parquet files: {path}")
    return ds.dataset(path, format="parquet", partitioning="hive").to_table()


def _rows(table: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], table.to_pylist())


def _path_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _reduction_percent(reduced_value: int | None, baseline_value: int | None) -> float | None:
    if reduced_value is None or baseline_value is None or baseline_value == 0:
        return None
    return round((1 - (reduced_value / baseline_value)) * 100, 3)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["run_id"]), int(row["tick"]), int(row["rank"]), int(row["agent_id"]))


def _normalise_state(row: dict[str, Any]) -> dict[str, int | float | str]:
    state: dict[str, int | float | str] = {}
    for column_name in FLOAT_COLUMNS:
        state[column_name] = round(float(row[column_name]), 6)
    for column_name in INTEGER_COLUMNS:
        state[column_name] = int(row[column_name])
    for column_name in STRING_COLUMNS:
        state[column_name] = str(row[column_name])
    return state


def _expected_state_rows(
    agent_log_path: Path, behavior_log_path: Path
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    agent_table = _read_hive_dataset(agent_log_path, "Agent log")
    behavior_table = _read_hive_dataset(behavior_log_path, "Behavior log")
    agent_rows = {_row_key(row): row for row in _rows(agent_table)}
    behavior_rows = {_row_key(row): row for row in _rows(behavior_table)}

    _require(
        set(agent_rows) == set(behavior_rows),
        "Agent and behavior logs have different run/tick/rank/agent keys",
    )

    expected_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for key in sorted(agent_rows):
        agent_row = agent_rows[key]
        behavior_row = behavior_rows[key]
        state_row = {
            "x": agent_row["x"],
            "y": agent_row["y"],
            "place_id": agent_row["place_id"],
            "rank_place_id": behavior_row["rank_place_id"],
            "last_decision": behavior_row["last_decision"],
            "last_memory_event_type": behavior_row["last_memory_event_type"],
            "last_plan_adjustment_kind": behavior_row["last_plan_adjustment_kind"],
            "safety_signal": behavior_row["safety_signal"],
            "social_signal": behavior_row["social_signal"],
            "obligation_signal": behavior_row["obligation_signal"],
            "schedule_signal": behavior_row["schedule_signal"],
            "reply_signal": behavior_row["reply_signal"],
        }
        expected_rows[key] = _normalise_state(state_row)
    return expected_rows


def _reconstructed_state_rows(reconstructed_table: Any) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    rows = _rows(reconstructed_table)
    reconstructed_rows = {_row_key(row): row for row in rows}
    _require(len(reconstructed_rows) == len(rows), "Reconstructed state contains duplicate run/tick/rank/agent keys")
    return reconstructed_rows


def _validate_reconstructed_rows(
    expected_rows: dict[tuple[str, int, int, int], dict[str, Any]],
    reconstructed_rows: dict[tuple[str, int, int, int], dict[str, Any]],
) -> None:
    missing_keys = sorted(set(expected_rows) - set(reconstructed_rows))
    extra_keys = sorted(set(reconstructed_rows) - set(expected_rows))
    _require(not missing_keys, f"Reconstructed state is missing rows: {missing_keys[:5]}")
    _require(not extra_keys, f"Reconstructed state has unexpected rows: {extra_keys[:5]}")

    for key in sorted(expected_rows):
        expected_state = expected_rows[key]
        actual_row = reconstructed_rows[key]
        actual_state = _normalise_state(actual_row)
        for column_name in STATE_COLUMNS:
            _require(
                actual_state[column_name] == expected_state[column_name],
                f"State mismatch for run/tick/rank/agent {key}, column {column_name}: "
                f"expected {expected_state[column_name]!r}, found {actual_state[column_name]!r}",
            )
        _require(int(actual_row["source_tick"]) <= key[1], f"source_tick is after tick for row: {key}")


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _change_summary(audit_table: Any) -> dict[str, Any]:
    audit_rows = _rows(audit_table)
    changed_by_tick: dict[tuple[str, int], dict[str, int]] = {}
    for row in audit_rows:
        run_id = str(row["run_id"])
        tick = int(row["tick"])
        tick_summary = changed_by_tick.setdefault(
            (run_id, tick),
            {"agents_evaluated": 0, "agents_changed": 0, "ranks": 0},
        )
        tick_summary["agents_evaluated"] += int(row["agents_evaluated"])
        tick_summary["agents_changed"] += int(row["agents_changed"])
        tick_summary["ranks"] += 1

    per_tick: list[dict[str, Any]] = []
    for run_id, tick in sorted(changed_by_tick):
        per_tick.append({"run_id": run_id, "tick": tick, **changed_by_tick[(run_id, tick)]})

    return {
        "total_agents_evaluated": sum(int(row["agents_evaluated"]) for row in per_tick),
        "total_agents_changed": sum(int(row["agents_changed"]) for row in per_tick),
        "ticks_with_changes": sum(1 for row in per_tick if int(row["agents_changed"]) > 0),
        "max_agents_changed_per_tick": max((int(row["agents_changed"]) for row in per_tick), default=0),
        "changed_agents_per_tick": per_tick,
    }


def _efficiency_summary(
    *,
    full_state_rows: int,
    delta_rows: int,
    audit_rows: int,
    reconstructed_rows: int,
    agent_log_path: Path,
    behavior_log_path: Path,
    delta_log_path: Path,
    audit_log_path: Path,
    reconstructed_output_path: Path | None,
    audit_table: Any,
) -> dict[str, Any]:
    delta_plus_audit_rows = delta_rows + audit_rows
    full_reference_bytes = (_path_size_bytes(agent_log_path) or 0) + (_path_size_bytes(behavior_log_path) or 0)
    delta_log_bytes = _path_size_bytes(delta_log_path)
    audit_log_bytes = _path_size_bytes(audit_log_path)
    delta_plus_audit_bytes = (delta_log_bytes or 0) + (audit_log_bytes or 0)
    reconstructed_bytes = _path_size_bytes(reconstructed_output_path)

    return {
        "rows": {
            "full_state_rows": full_state_rows,
            "delta_rows": delta_rows,
            "audit_rows": audit_rows,
            "delta_plus_audit_rows": delta_plus_audit_rows,
            "reconstructed_rows": reconstructed_rows,
            "delta_to_full_ratio": _ratio(delta_rows, full_state_rows),
            "delta_plus_audit_to_full_ratio": _ratio(delta_plus_audit_rows, full_state_rows),
            "full_to_delta_ratio": _ratio(full_state_rows, delta_rows),
            "delta_row_reduction_percent": _reduction_percent(delta_rows, full_state_rows),
        },
        "storage": {
            "full_reference_bytes": full_reference_bytes,
            "delta_log_bytes": delta_log_bytes,
            "audit_log_bytes": audit_log_bytes,
            "delta_plus_audit_bytes": delta_plus_audit_bytes,
            "reconstructed_log_bytes": reconstructed_bytes,
            "delta_plus_audit_to_reconstructed_ratio": _ratio(delta_plus_audit_bytes, reconstructed_bytes),
            "delta_plus_audit_file_reduction_percent": _reduction_percent(
                delta_plus_audit_bytes,
                reconstructed_bytes,
            ),
        },
        "changes": _change_summary(audit_table),
    }


def validate_agent_state_delta(
    *,
    agent_log_path: Path = DEFAULT_DELTA_AGENT_LOG_PATH,
    behavior_log_path: Path = DEFAULT_DELTA_BEHAVIOR_LOG_PATH,
    delta_log_path: Path = DEFAULT_DELTA_LOG_PATH,
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
    reconstructed_output_path: Path | None = DEFAULT_RECONSTRUCTED_LOG_PATH,
    report_path: Path | None = DEFAULT_DELTA_VALIDATION_REPORT_PATH,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_ticks: int = DEFAULT_EXPECTED_TICKS,
    expected_ranks: int = DEFAULT_EXPECTED_RANKS,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate delta state reconstruction against dense agent and behavior logs."""
    full_log_summary = validate_mvp_output(
        agent_log_path,
        behavior_log_path,
        expected_agents=expected_agents,
        expected_ticks=expected_ticks,
        expected_ranks=expected_ranks,
        expected_runs=expected_runs,
    )
    reconstructed_table = reconstruct_agent_state(delta_log_path, audit_log_path)
    reconstructed_summary = summarize_reconstruction(reconstructed_table)

    expected_rows = _expected_state_rows(agent_log_path, behavior_log_path)
    reconstructed_rows = _reconstructed_state_rows(reconstructed_table)
    _validate_reconstructed_rows(expected_rows, reconstructed_rows)

    delta_table = _read_hive_dataset(delta_log_path, "Delta agent state log")
    audit_table = _read_hive_dataset(audit_log_path, "Delta agent state audit log")
    if reconstructed_output_path is not None:
        write_reconstructed_agent_state(reconstructed_table, reconstructed_output_path, overwrite=overwrite)

    efficiency = _efficiency_summary(
        full_state_rows=len(expected_rows),
        delta_rows=delta_table.num_rows,
        audit_rows=audit_table.num_rows,
        reconstructed_rows=reconstructed_summary["rows"],
        agent_log_path=agent_log_path,
        behavior_log_path=behavior_log_path,
        delta_log_path=delta_log_path,
        audit_log_path=audit_log_path,
        reconstructed_output_path=reconstructed_output_path,
        audit_table=audit_table,
    )

    report = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "agent_log": str(agent_log_path),
            "behavior_log": str(behavior_log_path),
            "delta_log": str(delta_log_path),
            "audit_log": str(audit_log_path),
            "reconstructed_log": str(reconstructed_output_path) if reconstructed_output_path is not None else None,
        },
        "validation": {
            "full_logs": full_log_summary,
            "reconstructed": reconstructed_summary,
            "delta_rows": delta_table.num_rows,
            "audit_rows": audit_table.num_rows,
            "matched_rows": len(reconstructed_rows),
        },
        "efficiency": efficiency,
    }
    if report_path is not None:
        _write_report(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-log", type=Path, default=DEFAULT_DELTA_AGENT_LOG_PATH)
    parser.add_argument("--behavior-log", type=Path, default=DEFAULT_DELTA_BEHAVIOR_LOG_PATH)
    parser.add_argument("--delta-log", type=Path, default=DEFAULT_DELTA_LOG_PATH)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG_PATH)
    parser.add_argument("--reconstructed-output", type=Path, default=DEFAULT_RECONSTRUCTED_LOG_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_DELTA_VALIDATION_REPORT_PATH)
    parser.add_argument("--expected-agents", type=int, default=DEFAULT_EXPECTED_AGENTS)
    parser.add_argument("--expected-ticks", type=int, default=DEFAULT_EXPECTED_TICKS)
    parser.add_argument("--expected-ranks", type=int, default=DEFAULT_EXPECTED_RANKS)
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing reconstructed output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_agent_state_delta(
            agent_log_path=args.agent_log,
            behavior_log_path=args.behavior_log,
            delta_log_path=args.delta_log,
            audit_log_path=args.audit_log,
            reconstructed_output_path=args.reconstructed_output,
            report_path=args.report_output,
            expected_agents=args.expected_agents,
            expected_ticks=args.expected_ticks,
            expected_ranks=args.expected_ranks,
            expected_runs=args.expected_runs,
            overwrite=args.overwrite,
        )
    except (AgentStateDeltaValidationError, AgentStateReconstructionError, MvpValidationError) as exc:
        print(f"Agent state delta validation failed: {exc}", file=sys.stderr)
        return 1

    validation = report["validation"]
    reconstructed = validation["reconstructed"]
    print(
        "Agent state delta valid: "
        f"{args.delta_log} -> {args.reconstructed_output} "
        f"(rows={reconstructed['rows']}, runs={reconstructed['runs']}, agents={reconstructed['agents']}, "
        f"ticks={reconstructed['ticks']}, ranks={reconstructed['ranks']}, "
        f"delta_rows={validation['delta_rows']}, audit_rows={validation['audit_rows']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
