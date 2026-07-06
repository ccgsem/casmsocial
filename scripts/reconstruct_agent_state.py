"""Reconstruct dense agent state from changed-only delta logs."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.dataset import HivePartitioning

DEFAULT_DELTA_LOG_PATH = Path("output/agent_state_delta.parquet")
DEFAULT_AUDIT_LOG_PATH = Path("output/agent_state_delta_audit.parquet")
DEFAULT_RECONSTRUCTED_LOG_PATH = Path("output/agent_state_reconstructed.parquet")

STATE_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "place_id",
    "rank_place_id",
    "last_decision",
    "last_memory_event_type",
    "last_plan_adjustment_kind",
    "safety_signal",
    "social_signal",
    "obligation_signal",
    "schedule_signal",
    "reply_signal",
)

REQUIRED_DELTA_COLUMNS: tuple[str, ...] = (
    "run_id",
    "random_seed",
    "tick",
    "rank",
    "agent_id",
    "state_hash",
    "change_mask",
    *STATE_COLUMNS,
)

REQUIRED_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "random_seed",
    "tick",
    "rank",
    "agents_evaluated",
    "agents_changed",
)

DELTA_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("random_seed", pa.int64()),
        pa.field("tick", pa.int32()),
        pa.field("rank", pa.int32()),
        pa.field("agent_id", pa.int64()),
        pa.field("state_hash", pa.string()),
        pa.field("change_mask", pa.string()),
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("place_id", pa.int64()),
        pa.field("rank_place_id", pa.int64()),
        pa.field("last_decision", pa.string()),
        pa.field("last_memory_event_type", pa.string()),
        pa.field("last_plan_adjustment_kind", pa.string()),
        pa.field("safety_signal", pa.float64()),
        pa.field("social_signal", pa.float64()),
        pa.field("obligation_signal", pa.float64()),
        pa.field("schedule_signal", pa.float64()),
        pa.field("reply_signal", pa.float64()),
    ]
)

RECONSTRUCTED_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("random_seed", pa.int64()),
        pa.field("tick", pa.int32()),
        pa.field("rank", pa.int32()),
        pa.field("agent_id", pa.int64()),
        pa.field("source_tick", pa.int32()),
        pa.field("state_hash", pa.string()),
        pa.field("change_mask", pa.string()),
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("place_id", pa.int64()),
        pa.field("rank_place_id", pa.int64()),
        pa.field("last_decision", pa.string()),
        pa.field("last_memory_event_type", pa.string()),
        pa.field("last_plan_adjustment_kind", pa.string()),
        pa.field("safety_signal", pa.float64()),
        pa.field("social_signal", pa.float64()),
        pa.field("obligation_signal", pa.float64()),
        pa.field("schedule_signal", pa.float64()),
        pa.field("reply_signal", pa.float64()),
    ]
)


class AgentStateReconstructionError(ValueError):
    """Raised when delta state logs cannot be reconstructed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AgentStateReconstructionError(message)


def _has_parquet_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def _empty_delta_table() -> Any:
    return pa.Table.from_pylist([], schema=DELTA_SCHEMA)


def _read_hive_dataset(path: Path, label: str, *, allow_missing: bool = False) -> Any:
    if allow_missing and not _has_parquet_files(path):
        return _empty_delta_table()

    _require(path.exists(), f"{label} path does not exist: {path}")
    _require(_has_parquet_files(path), f"{label} path contains no parquet files: {path}")
    return ds.dataset(path, format="parquet", partitioning="hive").to_table()


def _validate_required_columns(table: Any, required_columns: tuple[str, ...], label: str) -> None:
    missing_columns = sorted(set(required_columns) - set(table.column_names))
    _require(not missing_columns, f"{label} is missing required columns: {missing_columns}")

    for column_name in required_columns:
        _require(table[column_name].null_count == 0, f"{label} column has null values: {column_name}")


def _rows(table: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], table.to_pylist())


def _audit_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["run_id"]), int(row["tick"]), int(row["rank"]))


def _validate_audit_rows(audit_rows: list[dict[str, Any]]) -> None:
    seen_keys: set[tuple[str, int, int]] = set()
    for row in audit_rows:
        key = _audit_key(row)
        _require(key not in seen_keys, f"Duplicate audit row for run/tick/rank: {key}")
        seen_keys.add(key)
        agents_evaluated = int(row["agents_evaluated"])
        agents_changed = int(row["agents_changed"])
        _require(agents_evaluated >= 0, f"Negative agents_evaluated for run/tick/rank: {key}")
        _require(agents_changed >= 0, f"Negative agents_changed for run/tick/rank: {key}")
        _require(
            agents_changed <= agents_evaluated,
            f"agents_changed exceeds agents_evaluated for run/tick/rank: {key}",
        )


def _validate_delta_rows(delta_rows: list[dict[str, Any]]) -> None:
    seen_keys: set[tuple[str, int, int]] = set()
    for row in delta_rows:
        key = (str(row["run_id"]), int(row["tick"]), int(row["agent_id"]))
        _require(key not in seen_keys, f"Duplicate delta row for run/tick/agent: {key}")
        seen_keys.add(key)
        _require(str(row["state_hash"]).strip() != "", f"Empty state_hash for run/tick/agent: {key}")
        _require(str(row["change_mask"]).strip() != "", f"Empty change_mask for run/tick/agent: {key}")


def _validate_audit_change_counts(
    audit_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    audit_changed_by_key = {_audit_key(row): int(row["agents_changed"]) for row in audit_rows}
    delta_changed_by_key = Counter((str(row["run_id"]), int(row["tick"]), int(row["rank"])) for row in delta_rows)

    extra_delta_keys = sorted(set(delta_changed_by_key) - set(audit_changed_by_key))
    _require(not extra_delta_keys, f"Delta rows exist without audit rows: {extra_delta_keys}")

    for key, expected_changed in sorted(audit_changed_by_key.items()):
        actual_changed = delta_changed_by_key.get(key, 0)
        _require(
            actual_changed == expected_changed,
            f"Audit changed count mismatch for run/tick/rank {key}: "
            f"expected {expected_changed}, found {actual_changed}",
        )


def _state_from_delta_row(row: dict[str, Any]) -> dict[str, Any]:
    state = {
        "run_id": str(row["run_id"]),
        "random_seed": int(row["random_seed"]),
        "rank": int(row["rank"]),
        "agent_id": int(row["agent_id"]),
        "source_tick": int(row["tick"]),
        "state_hash": str(row["state_hash"]),
        "change_mask": str(row["change_mask"]),
    }
    state.update({column_name: row[column_name] for column_name in STATE_COLUMNS})
    return state


def _reconstruct_rows(
    audit_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reconstructed_rows: list[dict[str, Any]] = []
    for run_id in sorted({str(row["run_id"]) for row in audit_rows}):
        audit_ticks = sorted({int(row["tick"]) for row in audit_rows if str(row["run_id"]) == run_id})
        sorted_delta_rows = sorted(
            (row for row in delta_rows if str(row["run_id"]) == run_id),
            key=lambda row: (int(row["tick"]), int(row["rank"]), int(row["agent_id"])),
        )
        current_state_by_agent_id: dict[int, dict[str, Any]] = {}
        delta_index = 0

        for tick in audit_ticks:
            while delta_index < len(sorted_delta_rows) and int(sorted_delta_rows[delta_index]["tick"]) <= tick:
                delta_row = sorted_delta_rows[delta_index]
                current_state_by_agent_id[int(delta_row["agent_id"])] = _state_from_delta_row(delta_row)
                delta_index += 1

            for agent_id in sorted(current_state_by_agent_id):
                reconstructed_row = {"tick": tick, **current_state_by_agent_id[agent_id]}
                reconstructed_rows.append(reconstructed_row)

    return reconstructed_rows


def _validate_audit_agent_counts(
    audit_rows: list[dict[str, Any]],
    reconstructed_rows: list[dict[str, Any]],
) -> None:
    reconstructed_by_key = Counter(
        (str(row["run_id"]), int(row["tick"]), int(row["rank"])) for row in reconstructed_rows
    )
    for audit_row in sorted(
        audit_rows,
        key=lambda row: (str(row["run_id"]), int(row["tick"]), int(row["rank"])),
    ):
        key = _audit_key(audit_row)
        expected_evaluated = int(audit_row["agents_evaluated"])
        actual_evaluated = reconstructed_by_key.get(key, 0)
        _require(
            actual_evaluated == expected_evaluated,
            f"Audit evaluated count mismatch for run/tick/rank {key}: "
            f"expected {expected_evaluated}, reconstructed {actual_evaluated}",
        )


def reconstruct_agent_state(
    delta_log_path: Path = DEFAULT_DELTA_LOG_PATH,
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH,
) -> Any:
    """Reconstruct dense agent state rows from delta and audit logs."""
    delta_log_path = delta_log_path.expanduser()
    audit_log_path = audit_log_path.expanduser()

    delta_table = _read_hive_dataset(delta_log_path, "Delta agent state log", allow_missing=True)
    audit_table = _read_hive_dataset(audit_log_path, "Delta agent state audit log")

    _validate_required_columns(delta_table, REQUIRED_DELTA_COLUMNS, "Delta agent state log")
    _validate_required_columns(audit_table, REQUIRED_AUDIT_COLUMNS, "Delta agent state audit log")

    delta_rows = _rows(delta_table)
    audit_rows = _rows(audit_table)
    _validate_audit_rows(audit_rows)
    _validate_delta_rows(delta_rows)
    _validate_audit_change_counts(audit_rows, delta_rows)

    reconstructed_rows = _reconstruct_rows(audit_rows, delta_rows)
    _validate_audit_agent_counts(audit_rows, reconstructed_rows)
    return pa.Table.from_pylist(reconstructed_rows, schema=RECONSTRUCTED_SCHEMA)


def write_reconstructed_agent_state(
    table: Any,
    output_path: Path = DEFAULT_RECONSTRUCTED_LOG_PATH,
    *,
    overwrite: bool = False,
) -> None:
    """Write reconstructed rows as a Hive-partitioned Parquet dataset."""
    output_path = output_path.expanduser()
    if output_path.exists():
        _require(overwrite, f"Output path already exists: {output_path}. Pass --overwrite to replace it.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if table.num_rows == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        return

    partition_schema = pa.schema(
        [pa.field("run_id", pa.string()), pa.field("tick", pa.int32()), pa.field("rank", pa.int32())]
    )
    ds.write_dataset(
        data=table,
        base_dir=output_path,
        format="parquet",
        partitioning=HivePartitioning(partition_schema),
    )


def summarize_reconstruction(table: Any) -> dict[str, int]:
    """Return a compact summary for reconstructed agent state rows."""
    if table.num_rows == 0:
        return {"rows": 0, "runs": 0, "agents": 0, "ticks": 0, "ranks": 0}

    rows = _rows(table)
    return {
        "rows": table.num_rows,
        "runs": len({str(row["run_id"]) for row in rows}),
        "agents": len({int(row["agent_id"]) for row in rows}),
        "ticks": len({int(row["tick"]) for row in rows}),
        "ranks": len({int(row["rank"]) for row in rows}),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-log",
        type=Path,
        default=DEFAULT_DELTA_LOG_PATH,
        help="Hive-partitioned changed-only agent state log directory.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=DEFAULT_AUDIT_LOG_PATH,
        help="Hive-partitioned delta agent state audit log directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Hive-partitioned Parquet directory for reconstructed dense state.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        table = reconstruct_agent_state(args.delta_log, args.audit_log)
        if args.output is not None:
            write_reconstructed_agent_state(table, args.output, overwrite=args.overwrite)
    except AgentStateReconstructionError as exc:
        print(f"Agent state reconstruction failed: {exc}", file=sys.stderr)
        return 1

    summary = summarize_reconstruction(table)
    destination = f" -> {args.output}" if args.output is not None else ""
    print(
        "Agent state reconstruction valid: "
        f"{args.delta_log}{destination} "
        f"(rows={summary['rows']}, runs={summary['runs']}, agents={summary['agents']}, ticks={summary['ticks']}, "
        f"ranks={summary['ranks']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
