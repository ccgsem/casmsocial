"""Write a machine-readable manifest for generated MVP artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.validate_mvp_output import (
    DEFAULT_EXPECTED_AGENTS,
    DEFAULT_EXPECTED_RUNS,
    DEFAULT_EXPECTED_TICKS,
    validate_mvp_output,
)

DEFAULT_MANIFEST_PATH = Path("data/output/mvp_manifest.json")


@dataclass(frozen=True)
class MvpRunSpec:
    name: str
    summary_path: Path
    agent_log_path: Path
    behavior_log_path: Path
    expected_ranks: int
    validation_report_path: Path | None = None
    extra_artifact_paths: tuple[Path, ...] = ()


DEFAULT_RUN_SPECS: tuple[MvpRunSpec, ...] = (
    MvpRunSpec(
        name="single_rank",
        summary_path=Path("data/output/mvp_summary.md"),
        agent_log_path=Path("data/output/mvp_agent_log.parquet"),
        behavior_log_path=Path("data/output/mvp_behavior_log.parquet"),
        expected_ranks=1,
    ),
    MvpRunSpec(
        name="two_rank",
        summary_path=Path("data/output/mvp_2rank_summary.md"),
        agent_log_path=Path("data/output/mvp_2rank_agent_log.parquet"),
        behavior_log_path=Path("data/output/mvp_2rank_behavior_log.parquet"),
        expected_ranks=2,
    ),
    MvpRunSpec(
        name="routed",
        summary_path=Path("data/output/mvp_routed_summary.md"),
        agent_log_path=Path("data/output/mvp_routed_agent_log.parquet"),
        behavior_log_path=Path("data/output/mvp_routed_behavior_log.parquet"),
        expected_ranks=1,
        validation_report_path=Path("data/output/mvp_routed_plan_validation.json"),
    ),
    MvpRunSpec(
        name="built_roads",
        summary_path=Path("data/output/mvp_built_roads_summary.md"),
        agent_log_path=Path("data/output/mvp_built_roads_agent_log.parquet"),
        behavior_log_path=Path("data/output/mvp_built_roads_behavior_log.parquet"),
        expected_ranks=1,
        validation_report_path=Path("data/output/mvp_built_roads_plan_validation.json"),
        extra_artifact_paths=(
            Path("data/output/mvp_built_road_nodes.parquet"),
            Path("data/output/mvp_built_road_edges.parquet"),
            Path("data/output/mvp_built_place_road_snap.parquet"),
            Path("data/output/mvp_built_road_artifacts.json"),
        ),
    ),
    MvpRunSpec(
        name="delta_state",
        summary_path=Path("data/output/mvp_delta_state_summary.md"),
        agent_log_path=Path("data/output/mvp_delta_state_agent_log.parquet"),
        behavior_log_path=Path("data/output/mvp_delta_state_behavior_log.parquet"),
        expected_ranks=1,
        validation_report_path=Path("data/output/mvp_delta_state_validation.json"),
        extra_artifact_paths=(
            Path("data/output/mvp_agent_state_delta.parquet"),
            Path("data/output/mvp_agent_state_delta_audit.parquet"),
            Path("data/output/mvp_agent_state_reconstructed.parquet"),
            Path("data/output/mvp_agent_state_delta_ducklake_report.md"),
        ),
    ),
)


def mvp_artifact_paths(
    run_specs: tuple[MvpRunSpec, ...] = DEFAULT_RUN_SPECS,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    include_manifest: bool = True,
) -> tuple[Path, ...]:
    """Return the MVP artifacts that should be retained or uploaded."""
    paths: list[Path] = []
    if include_manifest:
        paths.append(manifest_path)

    for run_spec in run_specs:
        paths.extend((run_spec.summary_path, run_spec.agent_log_path, run_spec.behavior_log_path))
        if run_spec.validation_report_path is not None:
            paths.append(run_spec.validation_report_path)
        paths.extend(run_spec.extra_artifact_paths)

    unique_paths = dict.fromkeys(paths)
    return tuple(unique_paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
    digest = hashlib.sha256()
    total_size = 0
    parquet_file_count = 0
    for file_path in files:
        file_size = file_path.stat().st_size
        file_digest = sha256_file(file_path)
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(f"{relative_path}\0{file_size}\0{file_digest}\n".encode())
        total_size += file_size
        if file_path.suffix == ".parquet":
            parquet_file_count += 1

    return {
        "path": str(path),
        "kind": "directory",
        "file_count": len(files),
        "parquet_file_count": parquet_file_count,
        "size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def build_mvp_manifest(
    run_specs: tuple[MvpRunSpec, ...] = DEFAULT_RUN_SPECS,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Validate MVP outputs and return a manifest with artifact metadata."""
    runs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for run_spec in run_specs:
        validation = validate_mvp_output(
            run_spec.agent_log_path,
            run_spec.behavior_log_path,
            expected_agents=DEFAULT_EXPECTED_AGENTS,
            expected_ticks=DEFAULT_EXPECTED_TICKS,
            expected_ranks=run_spec.expected_ranks,
            expected_runs=DEFAULT_EXPECTED_RUNS,
        )
        runs[run_spec.name] = {
            "expected": {
                "runs": DEFAULT_EXPECTED_RUNS,
                "agents": DEFAULT_EXPECTED_AGENTS,
                "ticks": DEFAULT_EXPECTED_TICKS,
                "ranks": run_spec.expected_ranks,
            },
            "summary_path": str(run_spec.summary_path),
            "agent_log_path": str(run_spec.agent_log_path),
            "behavior_log_path": str(run_spec.behavior_log_path),
            "validation": validation,
        }
        artifact_paths = [run_spec.summary_path, run_spec.agent_log_path, run_spec.behavior_log_path]
        if run_spec.validation_report_path is not None:
            runs[run_spec.name]["validation_report_path"] = str(run_spec.validation_report_path)
            artifact_paths.append(run_spec.validation_report_path)
        if run_spec.extra_artifact_paths:
            runs[run_spec.name]["extra_artifact_paths"] = [str(path) for path in run_spec.extra_artifact_paths]
            artifact_paths.extend(run_spec.extra_artifact_paths)

        for artifact_path in artifact_paths:
            artifacts[str(artifact_path)] = artifact_metadata(artifact_path)

    return {
        "version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "artifacts": artifacts,
    }


def write_mvp_manifest(
    output_path: Path = DEFAULT_MANIFEST_PATH,
    run_specs: tuple[MvpRunSpec, ...] = DEFAULT_RUN_SPECS,
) -> dict[str, Any]:
    """Validate MVP outputs and write their manifest to disk."""
    manifest = build_mvp_manifest(run_specs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path for the generated MVP artifact manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_mvp_manifest(args.output)
    print(f"MVP manifest written: {args.output} (runs={len(manifest['runs'])}, artifacts={len(manifest['artifacts'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
