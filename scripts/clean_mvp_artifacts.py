"""Remove generated artifacts from the local MVP scenario."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MVP_ARTIFACT_PATHS: tuple[Path, ...] = (
    Path("data/output/mvp_summary.md"),
    Path("data/output/mvp_agent_log.parquet"),
    Path("data/output/mvp_behavior_log.parquet"),
    Path("data/output/mvp_manifest.json"),
    Path("data/output/mvp_2rank_summary.md"),
    Path("data/output/mvp_2rank_agent_log.parquet"),
    Path("data/output/mvp_2rank_behavior_log.parquet"),
    Path("data/output/mvp_routed_summary.md"),
    Path("data/output/mvp_routed_agent_log.parquet"),
    Path("data/output/mvp_routed_behavior_log.parquet"),
    Path("data/output/mvp_routed_plan_validation.json"),
    Path("data/output/mvp_built_road_nodes.parquet"),
    Path("data/output/mvp_built_road_edges.parquet"),
    Path("data/output/mvp_built_place_road_snap.parquet"),
    Path("data/output/mvp_built_road_artifacts.json"),
    Path("data/output/mvp_built_roads_summary.md"),
    Path("data/output/mvp_built_roads_agent_log.parquet"),
    Path("data/output/mvp_built_roads_behavior_log.parquet"),
    Path("data/output/mvp_built_roads_plan_validation.json"),
    Path("data/output/mvp_delta_state_summary.md"),
    Path("data/output/mvp_delta_state_agent_log.parquet"),
    Path("data/output/mvp_delta_state_behavior_log.parquet"),
    Path("data/output/mvp_agent_state_delta.parquet"),
    Path("data/output/mvp_agent_state_delta_audit.parquet"),
    Path("data/output/mvp_agent_state_reconstructed.parquet"),
    Path("data/output/mvp_delta_state_validation.json"),
    Path("data/output/mvp_agent_state_delta_ducklake_report.md"),
    Path("examples/mvp/mvp.ducklake"),
)


def _remove_path(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def clean_mvp_artifacts(paths: Sequence[Path] = DEFAULT_MVP_ARTIFACT_PATHS) -> list[Path]:
    """Remove generated MVP artifact paths and return the paths that existed."""
    removed_paths = []
    for path in paths:
        if _remove_path(path):
            removed_paths.append(path)
    return removed_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_MVP_ARTIFACT_PATHS,
        help="Generated MVP artifact paths to remove.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    removed_paths = clean_mvp_artifacts(args.paths)
    if removed_paths:
        print("Removed generated MVP artifacts:")
        for path in removed_paths:
            print(f"- {path}")
    else:
        print("No generated MVP artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
