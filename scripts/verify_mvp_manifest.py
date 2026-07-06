"""Verify generated MVP artifacts against their manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.validate_mvp_output import validate_mvp_output
from scripts.write_mvp_manifest import DEFAULT_MANIFEST_PATH, artifact_metadata


class MvpManifestVerificationError(ValueError):
    """Raised when MVP artifacts do not match their manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MvpManifestVerificationError(message)


def _resolve_artifact_path(recorded_path: str, manifest_path: Path, artifact_root: Path | None) -> Path:
    """Resolve manifest paths for local output and flattened CI artifact downloads."""
    recorded = Path(recorded_path)
    roots = []
    if artifact_root is not None:
        roots.append(artifact_root)
    roots.append(manifest_path.parent)
    roots.append(Path.cwd())

    candidates = []
    for root in roots:
        candidates.append(root / recorded)
        if len(recorded.parts) > 1:
            candidates.append(root / Path(*recorded.parts[1:]))
        candidates.append(root / recorded.name)
    candidates.append(recorded)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized

    candidate_text = ", ".join(str(candidate) for candidate in candidates[:5])
    raise MvpManifestVerificationError(f"Manifest artifact is missing: {recorded_path}; checked {candidate_text}")


def _compare_metadata(recorded_path: str, expected: dict[str, Any], actual_path: Path) -> dict[str, Any]:
    actual = artifact_metadata(actual_path)
    for key in ("kind", "size_bytes", "sha256", "file_count", "parquet_file_count"):
        if key not in expected:
            continue
        _require(
            actual.get(key) == expected[key],
            f"Manifest artifact mismatch for {recorded_path}: {key} expected {expected[key]!r}, "
            f"found {actual.get(key)!r}",
        )
    return actual


def verify_mvp_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Verify manifest validation summaries and artifact checksums."""
    manifest_path = manifest_path.expanduser()
    _require(manifest_path.exists(), f"Manifest path does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("version") == 1, f"Unsupported MVP manifest version: {manifest.get('version')!r}")

    artifact_root = artifact_root.expanduser() if artifact_root is not None else None
    resolved_paths: dict[str, Path] = {}
    for recorded_path, expected_metadata in manifest.get("artifacts", {}).items():
        actual_path = _resolve_artifact_path(recorded_path, manifest_path, artifact_root)
        _compare_metadata(recorded_path, expected_metadata, actual_path)
        resolved_paths[recorded_path] = actual_path

    for run_name, run_data in manifest.get("runs", {}).items():
        expected = run_data.get("expected", {})
        agent_log_path = _resolve_artifact_path(str(run_data["agent_log_path"]), manifest_path, artifact_root)
        behavior_log_path = _resolve_artifact_path(str(run_data["behavior_log_path"]), manifest_path, artifact_root)
        validation = validate_mvp_output(
            agent_log_path,
            behavior_log_path,
            expected_agents=int(expected["agents"]),
            expected_ticks=int(expected["ticks"]),
            expected_ranks=int(expected["ranks"]),
            expected_runs=int(expected.get("runs", 1)),
        )
        _require(
            validation == run_data.get("validation"),
            f"Manifest validation mismatch for run {run_name}: expected {run_data.get('validation')!r}, "
            f"found {validation!r}",
        )

    return {
        "runs": len(manifest.get("runs", {})),
        "artifacts": len(manifest.get("artifacts", {})),
        "manifest_path": str(manifest_path),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "resolved_artifacts": {path: str(resolved_path) for path, resolved_path in resolved_paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the MVP artifact manifest.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Directory containing artifacts when verifying an extracted CI artifact.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print verification failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = verify_mvp_manifest(args.manifest, artifact_root=args.artifact_root)
    except (MvpManifestVerificationError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"MVP manifest verification failed: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("MVP manifest valid: " f"{args.manifest} " f"(runs={summary['runs']}, artifacts={summary['artifacts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
