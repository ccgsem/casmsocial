"""Reject local or identifier-bearing datasets from public release artifacts."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".dat",
    ".duckdb",
    ".feather",
    ".parquet",
    ".pbf",
    ".sqlite",
}
FORBIDDEN_TABLE_FILES = {
    "activities.csv",
    "hh.csv",
    "households.csv",
    "persons.csv",
    "places.csv",
    "social_networks.csv",
}
FORBIDDEN_PATH_PAIRS = {
    ("data", "local"),
    ("data", "output"),
    ("data", "raw"),
}
FORBIDDEN_PATH_PARTS = {"testdata"}
IGNORED_DIRECTORY_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


def _violation_reason(member: str) -> str | None:
    path = PurePosixPath(member.replace("\\", "/"))
    parts = tuple(part.lower() for part in path.parts if part not in {"", "."})
    if any(part in FORBIDDEN_PATH_PARTS for part in parts):
        return "forbidden dataset directory"
    if any(pair in zip(parts, parts[1:], strict=False) for pair in FORBIDDEN_PATH_PAIRS):
        return "local, raw, or generated data directory"
    if any(part.endswith(".ducklake") for part in parts):
        return "DuckLake database directory"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"forbidden dataset extension {path.suffix.lower()}"
    if path.name.lower() in FORBIDDEN_TABLE_FILES:
        return "identifier-bearing table filename"
    return None


def _archive_members(path: Path) -> Iterable[str] | None:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return tuple(archive.namelist())
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")) or tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            return tuple(member.name for member in archive.getmembers())
    return None


def scan_path(path: Path) -> list[str]:
    """Return release-boundary violations found below or within *path*."""
    if not path.exists():
        return [f"{path}: artifact does not exist"]
    if path.is_dir():
        violations: list[str] = []
        for candidate in sorted(path.rglob("*")):
            relative = candidate.relative_to(path)
            if any(part in IGNORED_DIRECTORY_PARTS for part in relative.parts):
                continue
            if candidate.is_file():
                violations.extend(scan_path(candidate))
        return violations

    members = _archive_members(path)
    if members is not None:
        return [f"{path}!{member}: {reason}" for member in members if (reason := _violation_reason(member)) is not None]

    reason = _violation_reason(path.as_posix())
    return [f"{path}: {reason}"] if reason is not None else []


def verify_release_artifacts(paths: Iterable[Path]) -> list[str]:
    """Return every release-boundary violation across *paths*."""
    return [violation for path in paths for violation in scan_path(path)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path, help="Archive or directory to scan")
    args = parser.parse_args()
    violations = verify_release_artifacts(args.artifacts)
    if violations:
        print("Release artifact scan failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print(f"Release artifact scan passed for {len(args.artifacts)} artifact path(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
