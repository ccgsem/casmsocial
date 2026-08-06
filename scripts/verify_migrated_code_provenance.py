"""Verify public migration records and optionally re-audit the private source repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml  # type: ignore[import-untyped]

HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")


def _license_manifest_pairs(manifest: dict) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    modules = manifest["software"]["migrated_mydatalakehouse_code"]["migrated_modules"]
    for module in modules:
        sources = module.get("sources", [module.get("source")])
        commits = module.get("source_commits")
        for source in sources:
            if commits:
                commit = commits[Path(source).name]
            else:
                commit = module["source_commit"]
            pairs.add((source, commit, module["destination"]))
    return pairs


def _git_bytes(repository: Path, revision: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), "show", revision],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def verify_migrated_code_provenance(
    provenance_path: Path,
    license_manifest_path: Path,
    repository_root: Path,
    source_repository: Path | None = None,
) -> dict[str, object]:
    """Verify provenance coverage and, when supplied, immutable private source evidence."""
    provenance = yaml.safe_load(provenance_path.read_text())
    licenses = yaml.safe_load(license_manifest_path.read_text())
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported migrated-code provenance schema")
    if provenance["source_repository"].get("license") != "MIT":
        raise ValueError("Migrated source repository must record its MIT license")
    if provenance.get("notice") != "THIRD_PARTY_NOTICES.md":
        raise ValueError("Migrated code must retain THIRD_PARTY_NOTICES.md")
    if provenance.get("authority", {}).get("status") != "organization_review_required":
        raise ValueError("Contributor authority must remain an explicit organizational review")
    license_evidence = provenance["source_repository"]["license_evidence"]
    if (
        not HEX_40.fullmatch(license_evidence.get("commit", ""))
        or not HEX_64.fullmatch(license_evidence.get("sha256", ""))
        or license_evidence.get("path") != "LICENSE"
    ):
        raise ValueError("Private repository license evidence is incomplete")

    migrations = provenance.get("migrations", [])
    recorded_pairs: set[tuple[str, str, str]] = set()
    destinations: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    for migration in migrations:
        source_path = migration["source_path"]
        commit = migration["source_commit"]
        digest = migration["source_sha256"]
        key = (source_path, commit)
        if key in source_keys:
            raise ValueError(f"Duplicate source provenance entry: {source_path}@{commit}")
        source_keys.add(key)
        if not source_path.startswith("mydatalakehouse/") or not HEX_40.fullmatch(commit):
            raise ValueError(f"Unsafe or invalid source revision: {source_path}@{commit}")
        if (
            not HEX_64.fullmatch(digest)
            or not migration.get("source_author")
            or not migration.get("source_authored_at")
        ):
            raise ValueError(f"Incomplete source evidence: {source_path}@{commit}")
        for destination in migration["destinations"]:
            destination_path = (repository_root / destination).resolve()
            if not destination.startswith("casmsocial/") or not destination_path.is_relative_to(
                repository_root.resolve()
            ):
                raise ValueError(f"Unsafe migrated destination: {destination}")
            if not destination_path.is_file():
                raise ValueError(f"Migrated destination does not exist: {destination}")
            destinations.add(destination)
            recorded_pairs.add((source_path, commit, destination))

        if source_repository is not None:
            content = _git_bytes(source_repository, f"{commit}:{source_path}")
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {source_path}@{commit}")
            author = _git_text(source_repository, "show", "-s", "--format=%an", commit)
            authored_at = _git_text(source_repository, "show", "-s", "--format=%aI", commit)
            if author != migration["source_author"] or authored_at != migration["source_authored_at"]:
                raise ValueError(f"Source authorship mismatch: {source_path}@{commit}")

    expected_pairs = _license_manifest_pairs(licenses)
    if recorded_pairs != expected_pairs:
        missing = sorted(expected_pairs - recorded_pairs)
        extra = sorted(recorded_pairs - expected_pairs)
        raise ValueError(f"Provenance and license manifests disagree; missing={missing}, extra={extra}")

    if source_repository is not None:
        license_bytes = _git_bytes(
            source_repository,
            f"{license_evidence['commit']}:{license_evidence['path']}",
        )
        if hashlib.sha256(license_bytes).hexdigest() != license_evidence["sha256"]:
            raise ValueError("Private repository license evidence hash mismatch")

    notice = repository_root / provenance["notice"]
    if not notice.is_file() or "Copyright (c) 2024, Jon Cline" not in notice.read_text():
        raise ValueError("Retained private-source MIT notice is missing")
    return {
        "sources": len(migrations),
        "destinations": len(destinations),
        "source_repository_verified": source_repository is not None,
        "authority_status": provenance["authority"]["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--license-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    try:
        result = verify_migrated_code_provenance(
            args.provenance,
            args.license_manifest,
            args.repository_root,
            args.source_repository,
        )
    except (KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Migrated-code provenance verification failed: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
