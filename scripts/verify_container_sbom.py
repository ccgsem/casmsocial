"""Validate a production-container SPDX SBOM against the reviewed policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PURL_TYPE = re.compile(r"^pkg:([^/]+)")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _package_type(package: dict[str, Any]) -> str:
    for reference in package.get("externalRefs", []):
        if reference.get("referenceType") != "purl":
            continue
        if match := PURL_TYPE.match(reference.get("referenceLocator", "")):
            return match.group(1)
    return "untyped"


def _missing_license(package: dict[str, Any]) -> bool:
    declared = package.get("licenseDeclared") or "NOASSERTION"
    concluded = package.get("licenseConcluded") or "NOASSERTION"
    return declared == "NOASSERTION" and concluded == "NOASSERTION"


def _validate_override(name: str, override: dict[str, Any], reviewed_licenses: set[str]) -> list[str]:
    errors: list[str] = []
    for key in ("package_type", "version", "license", "source_url"):
        if not isinstance(override.get(key), str) or not override[key].strip():
            errors.append(f"override {name!r} is missing {key}")
    if override.get("license") not in reviewed_licenses:
        errors.append(f"override {name!r} has an unreviewed license {override.get('license')!r}")
    if not str(override.get("source_url", "")).startswith("https://"):
        errors.append(f"override {name!r} must use an HTTPS source_url")
    evidence = override.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"override {name!r} is missing evidence")
        return errors
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].startswith("/"):
            errors.append(f"override {name!r} has invalid evidence path")
        if not isinstance(item, dict) or not SHA256.fullmatch(str(item.get("sha256", ""))):
            errors.append(f"override {name!r} has invalid evidence SHA-256")
    return errors


def _verify_evidence(name: str, override: dict[str, Any], evidence_root: Path) -> list[str]:
    errors: list[str] = []
    for item in override["evidence"]:
        path = evidence_root / item["path"].lstrip("/")
        if not path.is_file():
            errors.append(f"override {name!r} evidence file is missing: {item['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            errors.append(f"override {name!r} evidence SHA-256 changed: {item['path']}")
    return errors


def verify_container_sbom(
    sbom_path: Path,
    policy_path: Path,
    *,
    evidence_root: Path | None = None,
    expect_status: str = "review_required",
) -> dict[str, int | str]:
    """Validate SPDX coverage without representing organizational approval."""
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if sbom.get("spdxVersion") != "SPDX-2.3":
        raise ValueError("Container SBOM must be an SPDX 2.3 document")
    if policy.get("schema_version") != 1:
        raise ValueError("Container license policy must use schema_version 1")

    reviewed_licenses = set(policy.get("reviewed_override_licenses", []))
    overrides = policy.get("metadata_overrides", {})
    forbidden_types = set(policy.get("forbidden_package_types", []))
    scanner_subject = policy.get("scanner_subject", {})
    native_review = policy.get("native_review", {})
    status = native_review.get("status")
    violations: list[str] = []

    if status != expect_status:
        violations.append(f"expected native review status {expect_status!r}, found {status!r}")
    if not isinstance(overrides, dict) or not overrides:
        violations.append("container policy must define metadata_overrides")
        overrides = {}
    for name, override in overrides.items():
        if not isinstance(override, dict):
            violations.append(f"override {name!r} must be a mapping")
            continue
        override_errors = _validate_override(name, override, reviewed_licenses)
        violations.extend(override_errors)
        if evidence_root is not None and not override_errors:
            violations.extend(_verify_evidence(name, override, evidence_root))

    override_index = {
        (override.get("package_type"), name.casefold()): override
        for name, override in overrides.items()
        if isinstance(override, dict)
    }
    used_overrides: set[tuple[str, str]] = set()
    subject_count = 0
    native_packages = 0
    native_review_packages: set[str] = set()
    trigger_patterns = [re.compile(pattern) for pattern in native_review.get("review_trigger_patterns", [])]

    packages = sbom.get("packages", [])
    for package in packages:
        name = str(package.get("name", "<unnamed>"))
        version = package.get("versionInfo")
        package_type = _package_type(package)
        if package_type in forbidden_types:
            violations.append(f"{name}=={version}: forbidden package type {package_type}")

        if package_type == native_review.get("package_type"):
            native_packages += 1
            declared = str(package.get("licenseDeclared") or "NOASSERTION")
            concluded = str(package.get("licenseConcluded") or "NOASSERTION")
            license_metadata = " AND ".join(value for value in (declared, concluded) if value != "NOASSERTION")
            if native_review.get("require_license_metadata") and not license_metadata:
                violations.append(f"{name}=={version}: native package has no declared or concluded license")
            if any(pattern.search(license_metadata) for pattern in trigger_patterns):
                native_review_packages.add(name)

        if not _missing_license(package):
            continue
        key = (package_type, name.casefold())
        override = override_index.get(key)
        if override is not None:
            if override.get("version") != version:
                violations.append(f"{name}=={version}: metadata override is pinned to {override.get('version')}")
            else:
                used_overrides.add(key)
            continue
        if package_type == scanner_subject.get("package_type") and str(package.get("SPDXID", "")).startswith(
            str(scanner_subject.get("spdx_id_prefix", "!"))
        ):
            subject_count += 1
            continue
        violations.append(f"{name}=={version}: missing license declaration and reviewed classification")

    unused_overrides = set(override_index) - used_overrides
    if unused_overrides:
        violations.append(f"unused metadata overrides: {sorted(name for _, name in unused_overrides)}")
    required_subjects = scanner_subject.get("required_count")
    if subject_count != required_subjects:
        violations.append(f"expected {required_subjects} scanner subject record(s), found {subject_count}")
    if native_packages and not trigger_patterns:
        violations.append("native review must define review_trigger_patterns")

    if violations:
        raise ValueError("Container SBOM license policy failed:\n- " + "\n- ".join(violations))
    return {
        "packages": len(packages),
        "metadata_overrides": len(used_overrides),
        "scanner_subjects": subject_count,
        "native_packages": native_packages,
        "native_review_packages": len(native_review_packages),
        "violations": 0,
        "status": str(status),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sbom", type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--expect-status", choices=("review_required",), default="review_required")
    args = parser.parse_args()
    try:
        summary = verify_container_sbom(
            args.sbom,
            args.policy,
            evidence_root=args.evidence_root,
            expect_status=args.expect_status,
        )
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(error)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
