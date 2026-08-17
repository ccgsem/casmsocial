"""Evaluate machine controls and explicit approvals for a public dataset-builder release."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from casmsocial.datasets.colorado_front_range.profiles import (
    load_release_review_policy,
    load_source_licenses,
)


def _required_text(record: dict[str, Any], field: str, approval_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Approval {approval_id!r} requires non-empty {field!r}")
    return value.strip()


def _approved_at(record: dict[str, Any], approval_id: str) -> str:
    value = _required_text(record, "approved_at", approval_id)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Approval {approval_id!r} has an invalid approved_at timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Approval {approval_id!r} approved_at must include a UTC offset")
    return value


def _validate_attestations(
    attestations: dict[str, Any] | None,
    required_ids: set[str],
    release_version: str,
) -> dict[str, dict[str, str]]:
    if attestations is None:
        return {}
    if not isinstance(attestations, dict) or attestations.get("schema_version") != 1:
        raise ValueError("Release attestations must be a schema-version 1 mapping")
    attested_version = attestations.get("release_version")
    if attested_version != release_version:
        raise ValueError(f"Release attestations target version {attested_version!r}; expected {release_version!r}")
    records = attestations.get("approvals")
    if not isinstance(records, list):
        raise ValueError("Release attestations require an approvals list")

    approved: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each release approval must be a mapping")
        approval_id = _required_text(record, "id", "<unknown>")
        if approval_id not in required_ids:
            raise ValueError(f"Unknown release approval: {approval_id}")
        if approval_id in approved:
            raise ValueError(f"Duplicate release approval: {approval_id}")
        if record.get("decision") != "approved":
            raise ValueError(f"Approval {approval_id!r} must record decision: approved")
        approved[approval_id] = {
            "approved_by": _required_text(record, "approved_by", approval_id),
            "approved_at": _approved_at(record, approval_id),
            "evidence": _required_text(record, "evidence", approval_id),
        }
    return approved


def evaluate_release_readiness(
    attestations: dict[str, Any] | None = None,
    *,
    policy: dict[str, Any] | None = None,
    license_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return release readiness without inferring or fabricating human approvals."""
    policy = load_release_review_policy() if policy is None else policy
    license_manifest = load_source_licenses() if license_manifest is None else license_manifest
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported release-review policy schema")

    expectations = policy.get("machine_gate_expectations")
    requirements = policy.get("required_approvals")
    release_version = policy.get("release_version")
    if (
        not isinstance(expectations, dict)
        or not isinstance(requirements, list)
        or not isinstance(release_version, str)
        or not release_version
    ):
        raise ValueError("Release-review policy is missing gates or approvals")

    actual_gates = {
        gate["id"]: gate["status"]
        for gate in license_manifest.get("release_gates", [])
        if isinstance(gate, dict) and "id" in gate and "status" in gate
    }
    checks = [
        {
            "id": gate_id,
            "expected_status": expected,
            "actual_status": actual_gates.get(gate_id),
            "status": "passed" if actual_gates.get(gate_id) == expected else "failed",
        }
        for gate_id, expected in expectations.items()
    ]
    machine_status = "passed" if all(check["status"] == "passed" for check in checks) else "failed"

    requirement_by_id: dict[str, dict[str, Any]] = {}
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Each required approval must be a mapping")
        approval_id = requirement.get("id")
        resolved_gate = requirement.get("resolves_gate")
        if not isinstance(approval_id, str) or not approval_id or approval_id in requirement_by_id:
            raise ValueError("Required approval IDs must be non-empty and unique")
        if resolved_gate not in expectations:
            raise ValueError(f"Approval {approval_id!r} resolves an unknown gate")
        requirement_by_id[approval_id] = requirement

    approved = _validate_attestations(attestations, set(requirement_by_id), release_version)
    approval_results = []
    for approval_id, requirement in requirement_by_id.items():
        result = {
            "id": approval_id,
            "resolves_gate": requirement["resolves_gate"],
            "description": requirement.get("description", ""),
            "status": "approved" if approval_id in approved else "missing",
        }
        result.update(approved.get(approval_id, {}))
        approval_results.append(result)
    missing = [result["id"] for result in approval_results if result["status"] == "missing"]

    if machine_status != "passed":
        status = "machine_control_failed"
    elif missing:
        status = "review_required"
    else:
        status = "ready"
    return {
        "schema_version": 1,
        "scope": policy.get("scope"),
        "release_version": release_version,
        "status": status,
        "distribution_plan": policy.get("distribution_plan"),
        "machine_controls": {"status": machine_status, "checks": checks},
        "approvals": approval_results,
        "missing_approvals": missing,
    }
