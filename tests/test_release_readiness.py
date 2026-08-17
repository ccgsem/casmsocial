from copy import deepcopy

import pytest

from casmsocial.datasets.colorado_front_range import (
    evaluate_release_readiness,
    load_release_review_policy,
    load_source_licenses,
)


def _attestations(*approval_ids: str) -> dict:
    return {
        "schema_version": 1,
        "release_version": "2.5.8",
        "approvals": [
            {
                "id": approval_id,
                "decision": "approved",
                "approved_by": f"Reviewer for {approval_id}",
                "approved_at": "2026-08-03T12:00:00-06:00",
                "evidence": f"release-review/{approval_id}",
            }
            for approval_id in approval_ids
        ],
    }


def test_release_readiness_requires_all_explicit_approvals():
    result = evaluate_release_readiness()

    assert result["status"] == "review_required"
    assert result["machine_controls"]["status"] == "passed"
    assert result["missing_approvals"] == [
        "contributor_authority",
        "container_dependency_license_review",
        "final_distribution_plan",
    ]
    assert {approval["status"] for approval in result["approvals"]} == {"missing"}


def test_release_readiness_accepts_a_complete_attestation():
    result = evaluate_release_readiness(
        _attestations(
            "contributor_authority",
            "container_dependency_license_review",
            "final_distribution_plan",
        )
    )

    assert result["status"] == "ready"
    assert result["missing_approvals"] == []
    assert {approval["status"] for approval in result["approvals"]} == {"approved"}


def test_release_readiness_keeps_partial_attestations_gated():
    result = evaluate_release_readiness(_attestations("contributor_authority"))

    assert result["status"] == "review_required"
    assert result["missing_approvals"] == [
        "container_dependency_license_review",
        "final_distribution_plan",
    ]


@pytest.mark.parametrize(
    ("attestations", "message"),
    [
        (_attestations("unknown"), "Unknown release approval"),
        (
            {
                **_attestations("contributor_authority"),
                "approvals": _attestations("contributor_authority")["approvals"] * 2,
            },
            "Duplicate release approval",
        ),
        (
            {
                **_attestations("contributor_authority"),
                "approvals": [
                    {
                        **_attestations("contributor_authority")["approvals"][0],
                        "approved_at": "2026-08-03 12:00:00",
                    }
                ],
            },
            "UTC offset",
        ),
        (
            {**_attestations("contributor_authority"), "release_version": "2.5.4"},
            "expected '2.5.8'",
        ),
    ],
)
def test_release_readiness_rejects_invalid_attestations(attestations, message):
    with pytest.raises(ValueError, match=message):
        evaluate_release_readiness(attestations)


def test_release_readiness_reports_machine_gate_drift():
    manifest = deepcopy(load_source_licenses())
    manifest["release_gates"][0]["status"] = "unexpected"

    result = evaluate_release_readiness(policy=load_release_review_policy(), license_manifest=manifest)

    assert result["status"] == "machine_control_failed"
    assert result["machine_controls"]["status"] == "failed"
