# Colorado Front Range release review

The Colorado Front Range builder separates automated release controls from
organizational approval. A successful CI run proves that the recorded policies
and artifacts agree; it does not grant contributor authority or approve a
distribution.

The selected distribution plan publishes the CASMSocial code, wheel, source
distribution, documentation site, and production container. Generated
identifier-bearing population products remain in local user storage. OSM-derived
data also remains local under the project's `local_build_only` policy.

## Check the current state

Run the installed command:

```bash
casmsocial-data colorado release-readiness --format yaml
```

Before approvals are supplied, the expected result is `review_required`, with
machine controls `passed` and these three missing approvals:

1. `contributor_authority`: confirms authority to publish code migrated from the
   private repository.
2. `container_dependency_license_review`: confirms review of the generated
   Python and production-container inventories, the machine-readable container
   policy, retained notices, and native-package review queue for the intended
   distribution.
3. `final_distribution_plan`: approves public release of code and build
   artifacts under the local-only data policy.

CI enforces that state with:

```bash
python scripts/verify_release_readiness.py --expect-status review_required
```

## Record approvals

Approvals are release records, not repository defaults. Store the completed
attestation in the releasing organization's review system and pass an exported
YAML file to the command. Do not commit names or approvals merely to make the
gate pass.

```yaml
schema_version: 1
release_version: 2.6.1
approvals:
  - id: contributor_authority
    decision: approved
    approved_by: Responsible reviewer
    approved_at: '2026-08-03T12:00:00-06:00'
    evidence: release-review/contributor-authority-record
  - id: container_dependency_license_review
    decision: approved
    approved_by: Responsible reviewer
    approved_at: '2026-08-03T12:05:00-06:00'
    evidence: release-review/container-license-record
  - id: final_distribution_plan
    decision: approved
    approved_by: Release authority
    approved_at: '2026-08-03T12:10:00-06:00'
    evidence: release-review/final-distribution-record
```

Evaluate the record:

```bash
casmsocial-data colorado release-readiness \
  --attestations /path/to/release-attestations.yaml \
  --format json
```

The result is `ready` only when all machine expectations still match
`source_licenses.yaml` and all three approvals are present, explicit, unique,
timestamped with a UTC offset, and linked to evidence. Unknown, duplicate, or
malformed approvals are rejected. A machine-gate mismatch produces
`machine_control_failed` even if every approval is present. An attestation for a
different release version is rejected.

The authoritative contract is
`casmsocial/datasets/colorado_front_range/assets/release_review_policy.yaml`.
The container evidence record should reference the CI-produced SPDX artifact,
the passing `verify_container_sbom.py` summary, the reviewed native-package
decisions, and any notice or source-availability actions required for the
chosen container distribution. A policy result of `review_required` is
intentional and cannot substitute for that approval.
The broader source and dependency rationale is in the
[Colorado dataset licensing audit](colorado_front_range_data_licensing.md).
