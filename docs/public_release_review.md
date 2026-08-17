# Public Release Review Handoff

This page records the CASMSocial project team's response to the final package
review comments. It describes a review candidate, not an authorization to
publish the package.

## Review Target

- GitHub repository: `https://github.com/ccgsem/casmsocial`
- Branch: `public-release-candidate`
- Verified code and data baseline: `70a59a9`
- License: MIT
- Copyright: © 2026 The MITRE Corporation
- Public contact: Jon C. Cline `<jcline@mitre.org>`

Clone the review branch directly:

```bash
git clone --branch public-release-candidate --single-branch \
  https://github.com/ccgsem/casmsocial.git
cd casmsocial
git rev-parse HEAD
```

Review the latest branch tip. The baseline above identifies the last
code, data, and license change before this handoff document was added.

## Reviewer Comments and Disposition

| Review concern | Disposition | Evidence |
|---|---|---|
| The Wake County fixture caveat appeared to conflict with MIT redistribution rights. | Clarified. The project team recommends public release under MIT, while the final reviewer decision remains pending. The manifest now describes an internal release gate and expressly states that it does not add restrictions to MIT after an approved release. | `testdata/wake_county_heat_1000_households/manifest.yaml`, fixture `README.md`, and `docs/wake_county_heat_fixture.md` |
| Generated RPC files and their runtime raised a license-compatibility concern. | Resolved in the candidate. Generated stubs, the IDL source, build instructions, and the optional runtime dependency are absent. Live observation uses Apache Arrow Flight through the existing PyArrow dependency. The candidate began from a clean root commit, and every commit has been scanned for the removed generated-code and dependency identifiers. | `casmsocial/arrow_server.py`, `pyproject.toml`, clean candidate history |
| Package metadata used a personal contact address. | Resolved. Package and casmdb model metadata use the approved MITRE address. | `pyproject.toml`, `scripts/register_casmsocial.py`, `tests/test_register_casmsocial.py` |
| The registered model license was `Proprietary`. | Resolved. The registered model license is `MIT`, with regression coverage. | `scripts/register_casmsocial.py`, `tests/test_register_casmsocial.py` |
| Virtual-environment instructions omitted installation of project dependencies. | Resolved. The README now runs `python -m pip install -e .` after activation. | `README.md` |
| Package copyright needed correction. | Resolved. First-party notices use © 2026 The MITRE Corporation; third-party notices retain their original holders and years. | `LICENSE`, `mkdocs.yml`, source-license inventory, regression test |
| A DC test fixture was derived from data not suitable for redistribution. | Resolved in the candidate. That fixture, its generator, dedicated configuration, scenario, and tests are absent. The self-contained MVP now uses the neutral `casmsocial_mvp` schema, and the default review scenario uses Wake County. | `config/`, `scripts/create_mvp_ducklake.py`, `tests/test_public_release_sources.py` |

## Wake County Fixture Available for Review

The Git branch includes the complete review fixture:

```text
testdata/wake_county_heat_1000_households/
  README.md
  manifest.yaml
  tables/
    activities_1000_households.parquet
    hh_1000_households.parquet
    persons_1000_households.parquet
    places.parquet
```

The Parquet tables are available in a GitHub clone so the reviewer can inspect
and run the deployment workflow. They remain excluded from wheel and source
distribution artifacts while final approval is pending.

The manifest status is
`recommended_for_public_release_pending_final_review`. If the reviewer approves
the fixture, the project will record the decision, date, and stable evidence
URI and change the status before public package distribution.

## Verification Evidence

The candidate has been checked with:

```bash
python -m pytest -q
python -m ruff check .
mkdocs build --strict
```

The latest full verification completed with 381 tests passed and 3 tests
skipped: 373 non-socket tests plus 8 Arrow Flight socket tests. Wheel and source
distribution builds passed the release-artifact policy; neither distribution
contains the Wake fixture, the removed DC fixture, generated RPC material, or
the retired runtime dependency.

## Decisions Still Required

The candidate is not yet authorized for public release. The release-readiness
record remains open until these decisions are recorded:

1. Final reviewer decision on the Wake County fixture.
2. Contributor-authority approval.
3. Container-dependency license review.
4. Final distribution plan approval.

## Requested Reviewer Response

Please confirm one of the following for the Wake County fixture:

- **Approved:** The fixture may be included in the GitHub repository and
  distributed under the CASMSocial MIT license without additional
  redistribution restrictions.
- **Changes requested:** Identify the required changes or evidence.
- **Not approved:** The fixture must be removed before public release.

Please also report any remaining package-release findings so they can be
recorded in the release-readiness evidence.
