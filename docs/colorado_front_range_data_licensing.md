# Colorado Front Range code and data licensing audit

This is an engineering release audit, not legal advice. It records the source
terms verified on 2026-08-01 and the controls required before CASMSocial ships a
public Colorado dataset builder. The machine-readable companion is
`casmsocial/datasets/colorado_front_range/assets/source_licenses.yaml`.

Code migrated into the public builder is tracked by source path and commit in
that manifest. The retained MIT copyright and permission text is distributed
in `THIRD_PARTY_NOTICES.md`.

## Outcome

CASMSocial may ship code that downloads and locally transforms the identified
public sources, subject to the release gates below. CASMSocial should not bundle
the generated person-, household-, place-, schedule-, social-network-, or
OSM-derived tables in its wheel, source distribution, container image, or Git
repository.

| Material | Verified status | Public-release treatment |
|---|---|---|
| CASMSocial code | MIT, Copyright 2025 The MITRE Corporation | Retain the repository `LICENSE`. |
| Code migrated from `mydatalakehouse` | MIT, Copyright 2024 Jon Cline | Compatible with MIT distribution; preserve its copyright and permission notice in copied or substantially derived code and record file provenance. |
| OSF synthetic population (`fpnc2`) | Public OSF project; official API reports CC0 1.0 | Downloader and transformations may be public. Download at build time; do not bundle generated identifier-bearing tables. |
| OSF education sites (`ts9mg`) | Public OSF project; official API reports CC0 1.0 | Same treatment as the population archive. |
| BLS ATUS public-use files | BLS states its published material is public domain, aside from separately copyrighted images | Download official files and cite BLS. Do not bundle archives merely for convenience. |
| Census TIGER/Line | U.S. government material; Census says it may be reproduced and requests citation | Cite Census, retain its legal disclaimer when repackaging, and respect TIGER/Line trademark limits. |
| OpenStreetMap / Geofabrik extract | OpenStreetMap data under ODbL 1.0 | Always attribute OpenStreetMap. Keep derived databases local unless the project has selected and implemented an ODbL-compliant distribution path. |

## Source evidence

The official OSF API records for
[the synthetic-population project](https://api.osf.io/v2/nodes/fpnc2/) and
[education-sites project](https://api.osf.io/v2/nodes/ts9mg/) both reference
OSF's `CC0 1.0 Universal` license record. CC0 permits copying, modification,
and distribution, but it does not clear rights the affirmer did not own or
remove privacy, publicity, patent, or trademark concerns. CASMSocial therefore
continues to treat generated identifier-bearing tables as local inputs even
though the source copyright terms are permissive. See the
[OSF licensing guide](https://help.osf.io/article/148-licensing) and
[CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode).

The [BLS copyright statement](https://www.bls.gov/opub/copyright-information.htm)
says BLS-published material is public domain except for separately copyrighted
photographs and illustrations, and requests citation. The builder uses official
[ATUS public-use files](https://www.bls.gov/tus/data.htm), not page artwork or
BLS marks.

The Census Bureau's TIGER/Line technical documentation says U.S. government
works are not eligible for copyright protection and asks users to cite Census.
It also supplies accuracy, legal-boundary, repackaging, and trademark notices;
those notices must accompany any redistributed boundary package. See the
[TIGER/Line documentation](https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/tiger-geo-line.html).

[OpenStreetMap's copyright page](https://www.openstreetmap.org/copyright)
identifies the database license as ODbL 1.0 and requires attribution and a clear
license reference. Adapted or derived databases can trigger share-alike and
source-access obligations. The safe default is to distribute the extraction
code, mappings, source URL, checksum, and attribution text while requiring each
user to build the POI database locally.

## Code-migration controls

The private `mydatalakehouse` repository and CASMSocial both use MIT licenses,
but their copyright notices name different holders. Before moving a module:

1. Record its private-repository path, commit, author, and destination path.
2. Confirm the contributor has authority to release the code and that no
   employer, contract, or third-party restriction supersedes the repository
   license.
3. Preserve the applicable 2024 copyright and MIT permission notice in a
   third-party notice or source header.
4. Remove private paths, credentials, internal URLs, generated data, and copied
   notebook output.
5. Review embedded mappings or fixtures separately; an MIT code license does
   not automatically license their underlying data.

Do not merge the private repository's Git history into the public repository.
Move audited files in clean, reviewable commits with a provenance manifest.

## OSM distribution decision

Before a release contains any generated destination data, select exactly one
policy:

- **Local build only (recommended):** release code and metadata; generated OSM
  tables remain in ignored user storage.
- **Produced Work:** release only a rendered or aggregate output after confirming
  it qualifies and applying the required attribution.
- **Derived database:** distribute under the applicable ODbL terms, attribution,
  share-alike conditions, and source-access offer.

Until that decision is reviewed, the dataset builder must refuse any `publish`
or packaging operation that includes `osm_pois`, destination supply, or a
DuckLake populated from them.

## Remaining release gates

1. **Documented; authority review required:**
   `migrated_code_provenance.yaml` records every private source path, immutable
   commit, source SHA-256, author, timestamp, public destination, MIT license
   evidence, and retained notice. Public CI verifies complete agreement with
   the license manifest. Maintainers with the private repository can re-audit
   the Git objects with `scripts/verify_migrated_code_provenance.py`. Commit
   authorship and a repository license do not replace the releasing
   organization's contributor-authority review.

   Inspect the shipped record with:

   ```bash
   casmsocial-data colorado provenance --format yaml
   ```

   Maintainers who have both repositories can re-check every private Git blob,
   author, timestamp, and license hash with:

   ```bash
   python scripts/verify_migrated_code_provenance.py \
     --provenance casmsocial/datasets/colorado_front_range/assets/migrated_code_provenance.yaml \
     --license-manifest casmsocial/datasets/colorado_front_range/assets/source_licenses.yaml \
     --repository-root . \
     --source-repository /path/to/mydatalakehouse
   ```
2. **Automated inventory; final review required:** CI generates a reproducible
   CycloneDX inventory for Python dependencies, validates it against
   `dependency_license_policy.yaml`, and publishes an SPDX inventory for the
   production image. The image inventory includes the native geospatial, MPI,
   and base-system packages. The releasing organization must review that full
   inventory for its intended binary and container distribution.
3. **Implemented:** OSM-derived products use the selected `local_build_only`
   policy. Every destination build writes and hashes
   `OPENSTREETMAP_ATTRIBUTION.md`; its manifest explicitly records that
   redistribution is not authorized. Release artifact scans reject the
   generated database and table formats.
4. **Automated:** `scripts/verify_release_artifacts.py` scans the wheel, source
   distribution, documentation site, and production-image project payload for
   generated dataset files in CI. The source distribution and Docker context
   explicitly exclude local data, test fixtures, and generated DuckLake files.
5. **Machine-readable; organization review required:**
   `release_review_policy.yaml` fixes the intended public artifacts and the
   local-only treatment of generated data. The
   [release-review runbook](colorado_front_range_release_review.md) defines the
   three explicit approvals and attestation format. Until those records are
   supplied, `casmsocial-data colorado release-readiness` correctly reports
   `review_required`; CI must not fabricate or imply their completion.

## Dependency-inventory evidence

The 2026-08-03 Step 7 audit used CycloneDX Python 7.3.0 for the Python
environment and Docker Scout's SPDX 2.3 exporter for the production image. The
reproducible Python inventory contained 106 components and passed the reviewed
license policy with two version-pinned metadata overrides (`cligj` and
`repast4py`). Both overrides point to BSD license files installed with their
packages.

Removing development groups from the production image reduced its Python
installation from 103 dependencies to 50 before installing CASMSocial. The
resulting image inventory contains 1,145 SPDX packages: 562 Debian, 506 Cargo,
73 Python, two RPM, one OCI, and one generic component. Docker Scout reports
564 packages without a machine-readable concluded or declared license, so the
native/container license review remains an explicit organizational release
gate rather than being represented as complete.

CI regenerates both inventories on every change. It uploads the CycloneDX and
SPDX documents as workflow artifacts for 14 days; generated SBOM files are not
committed to the repository or included in release packages.
