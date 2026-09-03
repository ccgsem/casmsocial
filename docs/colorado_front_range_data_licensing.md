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
| CASMSocial code | MIT, © 2026 The MITRE Corporation | Retain the repository `LICENSE`. |
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

The 2026-08-09 Step 4 audit used CycloneDX Python 7.3.0 for the Python
environment and Syft 1.44.0's SPDX 2.3 exporter for the production image. The
reproducible Python inventory contained 106 components and passed the reviewed
license policy with 12 version-pinned metadata overrides. `cligj` and
`repast4py` omit usable license identifiers; ten other packages publish the
ambiguous `License :: OSI Approved :: BSD License` classifier. The gate no
longer assumes that classifier means BSD-3-Clause: each affected version points
to its installed license evidence. This matters because Numba 0.66.0 is
BSD-2-Clause while the other reviewed ambiguous-classifier packages are
BSD-3-Clause.

The optional Shapely 2.1.2 wheel is BSD-3-Clause and bundles GEOS under
LGPL-2.1, as recorded by its installed `LICENSE_GEOS`. Shapely is used only by
the `data-builder` extra and is not present in the CASMSocial wheel or current
production image. Any future artifact that redistributes that wheel must retain
the bundled notice and include the LGPL obligations in its distribution review.

Step 5 separates dependency compilation from the production runtime. The
builder uses a digest-pinned `uv` image and retains the compilers, headers,
native `-dev` packages, package-manager cache, and Rust executable metadata.
The final image starts from a clean Python runtime and receives only the
non-editable virtual environment, application payload, CA certificates,
`libexpat1`, and the MPICH runtime. CI rejects `uv`, C/C++ compilers, the uv
cache, and the named build-only Debian packages in the production target. It
also imports the compiled Rasterio, Torch, mpi4py, and repast4py dependencies
under a one-rank MPI launch.

Step 6 removes system `pip` and 14 Windows-only launcher binaries vendored by
pip and setuptools; none is needed by the Linux runtime. The resulting arm64
audit image is 623 MB, down from 1.02 GB before the builder/runtime split. Its
regenerated inventory contains 187 SPDX packages: 121 Debian, 64 Python, one
OCI scan-subject record, and one generic Python runtime record. There are no
Cargo or untyped-package records.

Syft leaves seven records without either a machine-readable concluded or
declared license. `container_dependency_license_policy.yaml` classifies six by
exact package type, name, and version: annotated-types, DuckDB, Loguru, and
markdown-it-py are MIT; Jinja2 is BSD-3-Clause; and the Python 3.12.14 runtime
uses PSF-2.0 while retaining its complete incorporated-software license file.
Every override points to an authoritative upstream URL and one or more hashed
files inside the image. Because Loguru's wheel declares only an MIT classifier,
CASMSocial also retains the
[tagged Loguru 0.7.3 license](https://github.com/Delgan/loguru/blob/0.7.3/LICENSE)
as a packaged third-party asset. The seventh record is the SPDX document root
for the OCI archive being scanned, not an additional dependency.
[Python's official documentation](https://docs.python.org/3.12/license.html)
confirms the primary PSF License Version 2 and warns that incorporated
components have separate terms, which is why the complete installed
`LICENSE.txt` remains controlling evidence.

The policy verifier rejects Cargo, an unclassified missing license, version
drift, changed evidence hashes, non-root OCI exceptions, or a native package
without declared or concluded metadata. The audited image passes with six
overrides, one scanner subject, and zero violations. Its conservative native
review patterns flag 117 of 121 Debian records because their aggregate
copyright expressions mention GPL/LGPL terms or custom `LicenseRef` entries.
That count is a review queue, not a finding that 117 runtime binaries impose a
particular obligation on CASMSocial; Debian copyright expressions can cover
source files that are not present in a given binary package.

The metadata-classification and toolchain-removal blockers are resolved, but
the production-container license review remains an explicit organizational
gate. An accountable reviewer must assess the native binary composition,
retained notices, and any source-availability obligations for the intended
distribution. Automated inventory generation and a passing policy do not
approve public container distribution.

CI regenerates both inventories on every change. It uploads the CycloneDX and
SPDX documents as workflow artifacts for 14 days; generated SBOM files are not
committed to the repository or included in release packages.
