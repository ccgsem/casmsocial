# Colorado Front Range data acquisition

CASMSocial ships the builder and source metadata, not the generated population
or identifier-bearing inputs. All commands below write beneath a local,
Git-ignored data directory.

## Inspect the source contract

```bash
casmsocial-data colorado sources
```

The inventory distinguishes two verification policies:

- `pinned_sha256`: the OSF population and education-site archives must match
  the checked-in byte count and SHA-256 digest.
- `record_sha256`: mutable or manually acquired inputs receive a local
  `*.provenance.json` sidecar recording their observed byte count and digest.

## Fetch automatic sources

Fetch one artifact at a time so a command cannot unexpectedly download every
large input:

```bash
casmsocial-data colorado fetch osf-colorado-population --data-dir data
casmsocial-data colorado fetch osf-colorado-education-sites --data-dir data
casmsocial-data colorado fetch census-2023-counties --data-dir data
casmsocial-data colorado fetch osm-geofabrik-colorado --data-dir data
```

Downloads use a temporary `.part` file. Checksum-pinned downloads use an HTTP
range request to resume when the server supports it; unpinned mutable downloads
restart so two source versions can never be spliced together. The final path is
replaced only after required size and checksum validation succeeds. Use
`--overwrite` to discard an existing partial download or replace an unverified
destination.

The Geofabrik URL is intentionally mutable. Its sidecar pins the resolved URL,
download time, size, and SHA-256 for the local run. Keep OpenStreetMap
attribution with derived work and review the ODbL distribution policy before
publishing any derived database.

## Register manual ATUS files

Download and extract the 2024 Respondent, Activity, and Roster files from the
[BLS ATUS page](https://www.bls.gov/tus/data/datafiles-2024.htm), placing them at:

```text
data/raw/atus/2024/atusresp_2024.dat
data/raw/atus/2024/atusact_2024.dat
data/raw/atus/2024/atusrost_2024.dat
```

Then record their local provenance:

```bash
casmsocial-data colorado record bls-atus-2024-respondents --data-dir data
casmsocial-data colorado record bls-atus-2024-activities --data-dir data
casmsocial-data colorado record bls-atus-2024-roster --data-dir data
```

## Verify staged inputs

```bash
casmsocial-data colorado verify osf-colorado-population --data-dir data
casmsocial-data colorado verify bls-atus-2024-roster --data-dir data
```

The command exits nonzero for a missing, unrecorded, or mismatched artifact.
Neither raw files nor provenance sidecars are included in release artifacts.

## Normalize the OSF archives

Install the optional geospatial builder dependencies and generate the four
canonical Colorado tables:

```bash
pip install 'casmsocial[data-builder]'
casmsocial-data colorado build-osf --data-dir data
```

The command requires both OSF archives to verify first, then writes
`places.parquet`, `hh.parquet`, `persons.parquet`, `social_networks.parquet`,
and `manifest.json` beneath
`data/local/osf-synthetic-population/source_state=CO/`. Social ties are
timeless potential household, school, and daycare relationships. Work-network
memberships are excluded because their source endpoint is not a person, and
ties with unresolved person endpoints are rejected.

## Materialize the OSF DuckLake

After normalization, build the managed local catalog:

```bash
casmsocial-data colorado build-ducklake \
  --input-dir data/local/osf-synthetic-population \
  --catalog data/local/osf-synthetic-ducklake/metadata.ducklake \
  --data-path data/local/osf-synthetic-ducklake/files
```

Every `source_state=*` manifest and table hash is verified before loading. The
four schemas must agree across states. Acceptance requires complete household
and home-place references plus canonical, unique, endpoint-complete social
ties. Unresolved activity anchors are retained as a diagnostic because the OSF
source has known unresolved assignments.

The catalog is published only after acceptance passes. Repeating the command
with unchanged source manifests returns the accepted build as resumed. Changed
inputs or outputs require the explicit `--overwrite` option.

## Build a profile population

Select a supported geographic and population profile from the statewide lake:

```bash
casmsocial-data colorado build-population example-1k --data-dir data
casmsocial-data colorado build-population example-10k --data-dir data
casmsocial-data colorado build-population north-corridor-full --data-dir data
```

The builder spatially assigns statewide home coordinates to the bundled 2023
county/CBSA boundary. Sample profiles reserve their configured minimum in each
CBSA, then fill remaining slots deterministically across CBSA, age-group, and
activity-assignment strata. Full profiles retain every resident whose home is
inside the selected CBSAs.

Each output contains `places`, `hh`, `persons`, and endpoint-complete
`social_networks` Parquet tables plus a manifest with input hashes, sampling
policy, CBSA coverage, and integrity checks. Known unresolved OSF activity
anchors remain diagnostic; schedule generation must fall them back to home.
Sample profiles also preserve one deterministic, household-preferred source
tie by swapping its missing endpoint for a non-seed resident in the same CBSA.
This keeps the exact population and CBSA counts while preventing an empty
induced network in runtime smoke examples.

The six-metro profile remains planned. It is rejected unless the operator uses
`--allow-planned`, and that override does not clear its routing and destination
coverage release blockers.

## Build weekday and weekend schedules

After recording and verifying all three official ATUS extracts, build the
profile's pre-routing schedule:

```bash
casmsocial-data colorado build-schedules example-1k --data-dir data
casmsocial-data colorado build-schedules example-10k --data-dir data
casmsocial-data colorado build-schedules north-corridor-full --data-dir data
```

The command stages the 2024 ATUS public-use extracts, normalizes each donor to
a contiguous 04:00-to-04:00 diary, assigns weighted weekday and weekend donors
to adults, and creates rule-based weekday school/daycare schedules for children.
Every person receives both day types and exactly one distinct weekday home.
ATUS travel intervals touching the 04:00 day boundary fall back to home because
the public diary does not identify a stationary endpoint outside the modeled
day; internal travel intervals remain explicit routing placeholders.

The accepted manifest hashes the profile population and ATUS inputs and records
donor fallback levels, coverage, continuity, interval, and place-reference
checks. Work, school, or daycare anchors missing from the profile's `places`
table fall back to home. Travel intervals remain explicit placeholders and
discretionary adult activities remain at home until the destination and routing
stage. These schedules are representative plans, not observed longitudinal
trajectories.

## Build and assign destination supply

Create the profile-bounded OSM supply after its population and schedules pass:

```bash
casmsocial-data colorado build-destinations example-1k --data-dir data
casmsocial-data colorado build-destinations example-10k --data-dir data
casmsocial-data colorado build-destinations north-corridor-full --data-dir data
```

The builder reads the verified Colorado Geofabrik extract directly, classifies
supported point and polygon features, and uses the verified 2023 Census county
polygons to assign each POI to an administrative CBSA. It rejects profiles that
do not meet the configured minimum supply for every discretionary activity kind
in every selected CBSA. The default breadth gate is 20 places per kind and CBSA;
the chosen value and observed coverage are recorded in the destination manifest.

Home and resolved work, school, and daycare anchors are combined with the OSM
supply. Discretionary destinations are selected independently for every event,
first from the person's home grid cell and then from the home CBSA. This
event-level contract deliberately permits one person to visit multiple places
for the same activity kind; person-level destination anchors are not created.
Travel placeholders are removed and stationary timing remains contiguous so
Step 10 can insert computed travel legs.

OSM capacities are explicit scenario controls multiplied according to the
selected profile. They are not observed building capacities or occupancy.
Outputs remain local identifier-bearing OSM-derived simulation inputs and must
not be distributed by this workflow. Every successful destination build writes
`OPENSTREETMAP_ATTRIBUTION.md`, and the manifest hashes that notice alongside
the tables. Print the same bundled notice at any time with:

```bash
casmsocial-data colorado osm-attribution
```

A separately reviewed and explicitly implemented ODbL-compliant distribution
path is required before redistributing these derived tables or databases.
The planned six-metro profile additionally requires `--allow-planned` and
explicit positive `--capacity-multiplier` and
`--full-population-capacity-multiplier` values; these exploratory overrides do
not clear its release blockers.

## Route and build the CASMSocial runtime

Complete the local product after Step 9 passes:

```bash
casmsocial-data colorado build-runtime example-1k --data-dir data
casmsocial-data colorado build-runtime example-10k --data-dir data
casmsocial-data colorado build-runtime north-corridor-full --data-dir data
```

Routing uses the profile's straight-line speed and duration controls. A trip
that cannot fit before the next event retains the last reachable location and
records the original purpose instead of fabricating an unrealistically short
trip. Full builds use manifest-checked person partitions and resume completed
partitions from the `.building` directory after interruption.

The command rejects continuity, interval, place-type, or peak-capacity results
outside the profile limits. It then exports event-place-aware CASMSocial tables,
materializes them in a relocatable local DuckLake, and writes `casmsocial.yaml`.
Peak capacity is evaluated independently for weekday and weekend because these
are alternative daily plans, not simultaneous populations.
Run it from the runtime directory with:

```bash
CASMSOCIAL_DUCKLAKE_PATH=$PWD/ducklake \
mpirun -n 1 /path/to/casmsocial/.venv/bin/python -m casmsocial casmsocial.yaml
```

`activities.sp_act_id` remains authoritative. Person destination columns are
only deterministic compatibility fallbacks and do not collapse repeated
activity purposes to one venue.

## Verify the runtime

Run the profile's required MPI smoke and equivalence gates:

```bash
casmsocial-data colorado verify-runtime example-1k --data-dir data
casmsocial-data colorado verify-runtime example-10k --data-dir data
```

The verifier always disables person-level, behavior, and state-delta logs. It
retains only aggregate occupancy and interaction datasets, checks the expected
25 ticks for a 24-hour example, and requires active people, shared occupancy,
in-person interactions, and remote messages. Profiles requiring two-rank
verification compare aggregate interaction totals and peak occupancy exactly
against the single-rank run. Successful results are resumable by runtime
manifest hash.

## Build one profile end to end

After staging and verifying the source files, inspect the complete build plan
without writing any generated data:

```bash
casmsocial-data colorado build-all example-1k --data-dir data --plan
```

Then run every required stage in dependency order:

```bash
casmsocial-data colorado build-all example-1k --data-dir data
casmsocial-data colorado build-all example-10k --data-dir data
```

`build-all` does not download sources. If prerequisites are missing or have
unverified provenance, it exits before building and prints the action required
for every source. Accepted intermediate products are resumed using their
manifests. Profiles with runtime smoke gates also run the privacy-safe MPI
verification from the preceding section; use `--skip-runtime-verification`
only when intentionally deferring that acceptance step. A required but skipped
verification produces a `built_unverified` receipt rather than a passing one.

The command writes a compact receipt to
`data/local/colorado-front-range-builds/<profile>/manifest.json`. It records
the accepted manifest hash for each stage, the final runtime location, and the
local-only governance constraint. Repeating an unchanged build returns
`resumed: true`. Generated populations, OSM-derived destinations, DuckLake
files, runtimes, verification output, and receipts all remain local and must
not be committed or published.
