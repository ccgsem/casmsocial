# MVP Operator Checklist

Use this checklist when you need to prove the local MVP smoke scenario is
healthy and inspectable.

## Run

From the repository root:

```bash
make mvp
```

The target removes generated MVP artifacts, recreates the local MVP DuckLake,
runs `config/mvp.yaml` with one MPI rank, validates both MVP logs, and writes a
Markdown summary.

## Expected Local Outputs

After a successful run, these generated artifacts should exist:

- `data/output/mvp_summary.md`
- `data/output/mvp_agent_log.parquet`
- `data/output/mvp_behavior_log.parquet`
- `examples/mvp/mvp.ducklake`

The generated MVP DuckLake contains the standard people, household, place,
activity, and partition fixtures plus a tiny routable road fixture:
`rti_synth_pop_v2_dmv_100.road_nodes`,
`rti_synth_pop_v2_dmv_100.road_edges`, and
`rti_synth_pop_v2_dmv_100.place_road_snap`.

The validation output should report:

```text
MVP agent log valid: data/output/mvp_agent_log.parquet (rows=48, runs=1, agents=2, ticks=24, ranks=1)
MVP behavior log valid: data/output/mvp_behavior_log.parquet (rows=48, runs=1, agents=2, ticks=24, ranks=1)
MVP summary written: data/output/mvp_summary.md (rows=48, runs=1, agents=2, ticks=24, ranks=1)
```

## Inspect

Open `data/output/mvp_summary.md` first. It gives the run shape, decision counts,
memory-event counts, applied plan-adjustment count, and signal averages without
requiring parquet tooling.

MVP output datasets are partitioned by `run_id`, `tick`, and `rank`; the default
run id for the smoke config is `seed_42`.

Use these helpers for existing outputs:

```bash
make mvp-check
make mvp-report
```

Use this helper to prove the MVP process starts and completes with two MPI
processes:

```bash
make mvp-2rank
```

The two-rank smoke path enables `partitions.mvp_two_rank_place_partitions`,
validates output from both ranks, and writes separate `data/output/mvp_2rank_*`
artifacts.

For a containerized MPI launcher check that avoids the local host MPI stack,
use the Docker Compose MPI helpers:

```bash
make docker-mpi-build
make docker-mpi-smoke
make docker-mpi-down
```

`docker-compose.mpi.yaml` starts `rank0` and `rank1`, and `config/mpi-hosts`
maps one MPICH slot to each service. Running `config/mvp.yaml` through that
Compose network proves cross-container MPI launch, but the default MVP config
has no partition table and assigns all places to `partition.default_rank: 0`.
Use the partitioned `make mvp-2rank` scenario, or another scenario with a
rank partition table, when you need observer output from every rank.

Use this helper to prove the MVP process can build routed plans from the road
fixture:

```bash
make mvp-routed
```

The routed smoke path enables the road tables, validates the generated logs,
writes separate `data/output/mvp_routed_*` artifacts, and checks routed leg metadata
for the expected road nodes, distances, and travel times. It also writes
`data/output/mvp_routed_plan_validation.json` with the routed-leg validation summary.

Use this helper to prove the MVP process can build road artifacts from OSM XML
and run against those generated files:

```bash
make mvp-built-roads
```

The built-road smoke path writes `data/output/mvp_built_road_*.parquet`,
`data/output/mvp_built_place_road_snap.parquet`, and
`data/output/mvp_built_road_artifacts.json`, runs the routed MVP scenario against
those generated files, and writes separate `data/output/mvp_built_roads_*` run
artifacts plus `data/output/mvp_built_roads_plan_validation.json`.

Use this helper to prove changed-only agent-state logging can be reconstructed
and matched against full MVP agent and behavior logs:

```bash
make mvp-delta-state
```

The delta-state smoke path enables `DeltaAgentStateLogger`, writes
`data/output/mvp_agent_state_delta.parquet` and
`data/output/mvp_agent_state_delta_audit.parquet`, reconstructs dense state rows at
`data/output/mvp_agent_state_reconstructed.parquet`, and writes
`data/output/mvp_delta_state_validation.json`. The validation report includes row
reduction, file-size reduction, and changed-agent counts by run and tick so the output
volume savings are visible alongside correctness. The target also loads those
outputs into `examples/mvp/mvp.ducklake` under `mvp_observability` with tables
for the delta rows, audit rows, reconstructed rows, validation summary, and
changed-agent counts by run and tick. It then writes
`data/output/mvp_agent_state_delta_ducklake_report.md` with copyable SQL examples
and query results over those tables.

To reload existing delta-state artifacts into DuckLake without rerunning the
simulation:

```bash
uv run python scripts/load_agent_state_delta_ducklake.py \
  --ducklake-path examples/mvp/mvp.ducklake \
  --delta-log data/output/mvp_agent_state_delta.parquet \
  --audit-log data/output/mvp_agent_state_delta_audit.parquet \
  --reconstructed-log data/output/mvp_agent_state_reconstructed.parquet \
  --validation-report data/output/mvp_delta_state_validation.json
```

To regenerate only the query report from already-loaded DuckLake tables:

```bash
make mvp-delta-state-report
```

The report includes examples for the efficiency snapshot, changed agents by
tick, per-rank audit rows, most frequently changed agents, change-mask counts,
reconstructed-state freshness, and latest reconstructed agent state.

After the standard, two-rank, routed, built-road, and delta-state smoke paths
have run, write the machine-readable artifact manifest with:

```bash
make mvp-manifest
```

The manifest validates all five output sets and writes
`data/output/mvp_manifest.json` with run summaries, artifact sizes, and SHA-256
checksums. The manifest also covers the routed plan validation reports,
generated road Parquets, road-build report, and delta-state reconstruction
artifacts and DuckLake query report.

Verify the generated artifacts against the manifest with:

```bash
make mvp-verify-manifest
```

Run the full local proof sequence with the same target CI uses:

```bash
make mvp-all
```

This runs the standard, two-rank, routed, built-road, and delta-state smoke
paths, then writes and verifies `data/output/mvp_manifest.json`.

List the artifact paths retained locally and uploaded by CI with:

```bash
make mvp-artifacts
```

Use this helper to remove generated MVP artifacts:

```bash
make mvp-clean
```

## CI Artifact

GitHub Actions runs `make mvp-all` in the `CI` workflow on pushes and pull
requests to `develop` and `main`. A successful run uploads the `mvp-output`
artifact, downloads that uploaded artifact back into the job, and verifies it
against `mvp_manifest.json`. The artifact contains the paths printed by
`make mvp-artifacts`, including the manifest itself, all five run summaries and
logs, routed plan validation reports, generated road Parquets, and the
road-build report, plus the delta-state validation report, reconstructed
agent-state logs, and DuckLake query report.

Find it on the workflow run page under the run artifacts section.

## Failure Triage

If `make mvp` fails during DuckLake creation, run:

```bash
make mvp-clean
make mvp
```

If validation fails, inspect the message first. The most useful classes of
failure are missing paths, missing columns, unexpected row counts, and
per-agent or per-tick coverage mismatches.

If CI fails but the local run passes, compare the failing workflow step:

- `Run quality checks`: run `make check`
- `Run tests`: run `make test`
- `Run MVP proof suite`: run `make mvp-all`
- `List MVP output artifact paths`: run `make mvp-artifacts`
- `Upload MVP output`: verify every path printed by `make mvp-artifacts` exists
- `Verify downloaded MVP output artifact`: download the `mvp-output` artifact
  and run `uv run python scripts/verify_mvp_manifest.py --manifest <download>/mvp_manifest.json --artifact-root <download>`

For a local `make mvp-all` failure, rerun the individual target named near the
failing output: `make mvp`, `make mvp-2rank`, `make mvp-routed`,
`make mvp-built-roads`, `make mvp-delta-state`, `make mvp-manifest`, or
`make mvp-verify-manifest`.
