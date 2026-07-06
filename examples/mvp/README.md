# MVP Example

This example creates a tiny local DuckLake dataset and runs the MVP scenario
from `config/mvp.yaml` with one MPI rank.

```bash
make mvp
```

The target removes generated MVP artifacts, creates the local DuckLake, runs the
scenario, validates the output logs, and writes a compact Markdown summary. It
runs these commands:

```bash
uv run python scripts/clean_mvp_artifacts.py
uv run python scripts/create_mvp_ducklake.py
CASMSOCIAL_DATA_PATH=examples/mvp \
CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
uv run mpirun -n 1 python -m casmsocial config/mvp.yaml
uv run python scripts/validate_mvp_output.py
uv run python scripts/summarize_mvp_output.py
```

Expected outputs:

- `output/mvp_summary.md`
- `output/mvp_agent_log.parquet`
- `output/mvp_behavior_log.parquet`

The behavior log is the primary MVP artifact. It records one row per person per
tick with the local behavior engine's latest decision, summary, memory event,
plan-adjustment metadata, and signal values. The agent log records the matching
per-person location snapshot for each tick.

The generated DuckLake also includes a tiny routable road fixture in
`rti_synth_pop_v2_dmv_100.road_nodes`,
`rti_synth_pop_v2_dmv_100.road_edges`, and
`rti_synth_pop_v2_dmv_100.place_road_snap`. The default MVP config keeps
`roads.enabled: false`; those tables are present to support a routed MVP smoke
path.

Run that routed smoke path with:

```bash
make mvp-routed
```

This enables the road tables, writes separate `output/mvp_routed_*` artifacts,
validates the normal output logs, and checks that the model built routed legs
with the expected node, distance, and travel-time metadata. It also writes
`output/mvp_routed_plan_validation.json`.

Run the generated-road-artifact smoke path with:

```bash
make mvp-built-roads
```

This builds Parquet road artifacts from `examples/mvp/roads.osm` and
`examples/mvp/road_builder_places.csv`, runs the routed MVP scenario against
those generated files, and writes `output/mvp_built_*` artifacts plus
`output/mvp_built_road_artifacts.json` and
`output/mvp_built_roads_plan_validation.json`.

Revalidate existing MVP logs with:

```bash
make mvp-check
```

Regenerate the Markdown summary for an existing behavior log with:

```bash
make mvp-report
```

Remove generated MVP artifacts with:

```bash
make mvp-clean
```

Run the two-process MPI smoke path with:

```bash
make mvp-2rank
```

This target enables `partitions.mvp_two_rank_place_partitions`, validates that
both ranks write output, and stores the generated artifacts in
`output/mvp_2rank_*`.

Run the changed-only agent-state smoke path with:

```bash
make mvp-delta-state
```

This writes `output/mvp_agent_state_delta.parquet`,
`output/mvp_agent_state_delta_audit.parquet`,
`output/mvp_agent_state_reconstructed.parquet`, and
`output/mvp_delta_state_validation.json`. It also loads those outputs into the
local DuckLake as `mvp_observability.agent_state_delta`,
`mvp_observability.agent_state_delta_audit`,
`mvp_observability.agent_state_reconstructed`,
`mvp_observability.agent_state_delta_validation`, and
`mvp_observability.agent_state_delta_changes_by_tick` with run-aware change
counts. The target also writes
`output/mvp_agent_state_delta_ducklake_report.md` with example SQL and result
tables.

Regenerate only that DuckLake query report with:

```bash
make mvp-delta-state-report
```

After the standard, two-rank, routed, built-road, and delta-state smoke paths
have run, write a JSON manifest for the generated artifacts with:

```bash
make mvp-manifest
```

Verify the generated artifacts against that manifest with:

```bash
make mvp-verify-manifest
```

Run the full local proof sequence with:

```bash
make mvp-all
```

This runs the standard, two-rank, routed, built-road, and delta-state smoke
paths, then writes and verifies the manifest.

List the same artifact paths that CI uploads with:

```bash
make mvp-artifacts
```

GitHub Actions runs `make mvp-all` in CI, derives the upload path list from
`make mvp-artifacts`, uploads the single-rank, two-rank, routed, built-road,
delta-state, and manifest outputs as the `mvp-output` artifact, then downloads
that artifact back into the job and verifies it against `mvp_manifest.json`.

For run expectations and failure triage, see the
[MVP operator checklist](../../docs/mvp_operator_checklist.md).
