# casmsocial

[![Release](https://img.shields.io/github/v/release/ccgsem/casmsocial)](https://img.shields.io/github/v/release/ccgsem/casmsocial)
[![Build status](https://github.com/ccgsem/casmsocial/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/ccgsem/casmsocial/actions/workflows/ci.yml?query=branch%3Adevelop)
[![codecov](https://codecov.io/gh/ccgsem/casmsocial/branch/develop/graph/badge.svg)](https://codecov.io/gh/ccgsem/casmsocial)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ccgsem/casmsocial)](https://img.shields.io/github/commit-activity/m/ccgsem/casmsocial)
[![License](https://img.shields.io/github/license/ccgsem/casmsocial)](https://img.shields.io/github/license/ccgsem/casmsocial)

casmsocial is a Python framework for implementing agent-based models that simulate the dynamics of a synthetic population

- **Github repository**: <https://github.com/ccgsem/casmsocial/>
- **Documentation** <https://ccgsem.github.io/casmsocial/>

## Installation

Install the environment with

```bash
export CC=mpicxx; export CXX=mpicxx
make install
```

For local runs against your own data, copy `.env.example` to `.env` and set
`CASMSOCIAL_DATA_PATH` and `CASMSOCIAL_DUCKLAKE_PATH`.

To build a Docker image for `casmsocial`:

```bash
docker build -t casmsocial . -f Dockerfile
```

## Launch the modeling environment:
First create the virtual environments with

```bash
% python -m venv .venv
```

To launch the virtualenv, run

```bash
% source ./.venv/bin/activate
(casmsocial) ...
```

## Quickstart: running the model
There are three ways to run the model

1. Run from the command line using `uv run`
2. Run from the command line using virtualenv
3. Run in the Docker image

The files in `config/` are runtime launch configs for direct local runs.
Canonical casmdb scenario definitions are maintained in
`scenarios/casmsocial/*.yaml` and registered with
`scripts/register_casmsocial.py`.

To run (option 1):

```bash
% uv run mpirun -n 1 python -m casmsocial config/casmsocial.yaml
```

For a self-contained smoke scenario that creates its own tiny DuckLake input,
see [examples/mvp/README.md](examples/mvp/README.md). For expected outputs and
quick failure triage, see [docs/mvp_operator_checklist.md](docs/mvp_operator_checklist.md).

### Input Tables

YAML configs identify DuckLake source tables with `places.table`,
`households.table`, `persons.table`, `activities.table`, and `contacts.table`.
`Place` agents represent physical locations. `Household` agents represent social
household units loaded from `households.table`, often named `hh` in input
schemas; each household links to a physical `Place` and to its member `Person`
agents. If a household row does not include `place_id`, casmsocial uses `sp_id`
as the household's physical place id. Person home assignment uses `sp_hh_id` to
find the household when `households.table` is configured, then resolves that
household to its linked place for rank ownership and place projection setup.

To run with the virtual environment (option 2):

```bash
% source ./.venv/bin/activate
(casmsocial)
(casmsocial) mpirun -n 1 python -m casmsocial config/casmsocial.yaml
....
(casmsocial) deactivate
%
```

To run in a Docker container (option 3):

```bash
docker build --target prod \
  --build-arg UV_INSECURE_HOST=download.pytorch.org \
  -t casmsocial:local .

docker run --rm --init \
  --mount type=bind,source="$(pwd)/data",target=/app/data \
  -e CASMSOCIAL_DATA_PATH=/app/data \
  -e CASMSOCIAL_DUCKLAKE_PATH=/app/data/datalakehouse \
  casmsocial:local \
  mpirun -n 1 python -m casmsocial config/casmsocial.yaml
```

The production image installs the project into `/app` and puts `/app/.venv/bin`
on `PATH`, so use `python` directly inside the container. The repository
`.dockerignore` excludes `data/`; mount the host `data` directory when running
configs that read the local DuckLake catalog or write observer output under
`data/output`.

### Docker Compose MPI

For local multi-container MPI validation, build the optional Compose MPI image
and run the smoke check:

```bash
make docker-mpi-build
make docker-mpi-smoke
make docker-mpi-down
```

`docker-compose.mpi.yaml` starts `rank0` and `rank1`, and `config/mpi-hosts`
maps one MPICH slot to each service. To run the MVP scenario through the
Compose network:

```bash
make docker-mpi-up
docker compose -f docker-compose.mpi.yaml -p casmsocial-mpi exec -T rank0 \
  mpirun -hostfile config/mpi-hosts -n 2 python -m casmsocial config/mvp.yaml
make docker-mpi-down
```

The MVP config proves cross-container MPI launch, but it does not prove
nonzero-rank model output by itself: with no partition table, all MVP places
fall back to `partition.default_rank: 0`. Use a partitioned scenario, such as
the local `make mvp-2rank` path or a scenario-specific partition table, when
you need output rows from every rank.

The Compose file uses `CASMSOCIAL_MPI_DATA_PATH` and
`CASMSOCIAL_MPI_DUCKLAKE_PATH` as optional overrides so the repository `.env`
does not accidentally inject host-only paths into the containers.

## Code Quality

The repository ships with dedicated helpers for the standard Python code-quality trio:

- `make format` – runs `black` followed by `isort` so files stay auto-formatted.
- `make lint` – runs `flake8` with the settings defined in `pyproject.toml`.
- `make check` – executes the heavier-weight pipeline (`uv lock`, `pre-commit`, and `mypy`) when you need the full suite.

All commands rely on `uv run` so they automatically use the project virtual environment. Running `make format` before committing is usually enough to clean up imports and spacing, and `make lint` provides a quick verification step without running the full check pipeline.

## Road Networks

`casmsocial` now has an initial scaffold for OpenStreetMap-backed road networks. The runtime expects three prebuilt artifacts:

- `road_nodes`
- `road_edges`
- `place_road_snap`

These are configured with:

```yaml
roads.enabled: false
roads.nodes.file: ''
roads.edges.file: ''
roads.place_snap.file: ''
roads.mode: 'drive'
roads.time_model: 'validated_gap'
```

When road support is enabled, weekday activity plans can insert routed `Leg` elements between activities. Each leg can carry origin / destination place ids, snapped road node ids, route distance, and routed travel time.

The preprocessing entry point is [scripts/build_road_network.py](scripts/build_road_network.py). The MVP path supports dependency-light OSM XML extracts (`.osm` / `.xml`), drive-mode road filtering, nearest-node place snapping, and Parquet export:

```bash
uv run python scripts/build_road_network.py \
  --osm-file data/roads.osm \
  --places-file data/places.parquet \
  --nodes-out data/road_nodes.parquet \
  --edges-out data/road_edges.parquet \
  --snaps-out data/place_road_snap.parquet \
  --report-out data/road_artifacts.json
```

The builder expects place records with `sp_id` or `place_id` plus `longitude` / `latitude` columns. The exported Parquet files can be referenced directly by `roads.nodes.file`, `roads.edges.file`, and `roads.place_snap.file`.
The optional JSON report records source paths, output paths, road-node and edge
counts, snapped-place coverage, road-type counts, and aggregate length and
travel-time totals.

The MVP smoke suite includes `make mvp-built-roads`, which builds road Parquets
from `examples/mvp/roads.osm` and runs the routed MVP scenario against those
generated files.

It also includes `make mvp-delta-state`, which enables changed-only agent-state
logging, reconstructs dense state rows, and validates them against the full MVP
agent and behavior logs. The generated `data/output/mvp_delta_state_validation.json`
reports row and file-size reduction plus changed-agent counts by run and tick. The same
target also loads the delta outputs into `examples/mvp/mvp.ducklake` under the
`mvp_observability` schema for interactive inspection and writes
`data/output/mvp_agent_state_delta_ducklake_report.md` with example queries and
results.

## Behavior Logging

`casmsocial` can now log lightweight behavior-engine state for each person at each tick using `BehaviorLogger`.

Enable it with:

```yaml
observers.output_dir: 'data/output'
observers.behavior_log.enabled: true
observers.behavior_log_file: 'behavior_log.parquet'
```

All observer outputs use `observers.output_dir`; observer file settings must be
filenames only. Output datasets are Hive-partitioned by `run_id`, `tick`, and
`rank`. The run id defaults to `seed_<random.seed>` when `simulation.run_id` is
omitted, and `random_seed` is stored as row metadata.

The behavior log records:

- `run_id`
- `random_seed`
- `tick`
- `rank`
- `agent_id`
- `place_id`
- `rank_place_id`
- `last_decision`
- `last_llm_summary`
- `last_memory_event_type`
- `last_plan_adjustment_requested_kind`
- `last_plan_adjustment_applied`
- `last_plan_adjustment_skip_reason`
- `last_plan_adjustment_kind`
- `last_plan_adjustment_delay_minutes`
- `last_plan_adjustment_target_activity_id`
- `last_plan_adjustment_target_place_id`
- `safety_signal`
- `social_signal`
- `obligation_signal`
- `schedule_signal`
- `reply_signal`

This is intended for inspecting the local `LLMBehaviorEngine` state during scenario runs without relying on debug logs. The latest values come from the most recent stored behavior-memory trace for each person.

For lower-volume state output, enable changed-only agent state logging:

```yaml
observers.delta_agent_state.enabled: true
observers.delta_agent_state_file: 'agent_state_delta.parquet'
observers.delta_agent_state_audit_file: 'agent_state_delta_audit.parquet'
```

`DeltaAgentStateLogger` writes a full state row only when an agent's normalized
state differs from the last state observed on the local rank. Rows include
`run_id`, `random_seed`, `state_hash`, and `change_mask` for reconstruction and
debugging. It also writes one audit row per run, rank, and tick with
`agents_evaluated` and `agents_changed`, so ticks with no state changes remain
visible.

To reconstruct dense state rows from the changed-only output:

```bash
uv run python scripts/reconstruct_agent_state.py \
  --delta-log output/agent_state_delta.parquet \
  --audit-log output/agent_state_delta_audit.parquet \
  --output output/agent_state_reconstructed.parquet \
  --overwrite
```

The reconstruction validates the audit counts against the delta rows and writes
Hive-partitioned Parquet rows by `run_id`, `tick`, and `rank` with
`source_tick`, the tick where that agent's state last changed.

To load the changed-only output, audit rows, reconstructed rows, and validation
metrics into the local MVP DuckLake:

```bash
uv run python scripts/load_agent_state_delta_ducklake.py \
  --ducklake-path examples/mvp/mvp.ducklake \
  --delta-log data/output/mvp_agent_state_delta.parquet \
  --audit-log data/output/mvp_agent_state_delta_audit.parquet \
  --reconstructed-log data/output/mvp_agent_state_reconstructed.parquet \
  --validation-report data/output/mvp_delta_state_validation.json
```

This creates `mvp_observability.agent_state_delta`,
`mvp_observability.agent_state_delta_audit`,
`mvp_observability.agent_state_reconstructed`,
`mvp_observability.agent_state_delta_validation`, and
`mvp_observability.agent_state_delta_changes_by_tick`.

To generate example DuckLake queries and a Markdown report from the loaded
tables:

```bash
make mvp-delta-state-report
```

or run the script directly:

```bash
uv run python scripts/report_agent_state_delta_ducklake.py \
  --ducklake-path examples/mvp/mvp.ducklake \
  --output data/output/mvp_agent_state_delta_ducklake_report.md
```

Interpretation of the adjustment fields:

- `last_plan_adjustment_requested_kind` is the adjustment requested by the behavior engine, even if it was later skipped.
- `last_plan_adjustment_applied` indicates whether that request actually changed the plan.
- `last_plan_adjustment_skip_reason` explains why a request was skipped. Common values include `no_slack`, `no_future_activity`, and `no_eligible_future_activity`.
- `last_plan_adjustment_kind` is the adjustment that was actually applied. It is empty when the request was skipped.
- `last_plan_adjustment_delay_minutes` records the applied delay for `defer_next_activity`.
- `last_plan_adjustment_target_activity_id` and `last_plan_adjustment_target_place_id` identify the activity and place that were targeted by the applied or attempted adjustment.

## Local Behavior Engine

The local behavior-engine scaffold is configured with the `behavior.*` and `behavior.llm.*` keys:

```yaml
behavior.engine: 'default'
behavior.llm.enabled: false
behavior.llm.deliberation_interval: 60
behavior.llm.max_memory_events: 20
behavior.llm.signal_cap: 1.5
behavior.llm.memory_decay: 0.65
behavior.activity_semantics.social_ids: []
behavior.activity_semantics.flexible_ids: []
behavior.activity_semantics.mandatory_ids: []
behavior.activity_semantics.travel_sensitive_ids: []
```

Meaning of each key:

- `behavior.engine`
  Selects the person behavior engine. Use `'default'` for the baseline deterministic engine or `'llm_local'` for the local no-network deliberation scaffold.
- `behavior.llm.enabled`
  Enables the local behavior-engine scaffold even if `behavior.engine` is left at `'default'`.
- `behavior.llm.deliberation_interval`
  Minimum number of ticks between coarse deliberation passes when no salient event has occurred.
- `behavior.llm.max_memory_events`
  Number of recent episodic-memory entries exposed to the local adapter and retained in the short-term memory window.
- `behavior.llm.signal_cap`
  Saturation cap for accumulated appraisal signals such as safety, social, obligation, schedule, and reply pressure.
- `behavior.llm.memory_decay`
  Decay factor applied when appraisal traces are reconstructed from memory across later ticks.
- `behavior.activity_semantics.social_ids`
  Activity ids that should be treated as social by the local behavior engine.
- `behavior.activity_semantics.flexible_ids`
  Activity ids that should be treated as flexible in addition to the conservative defaults.
- `behavior.activity_semantics.mandatory_ids`
  Activity ids that should be treated as mandatory when future policy logic needs that distinction.
- `behavior.activity_semantics.travel_sensitive_ids`
  Activity ids that should be treated as travel-sensitive in addition to the conservative defaults.

Practical guidance:

- Lower `behavior.llm.deliberation_interval` makes agents reconsider behavior more often.
- Higher `behavior.llm.max_memory_events` makes recent message history matter for longer.
- Higher `behavior.llm.signal_cap` lets repeated messages accumulate stronger influence before saturating.
- Lower `behavior.llm.memory_decay` makes old message influence fade faster.
- Activity-semantics overrides are additive: they layer on top of conservative defaults rather than replacing them.
- By default, no activity is treated as social; social behavior only activates when `behavior.activity_semantics.social_ids` is configured.

Current constrained adjustment support:

- safety-driven `stay_home` decisions from warnings or urgent recommendations can emit `preserve_home_activity`
- this rewrites the active plan from the current minute forward into a home activity for the remainder of the day
- coordination or travel-delay guidance can emit `defer_next_activity`
- this shifts only the next future activity later, bounded by available slack before the following activity or end of day
- narrow cancel or skip-style coordination / recommendation messages can emit `skip_flexible_activity`
- this removes only the next future flexible activity from the active plan and leaves the rest of the day intact
- social-cancel messages can emit `cancel_social_activity`
- this removes only the next future social activity and only when that activity id is marked via `behavior.activity_semantics.social_ids`
- without a social override, the same message leaves non-social activities unchanged
- `BehaviorLogger` records both requested and applied adjustment state, including skip reasons and target activity metadata

Example scenario override:

```yaml
behavior.engine: 'llm_local'
behavior.activity_semantics.social_ids: [1]
```

With that configuration, a future activity with `activity_id = 1` is treated as social. A message such as:

- kind: `coordination`
- topic: `cancel_social`

can trigger `cancel_social_activity`, which removes the next future social activity from the active plan. Without the `social_ids` override, the same message does not cancel that activity.

---

## Live Arrow/Ice Observation Server

`CasmPop` can optionally host a live Arrow data server inside the simulation
process, letting an external client (e.g. casmservice) pull an observer's
most recent output table -- via `CasmPop.get_observer_output_tables()` --
while the run is still in progress, instead of waiting for parquet files to
be flushed to disk. This is opt-in and requires the `service` extra:

```bash
uv sync --extra service
```

`zeroc-ice` only publishes wheels for Python >= 3.12; the `service` extra is
unavailable on older interpreters.

Enable it with:

```yaml
observers.arrow_server.enabled: true
observers.arrow_server.host: '127.0.0.1'
```

The server binds an OS-assigned ephemeral port on rank 0 only (each MPI rank
is a separate process with its own local agents; only rank 0's local
observer tables are exposed in this first iteration). Once started, it
writes `arrow_endpoint.txt` (`host:port`) into the simulation's current
working directory, so a launcher such as casmservice's `Repast4pyBackend`
(which already runs the subprocess with that directory as its `run_dir`) can
discover where to connect without any additional configuration.

The Ice IDL contract lives at `slice/arrowservice/ArrowService.ice` (kept
byte-identical to casmservice's copy of the same file -- there is no shared
package between the two repos, only the wire contract). Regenerate the
committed Python bindings under `arrowservice/` whenever that file changes:

```bash
uv run python scripts/build_slice.py
```

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
