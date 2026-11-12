# casmsocial

[![Release](https://img.shields.io/github/v/release/ccgsem/casmsocial)](https://img.shields.io/github/v/release/ccgsem/casmsocial)
[![Build status](https://github.com/ccgsem/casmsocial/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/ccgsem/casmsocial/actions/workflows/ci.yml?query=branch%3Adevelop)
[![codecov](https://codecov.io/gh/ccgsem/casmsocial/branch/develop/graph/badge.svg)](https://codecov.io/gh/ccgsem/casmsocial)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ccgsem/casmsocial)](https://img.shields.io/github/commit-activity/m/ccgsem/casmsocial)
[![License](https://img.shields.io/github/license/ccgsem/casmsocial)](https://img.shields.io/github/license/ccgsem/casmsocial)

casmsocial is a Python framework for implementing agent-based models that simulate the dynamics of a synthetic population

- **Github repository**: <https://github.com/ccgsem/casmsocial/>
- **Documentation** <https://ccgsem.github.io/casmsocial/>

## Installation for new users

Start with a small local example before connecting a county-scale dataset or
other CASM services. You need Git (downloads the code), uv (installs Python and
packages), and MPI (runs the simulation). This guide selects **Python 3.13**;
`pyproject.toml` supports Python 3.10–3.13 and excludes 3.14 and later.
R, RStudio, and Quarto are not required for this repository's Python MVP.

| Computer | Setup path | What it supports |
|---|---|---|
| macOS | Terminal and the MPI instructions below | CASM Social runtime and local MVP |
| Windows with WSL2 | Ubuntu terminal after initial PowerShell setup | CASM Social runtime and local MVP |
| Windows without WSL | Separate PowerShell instructions below | Standalone no-MPI Repast development; CASM Social adaptation is still required |

Choose your platform and run its commands in order. Copy the commands without
adding a `$` or `%` prompt. Initial downloads include large packages such as
PyTorch and PyArrow. Install software with administrator approval when required;
install Python packages as your ordinary user. A password prompt may show no
characters while you type.

### 1. macOS: install the tools

Open **Terminal** from Applications → Utilities. Install Apple's compiler tools:

```sh
xcode-select --install
```

Finish the installer before continuing. If the tools are already installed,
continue. Install Homebrew following [brew.sh](https://brew.sh/), including the
printed **Next steps** that add it to your shell. Reopen Terminal, then run:

```sh
brew install uv open-mpi
git --version
uv --version
mpicxx --version
mpirun --version
```

Open MPI supplies `mpicxx`, the C++ compiler wrapper used to build Repast4Py.
If your team already provides a working MPICH installation, use it instead of
installing another MPI. See Repast's
[macOS MPI guide](https://repast.github.io/repast4py.site/macos_mpi_install.html)
for its compiler requirements. Continue to Step 3.

### 2. Windows with WSL2: install Ubuntu and the tools

Use a Windows version supported by
[Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
Open **PowerShell as administrator** from the Start menu and run:

```powershell
wsl --install -d Ubuntu
```

Restart if requested. Open **Ubuntu** from the Start menu, and create the Linux
username and password requested on first launch. In PowerShell, verify that
Ubuntu's version is `2`:

```powershell
wsl --list --verbose
```

If it shows `1`, use `wsl --set-version Ubuntu 2` before continuing.
For this path, run all following commands **inside Ubuntu**, not PowerShell:

```sh
sudo apt update
sudo apt install -y build-essential git curl ca-certificates tzdata openmpi-bin libopenmpi-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`tzdata` provides the time-zone database the model needs to read scenario
time zones; without it, `make mvp` stops with `No time zone found with key`.
The last line uses the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/).

**On an organization network that inspects HTTPS** (common on corporate
networks), the last line can fail with `SSL certificate problem: self-signed
certificate in certificate chain`. Ubuntu doesn't share Windows' certificate
store, so add your organization's root certificate to Ubuntu once. Get the
root certificate file (PEM format, `.crt`) from IT, save it in Windows'
Downloads folder, and run these lines in Ubuntu, replacing `org-root-ca.crt`
with its file name and `YourWindowsName` with your Windows user folder name:

```sh
sudo cp /mnt/c/Users/YourWindowsName/Downloads/org-root-ca.crt /usr/local/share/ca-certificates/org-root-ca.crt
sudo update-ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`update-ca-certificates` should report `1 added`. uv and Git then use the
same certificate store, so the remaining steps need no further change.
Close Ubuntu and reopen it so uv is on your command path, then check:

```sh
git --version
uv --version
mpicxx --version
mpirun --version
```

Keep this checkout and its virtual environment inside Ubuntu's home directory,
as in Step 3. Do not reuse a Windows-created environment in WSL.

### 3. Download CASM Social and install the locked environment

In macOS Terminal or Ubuntu:

```sh
mkdir -p ~/Projects
cd ~/Projects
git clone --branch main https://github.com/ccgsem/casmsocial.git
cd casmsocial
uv python install 3.13
CC=mpicxx CXX=mpicxx uv sync --locked --python 3.13
```

If you already cloned the repository, enter its existing folder instead of
cloning again. Ask your team if you should use a different branch from `main`.

uv creates `.venv` in the repository and installs the versions from `uv.lock`,
including the Git-pinned `casmsim` dependency and CPU PyTorch source configured
in `pyproject.toml`. The compiler variables apply to this installation command.
Wait for the downloads and native builds to finish. Do not use `sudo` or delete
the lockfile to work around errors.

You do not need to activate `.venv`: `uv run --locked` selects it for each
command. For contributors, optionally install the repository's commit hooks:

```sh
uv run --locked pre-commit install
```

`make install` is an existing convenience target that also installs those hooks;
it does not install system MPI or compiler tools for you.

### 4. Verify the installation

From the repository folder:

```sh
uv run --locked python --version
uv run --locked python -c "from importlib.metadata import version; from casmsocial import CitySocialModel; from mpi4py import MPI; from repast4py import context, schedule; print('CASM Social:', version('casmsocial')); print('Model:', CitySocialModel.__name__); print('MPI ranks:', MPI.COMM_WORLD.Get_size())"
```

Expect Python `3.13.x`, CASM Social `2.11.0` for this checkout, model
`CitySocialModel`, and MPI ranks `1`. Framework log messages during import are
normal. Version numbers can change when the team updates the repository.

### 5. Run the self-contained MVP

The MVP creates tiny synthetic inputs locally. You do not need team population
data, Box access, a casmdb service, or an LLM API key for this example. Its local
behavior adapter does not contact a remote LLM provider. The initial data build
may download DuckDB extensions, so keep internet access available.

```sh
make mvp
```

This target removes prior generated MVP artifacts, recreates the local DuckLake,
runs `config/mvp.yaml` with one MPI process, validates the outputs, and writes a
summary. Preserve previous MVP results elsewhere before rerunning if you need
those results for comparison. Do not point the demo at a production catalog.

Success produces:

- `data/output/mvp_summary.md` — readable run summary; open this first.
- `data/output/mvp_agent_log.parquet` — person locations over time.
- `data/output/mvp_behavior_log.parquet` — behavior observations.
- `examples/mvp/mvp.ducklake` — generated local DuckLake directory.

The current smoke configuration has 2 agents, 24 ticks, and 1 rank; validation
reports 48 rows in each of the agent and behavior logs. Inspect the summary in
your editor, or print it in the terminal:

```sh
cat data/output/mvp_summary.md
```

On Windows/WSL, `explorer.exe .` opens the repository in File Explorer.
For revalidation, two-rank runs, and other smoke scenarios, use
[examples/mvp/README.md](examples/mvp/README.md) and the
[MVP operator checklist](docs/mvp_operator_checklist.md). The full developer
suite is available with `make test`; it is broader than this first-run check.

### Native Windows without WSL: standalone Repast development

Repast's [native Windows guide](https://repast.github.io/repast4py.site/windows_install.html)
provides a prebuilt no-MPI wheel. It avoids an MPI installation and a C++
compiler, but **it is not a complete deployment of this CASM Social checkout**:
[`CitySocialModel`](casmsocial/citysim/citysocialmodel.py) and
[`CasmPop`](casmsocial/casmpop.py) import `mpi4py` directly. Repast's single-rank
communicator is exposed through `from repast4py import MPI`, not through a
replacement `mpi4py` package. The published wheel is Repast4Py 1.3.0; the
repository lockfile currently selects MPI-based 1.3.1.

For a native **64-bit Intel/AMD Windows** learning environment, install
**64-bit Python 3.13** from [Python.org](https://www.python.org/downloads/windows/),
including its launcher. Open an ordinary PowerShell window and run:

```powershell
py -3.13 --version
py -3.13 -c "import struct; print(struct.calcsize('P') * 8)"
py -3.13 -m venv "$env:USERPROFILE\venvs\casmsocial-repast-native"
$NativePython = "$env:USERPROFILE\venvs\casmsocial-repast-native\Scripts\python.exe"
& $NativePython -m pip install --upgrade pip
$env:R4PY_NO_MPI = "1"
& $NativePython -m pip install "https://github.com/Repast/repast4py/releases/download/v1.3.0/repast4py-1.3.0-cp313-cp313-win_amd64.whl"
& $NativePython -c "from repast4py import MPI, context, schedule; print('Ranks:', MPI.COMM_WORLD.Get_size()); print('Rank:', MPI.COMM_WORLD.Get_rank())"
& $NativePython -m pip check
```

The architecture check must print `64`; the communicator check should print
size `1` and rank `0`. This wheel targets `win_amd64`, not native ARM64 Python.
Using the executable path avoids PowerShell activation-policy changes. In a new
PowerShell session, set `$NativePython` and `R4PY_NO_MPI` again before continuing.

Write standalone Repast experiments using `from repast4py import MPI` and run
them with `& $NativePython your_script.py`. Do not mix MPI-based Repast or
`mpi4py` into this environment, and do not use `uv sync`, `uv run`, `make install`,
`make mvp`, or `mpirun` for this no-MPI environment. It is separate from the
project's locked runtime; dependencies of the wheel are resolved by pip.

To deploy CASM Social in this mode, maintainers must adapt direct MPI imports
and MPI-dependent behavior to the single-rank interface, check the required
APIs, lock a compatible Windows environment, and validate model initialization,
scheduling, observation, and an end-to-end scenario on Windows. Setting
`R4PY_NO_MPI` alone does not perform that adaptation. Use the macOS/WSL MPI path
for the current model runtime.

### Returning to your MPI checkout

Open macOS Terminal or Ubuntu, then:

```sh
cd ~/Projects/casmsocial
uv run --locked python --version
```

Before updating, review `git status` and save your changes. With a clean working
tree, use `git pull --ff-only`, then repeat the locked sync from Step 3. Keep
`CC=mpicxx CXX=mpicxx` on installs that may build Repast. If switching from a
no-MPI experiment, use a fresh terminal without `R4PY_NO_MPI` set.

### Installation troubleshooting

| Symptom | Next step |
|---|---|
| `brew` or `uv` not found | Reopen the terminal and complete the installer's shell-path instructions. Install Linux uv inside Ubuntu for WSL. |
| `make` not found | Complete Xcode command-line tools installation on macOS; install `build-essential` in Ubuntu. |
| `mpicxx` not found | Install the MPI packages for your platform, then check `mpicxx --version`. |
| Repast compilation fails | Check the compiler wrapper and rerun Step 3 with both `CC` and `CXX`. Capture the first compiler error. |
| MPI library or launcher mismatch | Use one MPI implementation for compilation and execution; ask the team before replacing an existing MPI. |
| Certificate verification fails | uv uses system certificates. Ask IT about organization certificates; keep TLS verification enabled. In WSL, add the organization root to Ubuntu as in Step 2. If Homebrew Git alone fails, prefix the sync command with `PATH="/usr/bin:/bin:$PATH"` to select Apple's Git. |
| `No time zone found with key America/New_York` | Install the time-zone database: `sudo apt install -y tzdata` in Ubuntu. |
| DuckDB cannot download an extension | Check network access and organizational certificate/proxy settings. The first MVP build may need extension downloads. |
| MVP reads unexpected data | Check `.env` and inherited `CASMSOCIAL_DUCKLAKE_URI`/data-path settings; use the local demo configuration. |
| Missing files with `config/casmsocial.yaml` | That config needs its own dataset. Start with `make mvp`, which generates its inputs. |
| Native Windows wheel rejected | Check Python 3.13, 64-bit x86 Python, and the `cp313`/`win_amd64` wheel filename. |

When asking for help, share your OS, terminal type, exact command, and first
error message. Omit credentials and private data. These instructions follow the
current source and upstream installation guides. Steps 2–5 of the WSL path
were run in a fresh Ubuntu 24.04 environment (2026-11-12) through a successful
`make mvp`; the PowerShell setup and the native Windows path have not been
tested on a Windows machine.

## Configure your own data (after the MVP)

For local runs against your own data, copy `.env.example` to `.env` and set
`CASMSOCIAL_DATA_PATH` and `CASMSOCIAL_DUCKLAKE_PATH`.

By default the DuckLake catalog is the SQLite file
`$CASMSOCIAL_DUCKLAKE_PATH/metadata.sqlite`. Set `CASMSOCIAL_DUCKLAKE_URI` to
use another catalog, for example a DuckDB catalog served over DuckDB's beta
Quack protocol (`ducklake:quack:host:9494`, with the token in
`CASMSOCIAL_DUCKLAKE_QUACK_TOKEN`). That lets several processes, such as
parallel dataset builds, commit to the same lake without SQLite's file
locking. Parquet data stays under `$CASMSOCIAL_DUCKLAKE_PATH/storage`, which
every process must be able to reach. Model runs and
`python -m casmsocial.network_partitioner_ducklake` (also `--catalog-uri`)
honor it. Quack uses plain HTTP only for local servers, so a remote catalog
server needs a TLS-terminating proxy.

To move an existing lake's SQLite catalog into a DuckDB file that a Quack
server can serve, stop all writers and run
`python -m casmsocial.ducklake_migrate --ducklake-path <lake> --to <lake>/catalog.duckdb`.
It flushes inlined rows into Parquet, copies only the catalog metadata (Parquet
files are not touched), verifies every table's row count and contents against
the source, and creates the target only if everything matches. The SQLite
catalog stays usable until the first write through the new one.

To build a Docker image for `casmsocial`:

```bash
docker build -t casmsocial . -f Dockerfile
```

## Using the installed environment

The installation above already creates `.venv`; do not recreate it with
`python -m venv`. Prefer `uv run --locked` for commands. If you want to use an
activated environment in macOS Terminal or Ubuntu instead:

```sh
source .venv/bin/activate
```

Run `deactivate` when finished. Native Windows no-MPI instructions use their
own separate environment and do not use this activation command.

## Quickstart: running the model
There are three ways to run the model

1. Run from the command line using `uv run`
2. Run from the command line using virtualenv
3. Run in the Docker image

The files in `config/` are runtime launch configs for direct local runs.
Canonical casmdb scenario definitions are maintained in
`scenarios/casmsocial/*.yaml` and registered with
`scripts/register_casmsocial.py`.
If casmdb has `CASMDB_API_TOKEN` set, export the same value before running
the script; it sends it as `Authorization: Bearer <token>`.

The registration also declares a few editable run parameters, the row schemas
of the `agent_log`, `schedule_occupancy` and `social_interactions` channels,
and what their rows mean (`model_metadata.output_descriptors`). casmservice's
generic GUI uses these together with the reviewed manifest in
`presentation/casmsocial.runtime.json`: a parameter form, an occupancy time
series and latest-tick summary, and tables. `agent_log` `x`/`y` are the
longitude/latitude (WGS84 degrees) of each person's current place, so they
follow moves; they are null when the place has no coordinates or is not known
on the logging rank. The GUI shows them in the agent table and on an agent
location map (casmservice runtime-6 `geographic-points@1.1.0`), which leaves
rows without coordinates off the map and says how many. `agent_log` also
declares the `casmsocial.agent_log.v2` schema contract, which is the default
AgentLogger columns. casmservice applies it to every catalog run, so reads of
completed runs return the runner's exact Arrow schema (int32 `tick`/`rank`,
nullable `x`/`y`). A run that sets `observers.agent_log_columns` may add
columns but must keep `x`, `y` and `place_id`; otherwise its agent_log reads
fail with HTTP 400. The occupancy
chart needs a single-rank run (one row per tick) and
`observers.schedule_occupancy_log.enabled: true`. After changing the record,
re-register with `--force-update-model` and re-check the manifest with
casmservice's `scripts/presentation_manifest.py check`.

To run (option 1):

```bash
% uv run mpirun -n 1 python -m casmsocial config/casmsocial.yaml
```

For a self-contained smoke scenario that creates its own tiny DuckLake input,
see [examples/mvp/README.md](examples/mvp/README.md). For expected outputs and
quick failure triage, see [docs/mvp_operator_checklist.md](docs/mvp_operator_checklist.md).

### Input Tables

YAML configs identify DuckLake source tables with `places.table`,
`households.table`, `persons.table`, `activities.table`, and optionally
`social_networks.table`. Social-network rows are timeless potential ties, not
fixed-hour contacts: `person_id_a`, `person_id_b`, and `network_kind` are
required, while `tie_strength` is optional. Endpoints must be canonical
(`person_id_a < person_id_b`) and each `(person_id_a, person_id_b,
network_kind)` may appear once. Simulation behavior derives in-person
encounters from schedule overlap and co-location; it may derive remote messages
from the same ties without co-location.
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
  --build-arg UV_INSECURE_HOST=download-r2.pytorch.org \
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

That layout writes one Parquet file per tick, rank and output, which adds up in
long runs and large sweeps. MITRE HPC project directories allow 1 million
files; a 750-job, 96-tick sweep on 8 ranks can exceed that. Two options write
fewer, larger files:

```yaml
observers.partition_by_tick: false   # run_id=…/rank=…/, tick kept as a column
observers.flush_ticks: 24            # buffer 24 ticks per file; the rest is written at the end
```

With both set, a run writes about `ticks / flush_ticks` files per rank and
output instead of `ticks`. Readers using
`read_parquet(..., hive_partitioning=true)` see the same `run_id`, `tick` and
`rank` columns either way. `flush_ticks` holds that many ticks of rows in
memory per output, so keep it modest for full-population runs.

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

## Live Arrow Flight Observation Server

`CasmPop` can optionally host a live Arrow data server inside the simulation
process, letting an external client (e.g. casmservice) pull an observer's
most recent output table -- via `CasmPop.get_observer_output_tables()` --
while the run is still in progress, instead of waiting for parquet files to
be flushed to disk. It speaks Apache Arrow Flight, which ships inside the
`pyarrow` wheel -- already a hard dependency here -- so there's no extra to
install and no optional dependency to manage.

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

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
