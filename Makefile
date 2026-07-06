DOCKER_MPI_COMPOSE ?= docker compose -f docker-compose.mpi.yaml -p casmsocial-mpi
DOCKER_MPI_HOSTFILE ?= config/mpi-hosts
DOCKER_MPI_RANKS ?= 2
DOCKER_MPI_UV_INSECURE_HOST ?= download.pytorch.org

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: format
format: ## Auto-format the code with black and isort
	@echo "🚀 Formatting code with black"
	@uv run black casmsocial tests
	@echo "🚀 Sorting imports with isort"
	@uv run isort casmsocial tests

.PHONY: lint
lint: ## Run flake8 lint checks
	@echo "🚀 Linting code with flake8"
	@uv run flake8 casmsocial tests

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: mvp
mvp: mvp-clean ## Clean artifacts, create the local MVP DuckLake, run the scenario, validate output, and summarize it
	@echo "🚀 Creating MVP DuckLake"
	@uv run python scripts/create_mvp_ducklake.py
	@echo "🚀 Running MVP scenario"
	@CASMSOCIAL_DATA_PATH=examples/mvp \
	CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
	uv run mpirun -n 1 python -m casmsocial config/mvp.yaml
	@echo "🚀 Validating MVP output logs"
	@uv run python scripts/validate_mvp_output.py
	@echo "🚀 Summarizing MVP output"
	@uv run python scripts/summarize_mvp_output.py

.PHONY: mvp-2rank
mvp-2rank: ## Run the MVP smoke scenario with two MPI ranks
	@echo "🚀 Removing generated MVP two-rank artifacts"
	@uv run python scripts/clean_mvp_artifacts.py \
	output/mvp_2rank_summary.md \
	output/mvp_2rank_agent_log.parquet \
	output/mvp_2rank_behavior_log.parquet \
	examples/mvp/mvp.ducklake
	@echo "🚀 Creating MVP DuckLake"
	@uv run python scripts/create_mvp_ducklake.py
	@echo "🚀 Running MVP scenario with two MPI ranks"
	@CASMSOCIAL_DATA_PATH=examples/mvp \
	CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
	uv run mpirun --oversubscribe -n 2 python -m casmsocial config/mvp.yaml \
	'{"partition.table":"partitions.mvp_two_rank_place_partitions","partition.require_full_coverage":true,"observers.agent_log_file":"mvp_2rank_agent_log.parquet","observers.behavior_log_file":"mvp_2rank_behavior_log.parquet"}'
	@echo "🚀 Validating MVP two-rank output logs"
	@uv run python scripts/validate_mvp_output.py \
	--agent-log output/mvp_2rank_agent_log.parquet \
	--behavior-log output/mvp_2rank_behavior_log.parquet \
	--expected-ranks 2
	@echo "🚀 Summarizing MVP two-rank output"
	@uv run python scripts/summarize_mvp_output.py \
	--behavior-log output/mvp_2rank_behavior_log.parquet \
	--output output/mvp_2rank_summary.md \
	--expected-ranks 2

.PHONY: mvp-routed
mvp-routed: ## Run the MVP smoke scenario with road-network routing enabled
	@echo "🚀 Removing generated MVP routed artifacts"
	@uv run python scripts/clean_mvp_artifacts.py \
	output/mvp_routed_summary.md \
	output/mvp_routed_agent_log.parquet \
	output/mvp_routed_behavior_log.parquet \
	output/mvp_routed_plan_validation.json \
	examples/mvp/mvp.ducklake
	@echo "🚀 Creating MVP DuckLake"
	@uv run python scripts/create_mvp_ducklake.py
	@echo "🚀 Running MVP scenario with road-network routing"
	@CASMSOCIAL_DATA_PATH=examples/mvp \
	CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
	uv run mpirun -n 1 python -m casmsocial config/mvp.yaml \
	'{"roads.enabled":true,"roads.nodes.file":"rti_synth_pop_v2_dmv_100.road_nodes","roads.edges.file":"rti_synth_pop_v2_dmv_100.road_edges","roads.place_snap.file":"rti_synth_pop_v2_dmv_100.place_road_snap","observers.agent_log_file":"mvp_routed_agent_log.parquet","observers.behavior_log_file":"mvp_routed_behavior_log.parquet"}'
	@echo "🚀 Validating MVP routed output logs"
	@uv run python scripts/validate_mvp_output.py \
	--agent-log output/mvp_routed_agent_log.parquet \
	--behavior-log output/mvp_routed_behavior_log.parquet
	@echo "🚀 Summarizing MVP routed output"
	@uv run python scripts/summarize_mvp_output.py \
	--behavior-log output/mvp_routed_behavior_log.parquet \
	--output output/mvp_routed_summary.md
	@echo "🚀 Validating MVP routed plan metadata"
	@uv run python scripts/validate_mvp_routed_plans.py \
	--output output/mvp_routed_plan_validation.json

.PHONY: mvp-built-roads
mvp-built-roads: ## Build MVP road artifacts from OSM XML and run the routed smoke scenario against them
	@echo "🚀 Removing generated MVP built-road artifacts"
	@uv run python scripts/clean_mvp_artifacts.py \
	output/mvp_built_road_nodes.parquet \
	output/mvp_built_road_edges.parquet \
	output/mvp_built_place_road_snap.parquet \
	output/mvp_built_road_artifacts.json \
	output/mvp_built_roads_summary.md \
	output/mvp_built_roads_agent_log.parquet \
	output/mvp_built_roads_behavior_log.parquet \
	output/mvp_built_roads_plan_validation.json \
	examples/mvp/mvp.ducklake
	@echo "🚀 Building MVP road artifacts from OSM XML"
	@uv run python scripts/build_road_network.py \
	--osm-file examples/mvp/roads.osm \
	--places-file examples/mvp/road_builder_places.csv \
	--nodes-out output/mvp_built_road_nodes.parquet \
	--edges-out output/mvp_built_road_edges.parquet \
	--snaps-out output/mvp_built_place_road_snap.parquet \
	--report-out output/mvp_built_road_artifacts.json
	@echo "🚀 Creating MVP DuckLake"
	@uv run python scripts/create_mvp_ducklake.py
	@echo "🚀 Running MVP scenario with generated road artifacts"
	@CASMSOCIAL_DATA_PATH=examples/mvp \
	CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
	uv run mpirun -n 1 python -m casmsocial config/mvp.yaml \
	'{"roads.enabled":true,"roads.nodes.file":"../../output/mvp_built_road_nodes.parquet","roads.edges.file":"../../output/mvp_built_road_edges.parquet","roads.place_snap.file":"../../output/mvp_built_place_road_snap.parquet","observers.agent_log_file":"mvp_built_roads_agent_log.parquet","observers.behavior_log_file":"mvp_built_roads_behavior_log.parquet"}'
	@echo "🚀 Validating MVP built-road output logs"
	@uv run python scripts/validate_mvp_output.py \
	--agent-log output/mvp_built_roads_agent_log.parquet \
	--behavior-log output/mvp_built_roads_behavior_log.parquet
	@echo "🚀 Summarizing MVP built-road output"
	@uv run python scripts/summarize_mvp_output.py \
	--behavior-log output/mvp_built_roads_behavior_log.parquet \
	--output output/mvp_built_roads_summary.md
	@echo "🚀 Validating MVP built-road plan metadata"
	@uv run python scripts/validate_mvp_routed_plans.py \
	--roads-nodes-file ../../output/mvp_built_road_nodes.parquet \
	--roads-edges-file ../../output/mvp_built_road_edges.parquet \
	--roads-place-snap-file ../../output/mvp_built_place_road_snap.parquet \
	--skip-distance-check \
	--output output/mvp_built_roads_plan_validation.json

.PHONY: mvp-delta-state
mvp-delta-state: ## Run the MVP smoke scenario with delta agent-state logging enabled
	@echo "🚀 Removing generated MVP delta-state artifacts"
	@uv run python scripts/clean_mvp_artifacts.py \
	output/mvp_delta_state_summary.md \
	output/mvp_delta_state_agent_log.parquet \
	output/mvp_delta_state_behavior_log.parquet \
	output/mvp_agent_state_delta.parquet \
	output/mvp_agent_state_delta_audit.parquet \
	output/mvp_agent_state_reconstructed.parquet \
	output/mvp_delta_state_validation.json \
	output/mvp_agent_state_delta_ducklake_report.md \
	examples/mvp/mvp.ducklake
	@echo "🚀 Creating MVP DuckLake"
	@uv run python scripts/create_mvp_ducklake.py
	@echo "🚀 Running MVP scenario with delta agent-state logging"
	@CASMSOCIAL_DATA_PATH=examples/mvp \
	CASMSOCIAL_DUCKLAKE_PATH=examples/mvp/mvp.ducklake \
	uv run mpirun -n 1 python -m casmsocial config/mvp.yaml \
	'{"observers.agent_log_file":"mvp_delta_state_agent_log.parquet","observers.behavior_log_file":"mvp_delta_state_behavior_log.parquet","observers.delta_agent_state.enabled":true,"observers.delta_agent_state_file":"mvp_agent_state_delta.parquet","observers.delta_agent_state_audit_file":"mvp_agent_state_delta_audit.parquet"}'
	@echo "🚀 Summarizing MVP delta-state output"
	@uv run python scripts/summarize_mvp_output.py \
	--behavior-log output/mvp_delta_state_behavior_log.parquet \
	--output output/mvp_delta_state_summary.md
	@echo "🚀 Validating MVP delta-state reconstruction"
	@uv run python scripts/validate_agent_state_delta.py \
	--agent-log output/mvp_delta_state_agent_log.parquet \
	--behavior-log output/mvp_delta_state_behavior_log.parquet \
	--delta-log output/mvp_agent_state_delta.parquet \
	--audit-log output/mvp_agent_state_delta_audit.parquet \
	--reconstructed-output output/mvp_agent_state_reconstructed.parquet \
	--report-output output/mvp_delta_state_validation.json \
	--overwrite
	@echo "🚀 Loading MVP delta-state outputs into DuckLake"
	@uv run python scripts/load_agent_state_delta_ducklake.py \
	--ducklake-path examples/mvp/mvp.ducklake \
	--delta-log output/mvp_agent_state_delta.parquet \
	--audit-log output/mvp_agent_state_delta_audit.parquet \
	--reconstructed-log output/mvp_agent_state_reconstructed.parquet \
	--validation-report output/mvp_delta_state_validation.json
	@echo "🚀 Reporting MVP delta-state DuckLake queries"
	@uv run python scripts/report_agent_state_delta_ducklake.py \
	--ducklake-path examples/mvp/mvp.ducklake \
	--output output/mvp_agent_state_delta_ducklake_report.md

.PHONY: mvp-delta-state-report
mvp-delta-state-report: ## Write query examples over loaded MVP delta-state DuckLake tables
	@uv run python scripts/report_agent_state_delta_ducklake.py \
	--ducklake-path examples/mvp/mvp.ducklake \
	--output output/mvp_agent_state_delta_ducklake_report.md

.PHONY: mvp-manifest
mvp-manifest: ## Write a manifest for the generated MVP artifacts
	@echo "🚀 Writing MVP artifact manifest"
	@uv run python scripts/write_mvp_manifest.py

.PHONY: mvp-verify-manifest
mvp-verify-manifest: ## Verify generated MVP artifacts against the manifest
	@echo "🚀 Verifying MVP artifact manifest"
	@uv run python scripts/verify_mvp_manifest.py

.PHONY: mvp-artifacts
mvp-artifacts: ## List the generated MVP artifact paths uploaded in CI
	@uv run python scripts/list_mvp_artifacts.py

.PHONY: mvp-all
mvp-all: ## Run all MVP smoke scenarios, write the manifest, and verify it
	@$(MAKE) --no-print-directory mvp
	@$(MAKE) --no-print-directory mvp-2rank
	@$(MAKE) --no-print-directory mvp-routed
	@$(MAKE) --no-print-directory mvp-built-roads
	@$(MAKE) --no-print-directory mvp-delta-state
	@$(MAKE) --no-print-directory mvp-manifest
	@$(MAKE) --no-print-directory mvp-verify-manifest

.PHONY: mvp-clean
mvp-clean: ## Remove generated MVP artifacts
	@echo "🚀 Removing generated MVP artifacts"
	@uv run python scripts/clean_mvp_artifacts.py

.PHONY: mvp-check
mvp-check: ## Validate the MVP log artifacts
	@echo "🚀 Validating MVP output logs"
	@uv run python scripts/validate_mvp_output.py

.PHONY: mvp-report
mvp-report: ## Summarize the MVP behavior log artifact
	@echo "🚀 Summarizing MVP output"
	@uv run python scripts/summarize_mvp_output.py

.PHONY: docker-mpi-build
docker-mpi-build: ## Build the Docker Compose MPI image
	@echo "🚀 Building Docker Compose MPI image"
	@$(DOCKER_MPI_COMPOSE) build --build-arg UV_INSECURE_HOST=$(DOCKER_MPI_UV_INSECURE_HOST)

.PHONY: docker-mpi-up
docker-mpi-up: ## Start Docker Compose MPI rank containers
	@echo "🚀 Starting Docker Compose MPI rank containers"
	@$(DOCKER_MPI_COMPOSE) up -d --no-build

.PHONY: docker-mpi-smoke
docker-mpi-smoke: docker-mpi-up ## Run a hostfile-based MPI smoke check across Docker Compose rank containers
	@echo "🚀 Running Docker Compose MPI smoke check"
	@$(DOCKER_MPI_COMPOSE) exec -T rank0 mpirun -hostfile $(DOCKER_MPI_HOSTFILE) -n $(DOCKER_MPI_RANKS) hostname

.PHONY: docker-mpi-down
docker-mpi-down: ## Stop Docker Compose MPI rank containers
	@echo "🚀 Stopping Docker Compose MPI rank containers"
	@$(DOCKER_MPI_COMPOSE) down

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
