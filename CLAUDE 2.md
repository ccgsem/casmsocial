# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

casmsocial is a Python framework for implementing agent-based models (ABMs) that simulate synthetic population dynamics. The framework uses MPI for distributed computing and repast4py as the core ABM engine.

## Development Commands

**Environment Setup:**
```bash
make install              # Create virtual environment with uv and install pre-commit hooks
uv sync                   # Install dependencies
```

**Code Quality:**
```bash
make check               # Run all code quality checks (lock file, pre-commit, mypy)
uv run pre-commit run -a # Run linting and formatting (ruff)
uv run mypy              # Type checking
```

**Testing:**
```bash
make test                # Run pytest with doctest modules
uv run python -m pytest # Direct pytest execution
```

**Building:**
```bash
make build               # Build wheel file
```

**Running Models:**
```bash
uv run mpirun -n 1 python -m casmsocial config/casmsocial.yaml
# or
uv run python -m casmsocial config/casmsocial.yaml
```

**Documentation:**
```bash
make docs                # Build and serve documentation with mkdocs
make docs-test           # Test documentation build
```

## Architecture Overview

### Core Components

**Model Factory Pattern:**
- `Models` class in `factory.py` provides a registry for model types
- Models are registered automatically when imported in `__init__.py`
- Runtime model creation via `Models.create_model(model_name)`

**Agent Hierarchy:**
- `Person(core.Agent)` - Individual agents with activity schedules and social interactions
- `Place(core.Agent)` - Locations where activities occur (Household, School, Workplace subclasses)
- Both use dataclass-based configuration with class-level factories

**Model Interface:**
- Abstract `Model` class defines `start()` and `step()` methods
- Concrete models in subdirectories: `dcsim/`, `heat_risk/`
- Models use MPI communicators for distributed simulation

**Data Architecture:**
- Configuration-driven agent creation using dataclasses
- Separate data classes (PersonData, PlaceData) for serialization
- Factory methods for creating agents from configuration dictionaries

### Key Patterns

**Class Configuration:**
- Agent classes use class variables for data class configuration
- `setPersonDataClass()` and `setPlaceDataClass()` methods for runtime configuration
- Factory pattern for creating agents: `Person.restore(person_data)`

**Activity Scheduling:**
- `activities.py` manages agent schedules and place assignments
- Time-based movement between locations
- Integration with repast4py's scheduling system

**Error Handling:**
- Custom exception hierarchy for domain-specific errors
- Validation patterns with descriptive error messages
- Loguru for structured logging throughout

**MPI Distribution:**
- Distributed agents across MPI ranks
- Synchronization and communication patterns for parallel execution
- Place projections for spatial relationships

### Configuration

Models are configured via YAML files in `config/` directory. The framework uses:
- `repast4py.parameters` for parameter management
- Environment variables via python-dotenv
- Runtime model selection via `model.name` parameter

### Testing

- pytest framework with doctest integration
- Development dependencies managed in pyproject.toml dependency groups
- Pre-commit hooks for code quality enforcement