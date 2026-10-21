"""Two-rank MPI test: a Person's behavioral state survives rank migration."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from mpi4py import MPI
except (ImportError, RuntimeError) as error:
    pytest.skip(f"MPI Python/runtime unavailable: {error}", allow_module_level=True)

from tests.test_runner_mpi_failures import launcher_env

MPI_LAUNCHER = shutil.which("mpiexec")
pytestmark = [
    pytest.mark.skipif(MPI_LAUNCHER is None, reason="MPI launcher unavailable"),
    pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 1, reason="cannot nest MPI subprocess launches"),
]


@pytest.fixture(scope="module")
def migration():
    harness = Path(__file__).parent / "mpi" / "person_migration_case.py"
    result = subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness)],
        cwd=harness.parents[2],
        env=launcher_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", ["v2", "llm"])
def test_cognitive_state_survives_migration(migration, name):
    sent = migration["sent"][name]
    received = migration["received"][name]

    assert received["beliefs"]["routine_preference"] == 0.123
    assert received["beliefs"]["persona_prior"] == 0.456
    for key in (
        "engine",
        "beliefs",
        "goals",
        "memory",
        "episodic_memory",
        "stress",
        "last_llm_tick",
        "last_llm_summary",
    ):
        assert received[key] == sent[key], key


@pytest.mark.parametrize("name", ["v2", "llm"])
def test_communication_buffers_survive_migration(migration, name):
    sent = migration["sent"][name]
    received = migration["received"][name]

    assert received["inbox"] == ["inbox-1"]
    for key in ("inbox", "recent_messages", "outbound_message_intents", "pending_acks"):
        assert received[key] == sent[key], key


def test_schedule_agent_migrates_without_cognition(migration):
    received = migration["received"]["schedule"]

    assert received["engine"] == "ScheduleBehaviorEngine"
    assert received["inbox"] == []
    assert "beliefs" not in received


def test_synchronize_path_never_serializes_persons():
    """CasmPop's context.synchronize() branch moves and ghosts no Persons, so save() never runs.

    PlacesProjectionV2 is neither a repast4py BoundedProjection (no out-of-bounds
    moves) nor a ghosting projection, so the larger index-5 payload stays a
    per-migration cost and never becomes a per-tick ghost-update cost.
    """
    harness = Path(__file__).parent / "mpi" / "person_migration_case.py"
    result = subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness), "synchronize"],
        cwd=harness.parents[2],
        env=launcher_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = json.loads(result.stdout.strip().splitlines()[-1])

    assert ranks == [{"saves": 0, "bounded_projections": 0, "ghosted": 0, "ghosts": 0, "local": 3}] * 2
