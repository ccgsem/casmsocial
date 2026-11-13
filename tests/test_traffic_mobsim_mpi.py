"""Multi-rank tests: the vehicle mobsim gives the same traffic on 1, 2 and 3 ranks."""

from __future__ import annotations

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

GRID_WIDTH = 12
N_TRIPS = 4000
SYNC_WINDOW_S = 1.0
STRATEGIC_SHARE = 0.1


def run_case(n_ranks: int) -> dict:
    harness = Path(__file__).parent / "mpi" / "traffic_mobsim_case.py"
    result = subprocess.run(
        [
            MPI_LAUNCHER,
            "--oversubscribe",
            "-n",
            str(n_ranks),
            sys.executable,
            str(harness),
            str(GRID_WIDTH),
            str(N_TRIPS),
            str(SYNC_WINDOW_S),
            str(STRATEGIC_SHARE),
        ],
        cwd=harness.parents[2],
        env=launcher_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {n: run_case(n) for n in (1, 2, 3)}


@pytest.mark.parametrize("n_ranks", [1, 2, 3])
def test_every_trip_arrives_and_is_recorded_once(runs, n_ranks):
    run = runs[n_ranks]
    stats = run["stats"]

    assert stats["spawned"] == stats["arrived"] == N_TRIPS
    assert stats["in_network"] == 0
    assert [person for person, _ in run["arrivals"]] == list(range(N_TRIPS))
    # Each edge is owned by one rank, so gathered profile rows never collide.
    assert run["profile_edges_unique"]
    assert run["profile_traversals"] == stats["edge_entries"]


@pytest.mark.parametrize("n_ranks", [2, 3])
def test_vehicles_cross_rank_boundaries(runs, n_ranks):
    run = runs[n_ranks]

    assert run["boundary_edge_share"] > 0
    assert run["stats"]["migrations"] > N_TRIPS / 2  # includes forwarding from spawn rank 0
    assert run["arrival_ranks"] == list(range(n_ranks))
    assert run["max_vehicle_migrations"] >= n_ranks - 1


@pytest.mark.parametrize("n_ranks", [2, 3])
def test_traffic_matches_single_rank(runs, n_ranks):
    """Rank count changes only event ordering within a sync window and which
    loads a reroute can see, so aggregate results stay within a fraction of a
    percent and nearly all trips match closely."""
    baseline = dict(runs[1]["arrivals"])
    other = dict(runs[n_ranks]["arrivals"])

    mean_base = sum(baseline.values()) / N_TRIPS
    mean_other = sum(other.values()) / N_TRIPS
    assert mean_other == pytest.approx(mean_base, rel=0.005)

    close = sum(1 for person, t in baseline.items() if abs(other[person] - t) <= max(5.0, 0.02 * t))
    assert close / N_TRIPS > 0.95
