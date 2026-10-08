"""Bounded subprocess tests for real MPI failure recovery and job abortion."""

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

MPI_LAUNCHER = shutil.which("mpiexec")
pytestmark = [
    pytest.mark.skipif(MPI_LAUNCHER is None, reason="MPI launcher unavailable"),
    pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 1, reason="cannot nest MPI subprocess launches"),
]


def run_case(phase, rank=0):
    harness = Path(__file__).parent / "mpi" / "runner_failure_case.py"
    return subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness), phase, str(rank)],
        cwd=harness.parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("phase", ["invalid_config", "startup_interrupt", "server_startup"])
def test_pre_dispatch_failure_releases_both_ranks(phase):
    result = run_case(phase)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["ranks"][0]["startup_failed"]
    assert output["ranks"][1]["worker_released"]


@pytest.mark.parametrize("phase", ["constructor", "start", "flush", "completion"])
@pytest.mark.parametrize("rank", [0, 1])
def test_rank_local_failure_terminates_whole_job_without_deadlock(phase, rank):
    result = run_case(phase, rank)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"injected {phase} failure on rank {rank}" in result.stderr, result.stdout + result.stderr
    assert "MPI runner failed; aborting all ranks" in result.stderr
