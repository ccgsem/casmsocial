"""Bounded subprocess tests for real MPI failure recovery and job abortion."""

import json
import os
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


def launcher_env():
    """Environment for a nested ``mpiexec``, built from ``os.environ``.

    Importing mpi4py initializes this pytest process as an MPI singleton, and
    Open MPI records that in C-level environment variables that ``os.environ``
    doesn't show. A child launched with the inherited environment sees them,
    takes itself for part of an existing job and exits with status 1 and no
    output. Passing an explicit copy of ``os.environ`` (as
    test_grpc_runner_mpi.py does) leaves them out; any OMPI_/PMIX_ variables
    that are visible are dropped too, except the run-as-root opt-ins.
    """
    keep = {"OMPI_ALLOW_RUN_AS_ROOT", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"}
    return {
        key: value
        for key, value in os.environ.items()
        if key in keep or not key.startswith(("OMPI_", "PMIX_", "PRTE_"))
    }


def run_case(phase, rank=0):
    harness = Path(__file__).parent / "mpi" / "runner_failure_case.py"
    return subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness), phase, str(rank)],
        cwd=harness.parents[2],
        env=launcher_env(),
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
