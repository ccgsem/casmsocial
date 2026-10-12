"""Exercise cancellation gates in two real MPI processes with a time limit."""

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


@pytest.mark.skipif(MPI_LAUNCHER is None, reason="MPI launcher unavailable")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 1, reason="cannot nest MPI subprocess launches")
@pytest.mark.parametrize("phase", ["queued", "resolver", "constructor", "observer_registration", "running", "flush"])
def test_two_rank_cancellation_finishes_without_skipping_collectives(phase):
    harness = Path(__file__).parent / "mpi" / "runner_cancellation_case.py"
    result = subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness), phase],
        cwd=harness.parents[2],
        env=launcher_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["phase"] == phase
    assert [rank["rank"] for rank in output["ranks"]] == [0, 1]
