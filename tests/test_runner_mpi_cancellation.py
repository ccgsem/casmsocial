"""Exercise cancellation gates in two real MPI processes with a time limit."""

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


@pytest.mark.skipif(MPI_LAUNCHER is None, reason="MPI launcher unavailable")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 1, reason="cannot nest MPI subprocess launches")
@pytest.mark.parametrize("phase", ["queued", "resolver", "constructor", "observer_registration", "running", "flush"])
def test_two_rank_cancellation_finishes_without_skipping_collectives(phase):
    harness = Path(__file__).parent / "mpi" / "runner_cancellation_case.py"
    result = subprocess.run(
        [MPI_LAUNCHER, "-n", "2", sys.executable, str(harness), phase],
        cwd=harness.parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["phase"] == phase
    assert [rank["rank"] for rank in output["ranks"]] == [0, 1]
