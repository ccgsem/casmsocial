"""Two-rank MPI test: messages reach people who live away from the rank they were created on."""

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


def test_messages_reach_people_away_from_their_creation_rank():
    harness = Path(__file__).parent / "mpi" / "message_routing_case.py"
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
    ranks = json.loads(result.stdout.strip().splitlines()[-1])

    for record in ranks:
        assert record["inbox"] == ["across", "local"], record
        assert record["acks"] == ["received"], record
