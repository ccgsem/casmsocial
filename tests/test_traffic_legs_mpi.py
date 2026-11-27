"""Routed legs on the vehicle mobsim give the same day on 1 and 2 ranks."""

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

JAM_CAPACITY_VPH = 0.05


def run_case(n_ranks: int) -> dict:
    harness = Path(__file__).parent / "mpi" / "traffic_legs_case.py"
    result = subprocess.run(
        [MPI_LAUNCHER, "--oversubscribe", "-n", str(n_ranks), sys.executable, str(harness), str(JAM_CAPACITY_VPH)],
        cwd=harness.parents[2],
        env=launcher_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {n: run_case(n) for n in (1, 2)}


def _without_rank(rows, rank_index):
    return [tuple(row[:rank_index]) + tuple(row[rank_index + 1 :]) for row in rows]


def test_two_ranks_split_the_road_network(runs):
    assert set(runs[2]["node_rank"].values()) == {0, 1}
    assert runs[2]["node_rank"]["2"] == 1  # place 200's node
    assert runs[2]["node_rank"]["3"] == 0  # place 300's node


def test_every_leg_arrives_once_on_both_rank_counts(runs):
    for run in runs.values():
        summary = run["summary"]
        assert summary["spawned"] == summary["arrived"] == summary["trips_completed"] == 4
        assert summary["held_now"] == 0
        assert summary["in_network"] == 0


def test_person_two_vehicles_cross_ranks_and_report_home(runs):
    trips = runs[2]["trips"]  # (person, trip, depart, arrive, scheduled, migrations, rank)
    person_two = [trip for trip in trips if trip[0] == 2]
    assert len(person_two) == 2
    assert all(trip[5] >= 1 for trip in person_two)
    # Morning: leaves place 200 (rank 1); the record lands where person 2 lives then.
    morning, evening = sorted(person_two, key=lambda trip: trip[2])
    assert morning[6] == 1
    assert evening[6] == 0
    assert runs[2]["summary"]["migrations"] >= 2


def test_trips_and_timeline_match_one_rank(runs):
    one, two = runs[1], runs[2]
    trips_one = [trip[:5] for trip in one["trips"]]
    trips_two = [trip[:5] for trip in two["trips"]]
    assert len(trips_one) == len(trips_two)
    for a, b in zip(trips_one, trips_two, strict=True):
        assert a[:3] == b[:3]
        # A migrating vehicle may enter its next edge up to one sync window late.
        assert b[3] == pytest.approx(a[3], abs=10.0)
    assert _without_rank(one["timeline"], 3) == _without_rank(two["timeline"], 3)


def test_jammed_leg_is_late_on_both_rank_counts(runs):
    for run in runs.values():
        late = [trip for trip in run["trips"] if trip[0] == 1 and trip[3] > trip[4]]
        assert len(late) == 1
        assert run["summary"]["trips_delayed"] == 1
