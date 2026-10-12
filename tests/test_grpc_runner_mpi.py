"""CLI smoke tests with real MPI ranks, control RPCs, and Flight retrieval."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import grpc
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from casmsim.proto import casm_runner_pb2 as pb, casm_runner_pb2_grpc as rpc

try:
    from mpi4py import MPI
except (ImportError, RuntimeError) as error:
    pytest.skip(f"MPI Python/runtime unavailable: {error}", allow_module_level=True)

MPI_LAUNCHER = shutil.which("mpiexec")
pytestmark = [
    pytest.mark.skipif(MPI_LAUNCHER is None, reason="MPI launcher unavailable"),
    pytest.mark.skipif(MPI.COMM_WORLD.Get_size() != 1, reason="cannot nest MPI subprocess launches"),
    pytest.mark.skipif(os.name != "posix", reason="fixture cleanup requires POSIX process groups"),
]


def wait_until(check, process, description, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            pytest.fail(f"runner exited while waiting for {description}:\n{stdout}\n{stderr}")
        Event().wait(0.01)
    pytest.fail(f"timed out waiting for {description}")


def stop_fixture(process, tmp_path):
    (tmp_path / "stop").write_text("stop")
    if process.poll() is None:
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)


@pytest.mark.parametrize("ranks,mode", [(1, "complete"), (2, "complete"), (2, "cancel"), (2, "worker_flush")])
def test_cli_collective_run_owns_endpoints_only_on_rank0(tmp_path, ranks, mode):
    harness = Path(__file__).parent / "mpi" / "runner_launcher_case.py"
    fixture_env = dict(os.environ, MPI4PY_RC_THREAD_LEVEL="funneled")
    process = subprocess.Popen(
        [MPI_LAUNCHER, "-n", str(ranks), sys.executable, str(harness), str(tmp_path)],
        cwd=harness.parents[2],
        env=fixture_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        wait_until((tmp_path / "transport_ready").exists, process, "transport startup")
        run_dir = tmp_path / "run"
        manifest = json.loads((run_dir / "runner_endpoints.json").read_text())
        params = {
            "runner.entry_point": "__main__:CollectiveAdapter",
            "simulation.run_id": "untrusted",
            "observers.arrow_server.enabled": True,
            "mode": mode,
        }
        with grpc.insecure_channel(manifest["control"]["address"]) as channel:
            client = rpc.SimulatorControlStub(channel)
            client.Start(pb.StartRequest(run_id="run-1", config_json=json.dumps(params).encode()), timeout=5)
            if mode == "cancel":
                wait_until((tmp_path / "model_running").exists, process, "active model")
                request = pb.CancelRequest(run_id="run-1")
                assert client.Cancel(request, timeout=5).acknowledged
                assert client.Cancel(request, timeout=5).acknowledged
                assert client.GetState(pb.GetStateRequest(run_id="run-1"), timeout=5).state == pb.RUN_STATE_RUNNING
                (tmp_path / "release_cancel").write_text("release")
            elif mode == "worker_flush":
                wait_until((tmp_path / "worker_flush_waiting").exists, process, "worker flush")
                wait_until((tmp_path / "rank0_completion_barrier").exists, process, "rank 0 completion barrier")
                assert client.GetState(pb.GetStateRequest(run_id="run-1"), timeout=5).state == pb.RUN_STATE_RUNNING
                (tmp_path / "release_worker_flush").write_text("release")

            expected_state = pb.RUN_STATE_CANCELLED if mode == "cancel" else pb.RUN_STATE_COMPLETED
            wait_until(
                lambda: client.GetState(pb.GetStateRequest(run_id="run-1"), timeout=5).state == expected_state,
                process,
                "terminal run state",
            )
            assert not client.Cancel(pb.CancelRequest(run_id="run-1"), timeout=5).acknowledged
            batches = list(client.StreamObs(pb.StreamObsRequest(run_id="run-1", channel="ranks"), timeout=5))
            assert [batch.tick for batch in batches] == [0]
            grpc_table = pa.ipc.open_stream(pa.py_buffer(batches[0].arrow_ipc)).read_all()
        with flight.FlightClient(f"grpc://{manifest['flight']['address']}") as client:
            flight_table = client.do_get(flight.Ticket(b"ranks")).read_all()
        assert grpc_table.equals(flight_table)
        assert flight_table.to_pylist() == [{"rank": rank, "run_id": "run-1"} for rank in range(ranks)]
        assert run_dir.stat().st_mode & 0o777 == 0o700
        for filename in ("runner_endpoints.json", "arrow_endpoint.txt"):
            assert (run_dir / filename).stat().st_mode & 0o777 == 0o600

        (tmp_path / "stop").write_text("stop")
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stdout + stderr
        records = json.loads((tmp_path / "result.json").read_text())
        assert [item["rank"] for item in records] == list(range(ranks))
        for item in records:
            assert item["constructed"] and item["started"] and item["adapter_returned"]
            assert item["exit_code"] == 0
            assert item["params"]["simulation.run_id"] == "run-1"
            assert item["params"]["observers.arrow_server.enabled"] is False
            assert item["thread_level"] >= MPI.THREAD_FUNNELED
            assert item["completion_barriers"] == (1 if ranks > 1 else 0)
            assert item["observer_attached"] == (item["rank"] == 0)
            assert item["collective_cancelled"] == (mode == "cancel")
            assert item["cancel_calls"] == (1 if mode == "cancel" and item["rank"] == 0 else 0)
            assert item["control_starts"] == (1 if item["rank"] == 0 else 0)
            assert item["flight_starts"] == (1 if item["rank"] == 0 else 0)
            assert item["endpoint_writes"] > 0 if item["rank"] == 0 else item["endpoint_writes"] == 0
    finally:
        stop_fixture(process, tmp_path)
