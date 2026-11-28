"""Supervised one-run mode: atomic endpoints, durable state and exit codes."""

import json
import subprocess
import sys
import time
from unittest.mock import MagicMock

import grpc
import pytest
from casmsim.grpc_runner import ENDPOINT_FILENAME, SimulatorControlServicer
from casmsim.proto import casm_runner_pb2 as pb, casm_runner_pb2_grpc as pb_grpc

from casmsocial import grpc_runner


@pytest.mark.parametrize(
    "outcome, code",
    [("complete", 0), ("fail", 4), ("cancel", 143)],
)
def test_supervised_cli_serves_one_run_records_state_and_exits(monkeypatch, tmp_path, outcome, code):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / ENDPOINT_FILENAME).write_text('{"control": {"address": "127.0.0.1:1"}}')
    (run_dir / "arrow_endpoint.txt").write_text("127.0.0.1:2\n")
    (run_dir / ".staging-1").mkdir()
    artifacts = tmp_path / "artifacts" / "run-1"
    monkeypatch.setattr(grpc_runner, "get_comm", lambda: None)
    monkeypatch.setattr(sys, "argv", ["casmsocial-runner", "--run-dir", str(run_dir), "--supervised",
                                      "--drain-seconds", "0"])
    control, flights, adapter = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(grpc_runner, "resolve_adapter", MagicMock(return_value=adapter))
    runtime = {}

    def start_control(path, broker, start_run, *, cancel_run):
        assert path.parent == run_dir and path.name.startswith(".staging-")
        assert not (run_dir / ENDPOINT_FILENAME).exists(), "stale endpoints are cleared first"
        assert not (run_dir / ".staging-1").exists()
        path.mkdir(mode=0o700)
        (path / ENDPOINT_FILENAME).write_text(json.dumps({"control": {"address": "127.0.0.1:1234"}}))
        servicer = SimulatorControlServicer(broker, start_run, cancel_run)
        runtime["servicer"] = servicer
        config = {"model.name": "test", "runner.artifact_root": str(artifacts)}
        servicer.Start(pb.StartRequest(run_id="run-1", config_json=json.dumps(config).encode()), MagicMock())
        return control

    def start_flight(path, broker):
        (path / "arrow_endpoint.txt").write_text("127.0.0.1:4321\n")
        return flights

    def model_start():
        published = json.loads((run_dir / ENDPOINT_FILENAME).read_text())
        assert published["capabilities"]["durable_observations"] is True
        assert published["control"]["address"] == "127.0.0.1:1234"
        assert (run_dir / "arrow_endpoint.txt").read_text() == "127.0.0.1:4321\n"
        if outcome == "fail":
            raise ValueError("model failed")
        if outcome == "cancel":
            assert runtime["servicer"].Cancel(pb.CancelRequest(run_id="run-1"), MagicMock()).acknowledged

    adapter.start.side_effect = model_start
    monkeypatch.setattr(grpc_runner, "start_control_server", start_control)
    monkeypatch.setattr(grpc_runner, "start_broker_flight_server", start_flight)
    assert grpc_runner.main() == code
    control.wait_for_termination.assert_not_called()
    control.stop.assert_called_once_with(0)
    flights.shutdown.assert_called_once_with()
    state = {"complete": "Completed", "fail": "Failed", "cancel": "Cancelled"}[outcome]
    assert json.loads((artifacts / "manifest.json").read_text()) == {
        "schema_version": 1, "run_id": "run-1", "state": state}
    assert sorted(p.name for p in run_dir.iterdir()) == [], "endpoints withdrawn and staging removed"


def test_withdraw_keeps_a_newer_jobs_endpoints(tmp_path):
    (tmp_path / ENDPOINT_FILENAME).write_text('{"control": {"address": "127.0.0.1:9"}}')
    grpc_runner.withdraw_endpoints(tmp_path, {"control": {"address": "127.0.0.1:1"}})
    assert (tmp_path / ENDPOINT_FILENAME).exists()


def test_durable_state_needs_an_absolute_artifact_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = grpc_runner._RunRequest("r", b'{"runner.artifact_root": "relative"}')
    grpc_runner.record_durable_state(request, "Completed")
    grpc_runner.record_durable_state(grpc_runner._RunRequest("r", b"not json"), "Failed")
    grpc_runner.record_durable_state(None, "Failed")
    assert list(tmp_path.iterdir()) == []


def test_real_supervised_runner_fails_bad_run_and_exits_4(tmp_path):
    run_dir = tmp_path / "runner"
    artifacts = tmp_path / "artifacts" / "a1"
    process = subprocess.Popen(
        [sys.executable, "-m", "casmsocial.grpc_runner", "--run-dir", str(run_dir), "--supervised",
         "--drain-seconds", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 60
        manifest = None
        while time.monotonic() < deadline:
            try:
                manifest = json.loads((run_dir / ENDPOINT_FILENAME).read_text())
                break
            except (OSError, ValueError):
                time.sleep(0.1)
        assert manifest and manifest["capabilities"]["durable_observations"] is True
        stub = pb_grpc.SimulatorControlStub(grpc.insecure_channel(manifest["control"]["address"]))
        config = {"runner.entry_point": "casmsocial.no_such_module:Adapter", "runner.artifact_root": str(artifacts)}
        stub.Start(pb.StartRequest(run_id="r1", config_json=json.dumps(config).encode()))
        state = None
        while time.monotonic() < deadline:
            state = stub.GetState(pb.GetStateRequest(run_id="r1")).state
            if state != pb.RUN_STATE_RUNNING:
                break
            time.sleep(0.1)
        assert state == pb.RUN_STATE_FAILED, "the final state is served during the drain window"
        assert process.wait(timeout=30) == 4, process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
    assert json.loads((artifacts / "manifest.json").read_text())["state"] == "Failed"
    assert sorted(p.name for p in run_dir.iterdir()) == []
