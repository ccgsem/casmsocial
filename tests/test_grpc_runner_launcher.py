"""Rank ownership and control-to-main-thread wiring in the CLI launcher."""

import json
import sys
from threading import Event, current_thread, main_thread
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from casmsim.grpc_runner import ENDPOINT_FILENAME, SimulatorControlServicer
from casmsim.proto import casm_runner_pb2 as pb

from casmsocial import grpc_runner


def configure_launcher(monkeypatch, run_dir, rank=0, size=1, with_comm=True):
    comm = MagicMock() if with_comm else None
    if comm is not None:
        comm.Get_rank.return_value = rank
        comm.Get_size.return_value = size
    monkeypatch.setattr(grpc_runner, "get_comm", lambda: comm)
    monkeypatch.setattr(sys, "argv", ["casmsocial-runner", "--run-dir", str(run_dir)])
    return comm


@pytest.mark.parametrize("size,with_comm", [(1, False), (1, True), (2, True)])
def test_rank0_cli_runs_adapter_on_main_thread_and_keeps_output_available(monkeypatch, tmp_path, size, with_comm):
    run_dir = tmp_path / "run"
    comm = configure_launcher(monkeypatch, run_dir, size=size, with_comm=with_comm)
    control, flights, adapter = MagicMock(), MagicMock(), MagicMock()
    flights.port = 4321
    runtime = {}
    resolver = MagicMock(return_value=adapter)
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)

    def start_control(path, broker, start_run, *, cancel_run):
        assert current_thread() is main_thread()
        assert path == run_dir
        path.mkdir(mode=0o700)
        (path / ENDPOINT_FILENAME).write_text(json.dumps({"control": {"address": "127.0.0.1:1234"}}))
        servicer = SimulatorControlServicer(broker, start_run, cancel_run)
        runtime.update(servicer=servicer, broker=broker)
        servicer.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), MagicMock())
        # Start only queues a request; model construction waits for main().
        resolver.assert_not_called()
        return control

    def start_flight(path, broker):
        assert path == run_dir
        assert broker is runtime["broker"]
        (path / "arrow_endpoint.txt").write_text("127.0.0.1:4321\n")
        return flights

    def model_start():
        assert current_thread() is main_thread()
        if comm is not None:
            assert resolver.call_args.args[0] is comm
        assert (
            runtime["servicer"].GetState(pb.GetStateRequest(run_id="run-1"), MagicMock()).state == pb.RUN_STATE_RUNNING
        )
        sink = adapter.add_observer.call_args.args[0]
        sink.publish("agents", pa.table({"id": [1]}))
        sink.flush()

    def wait_for_termination():
        # Join the control callback to observe its terminal state transition.
        servicer = runtime["servicer"]
        servicer._worker.join(timeout=2)
        assert not servicer._worker.is_alive()
        assert servicer.GetState(pb.GetStateRequest(run_id="run-1"), MagicMock()).state == pb.RUN_STATE_COMPLETED
        assert runtime["broker"].read("agents").batches[0].table.to_pydict() == {"id": [1]}
        flights.shutdown.assert_not_called()

    adapter.start.side_effect = model_start
    control.wait_for_termination.side_effect = wait_for_termination
    monkeypatch.setattr(grpc_runner, "start_control_server", start_control)
    monkeypatch.setattr(grpc_runner, "start_broker_flight_server", start_flight)
    assert grpc_runner.main() == 0
    adapter.start.assert_called_once_with()
    control.wait_for_termination.assert_called_once_with()
    control.stop.assert_called_once_with(0)
    control.stop.return_value.wait.assert_called_once_with()
    flights.shutdown.assert_called_once_with()
    manifest_path = run_dir / ENDPOINT_FILENAME
    assert json.loads(manifest_path.read_text())["flight"]["address"] == "127.0.0.1:4321"
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    if comm is not None:
        if size == 1:
            comm.bcast.assert_not_called()
        else:
            comm.bcast.assert_called_once_with(
                (
                    grpc_runner._MSG_START,
                    {
                        "model.name": "test",
                        "runner.entry_point": "casmsocial.adapters.runner:CasmPopAdapter",
                        "simulation.run_id": "run-1",
                        "observers.arrow_server.enabled": False,
                    },
                ),
                root=0,
            )


def test_worker_cli_starts_adapter_without_servers_or_endpoint_files(monkeypatch, tmp_path):
    run_dir = tmp_path / "worker-must-not-create"
    comm = configure_launcher(monkeypatch, run_dir, rank=1, size=2)
    params = {"runner.entry_point": "custom:Adapter", "simulation.run_id": "run-1"}
    comm.bcast.return_value = (grpc_runner._MSG_START, params)
    control_start, flight_start = MagicMock(), MagicMock()
    monkeypatch.setattr(grpc_runner, "start_control_server", control_start)
    monkeypatch.setattr(grpc_runner, "start_broker_flight_server", flight_start)
    adapter = MagicMock()

    def resolve(actual_comm, actual_params):
        assert current_thread() is main_thread()
        assert actual_comm is comm
        assert actual_params == params
        return adapter

    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolve)
    assert grpc_runner.main() == 0
    comm.bcast.assert_called_once_with(None, root=0)
    adapter.start.assert_called_once_with()
    adapter.add_observer.assert_not_called()
    control_start.assert_not_called()
    flight_start.assert_not_called()
    assert not run_dir.exists()


def test_direct_endpoint_start_on_worker_fails_before_touching_files(monkeypatch, tmp_path):
    run_dir = tmp_path / "worker-must-not-create"
    configure_launcher(monkeypatch, run_dir, rank=1, size=2)
    start_control = MagicMock()
    monkeypatch.setattr(grpc_runner, "start_control_server", start_control)
    with pytest.raises(RuntimeError, match="only rank 0"):
        grpc_runner.start_runner(run_dir)
    start_control.assert_not_called()
    assert not run_dir.exists()


def test_multi_rank_endpoint_start_requires_handoff(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path / "run", size=2)
    with pytest.raises(RuntimeError, match="require a main-thread handoff"):
        grpc_runner.start_runner(tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_server_startup_failure_stops_control_and_releases_workers(monkeypatch, tmp_path):
    comm = configure_launcher(monkeypatch, tmp_path / "run", size=2)
    control = MagicMock()
    monkeypatch.setattr(grpc_runner, "start_control_server", MagicMock(return_value=control))
    monkeypatch.setattr(grpc_runner, "start_broker_flight_server", MagicMock(side_effect=RuntimeError("Flight failed")))
    with pytest.raises(RuntimeError, match="Flight failed"):
        grpc_runner.main()
    control.stop.assert_called_once_with(0)
    control.stop.return_value.wait.assert_called_once_with()
    comm.bcast.assert_called_once_with((grpc_runner._MSG_ABORT, None), root=0)


def test_server_startup_failure_releases_pending_control_callback(monkeypatch, tmp_path):
    comm = configure_launcher(monkeypatch, tmp_path / "run", size=2)
    control = MagicMock()
    entered = Event()
    runtime = {}

    def start_control(path, broker, start_run, *, cancel_run):
        def submit(run_id, config_json):
            entered.set()
            start_run(run_id, config_json)

        servicer = SimulatorControlServicer(broker, submit, cancel_run)
        runtime.update(servicer=servicer, broker=broker)
        servicer.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), MagicMock())
        assert entered.wait(2)
        return control

    monkeypatch.setattr(grpc_runner, "start_control_server", start_control)
    monkeypatch.setattr(grpc_runner, "start_broker_flight_server", MagicMock(side_effect=RuntimeError("Flight failed")))
    with pytest.raises(RuntimeError, match="Flight failed"):
        grpc_runner.main()
    servicer = runtime["servicer"]
    servicer._worker.join(timeout=2)
    assert not servicer._worker.is_alive()
    assert servicer.GetState(pb.GetStateRequest(run_id="run-1"), MagicMock()).state == pb.RUN_STATE_FAILED
    assert runtime["broker"].closed
    comm.bcast.assert_called_once_with((grpc_runner._MSG_ABORT, None), root=0)


def test_cli_interrupt_before_submission_releases_workers_and_closes_servers(monkeypatch, tmp_path):
    comm = configure_launcher(monkeypatch, tmp_path / "run", size=2)
    handoff = grpc_runner._MainThreadRunHandoff()
    handoff.wait_for_request = MagicMock(side_effect=KeyboardInterrupt())
    monkeypatch.setattr(grpc_runner, "_MainThreadRunHandoff", lambda: handoff)
    control, flights = MagicMock(), MagicMock()
    monkeypatch.setattr(grpc_runner, "start_runner", MagicMock(return_value=(control, flights)))
    assert grpc_runner.main() == 0
    comm.bcast.assert_called_once_with((grpc_runner._MSG_ABORT, None), root=0)
    comm.Abort.assert_not_called()
    control.stop.assert_called_once_with(0)
    flights.shutdown.assert_called_once_with()
    with pytest.raises(RuntimeError, match="Run interrupted"):
        handoff.submit("late-run", b"{}")


def test_cli_closes_flight_when_control_shutdown_raises(monkeypatch, tmp_path):
    configure_launcher(monkeypatch, tmp_path / "run", with_comm=False)
    control, flights = MagicMock(), MagicMock()
    control.stop.side_effect = RuntimeError("control shutdown failed")
    monkeypatch.setattr(grpc_runner, "start_runner", MagicMock(return_value=(control, flights)))
    monkeypatch.setattr(grpc_runner, "_rank0_main", MagicMock())
    with pytest.raises(RuntimeError, match="control shutdown failed"):
        grpc_runner.main()
    flights.shutdown.assert_called_once_with()
