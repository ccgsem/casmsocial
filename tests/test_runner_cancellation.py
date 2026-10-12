"""Cancellation through the real control/Flight servers and CASMSocial adapter.

Small event-gated models replace scientific initialization; no input dataset or
MPI ensemble is needed. Events make startup and flush races deterministic.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import grpc
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from casmsim.grpc_runner import ENDPOINT_FILENAME
from casmsim.observation_broker import ObservationBroker
from casmsim.proto import casm_runner_pb2 as pb, casm_runner_pb2_grpc as rpc
from casmsim.run_state import RunState

from casmsocial import grpc_runner
from casmsocial.adapters.runner import CasmPopAdapter


def install_model(monkeypatch, constructor):
    factory = Mock()
    factory.Models.create_model.return_value = constructor
    monkeypatch.setitem(sys.modules, "casmsocial.factory", factory)
    monkeypatch.setitem(sys.modules, "casmsocial.__main__", Mock())
    monkeypatch.setattr(grpc_runner, "get_comm", lambda: None)
    return factory


def wait_for_state(client, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.GetState(pb.GetStateRequest(run_id="run-1"), timeout=2)
        if state.state == expected:
            return state
        Event().wait(0.01)
    pytest.fail(f"expected state {expected}, got {state}")


@pytest.mark.parametrize("phase", ["resolver", "constructor", "before_start", "running", "flush"])
def test_rpc_cancel_reaches_model_and_waits_for_shutdown(monkeypatch, tmp_path, phase):
    entered, release, cancelled, stopped = Event(), Event(), Event(), Event()
    params_seen = {}
    calls = []

    def gate():
        entered.set()
        assert release.wait(5), "test did not release worker"

    class Model:
        def __init__(self, comm, params):
            params_seen.update(params)
            self.observer = None
            if phase == "constructor":
                gate()

        def add_observer(self, observer):
            self.observer = observer

        def cancel(self):
            calls.append("cancel")
            cancelled.set()

        def get_observer_output_tables(self):
            return {}

        def start(self):
            calls.append("start")
            if phase == "before_start":
                gate()
                assert cancelled.is_set()
            elif phase == "running":
                entered.set()
                assert cancelled.wait(5), "cancel never reached model"
                assert release.wait(5)
            elif phase == "flush":
                # Block the final sink flush, after model execution has ended.
                original_flush = self.observer._observer.flush

                def flush():
                    gate()
                    original_flush()

                self.observer._observer.flush = flush
            self.observer.on_end(self)
            stopped.set()

    install_model(monkeypatch, Model)
    if phase == "resolver":
        real_resolver = grpc_runner.resolve_adapter

        def resolve(comm, params):
            gate()
            return real_resolver(comm, params)

        monkeypatch.setattr(grpc_runner, "resolve_adapter", resolve)

    control, flights = grpc_runner.start_runner(tmp_path)
    manifest = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())
    channel = grpc.insecure_channel(manifest["control"]["address"])
    client = rpc.SimulatorControlStub(channel)
    try:
        client.Start(
            pb.StartRequest(
                run_id="run-1",
                config_json=json.dumps({"model.name": "test", "simulation.run_id": "untrusted"}).encode(),
            ),
            timeout=2,
        )
        assert entered.wait(5)
        request = pb.CancelRequest(run_id="run-1")
        assert client.Cancel(request, timeout=2).acknowledged
        assert client.Cancel(request, timeout=2).acknowledged
        assert client.GetState(pb.GetStateRequest(run_id="run-1"), timeout=2).state == pb.RUN_STATE_RUNNING
        assert not stopped.is_set()
        release.set()
        wait_for_state(client, pb.RUN_STATE_CANCELLED)
        assert not client.Cancel(request, timeout=2).acknowledged
        if phase in {"resolver", "constructor"}:
            assert "start" not in calls
        else:
            assert calls.count("cancel") == 1
            assert stopped.is_set()
        # A cancelled run is isolated from requests for a different identity.
        with pytest.raises(grpc.RpcError) as error:
            client.Cancel(pb.CancelRequest(run_id="another-run"), timeout=2)
        assert error.value.code() == grpc.StatusCode.NOT_FOUND
        if phase != "resolver":
            assert params_seen["simulation.run_id"] == "run-1"
            assert params_seen["observers.arrow_server.enabled"] is False
    finally:
        release.set()
        cancelled.set()
        channel.close()
        control.stop(0).wait()
        flights.shutdown()


def test_adapter_keeps_running_until_flush_and_cancel_is_idempotent(monkeypatch):
    entered, finish, flushing, flush_release = Event(), Event(), Event(), Event()
    model = Mock()

    def start():
        entered.set()
        assert finish.wait(5)

    def flush():
        flushing.set()
        assert flush_release.wait(5)

    model.start.side_effect = start
    install_model(monkeypatch, lambda *_: model)
    sink = Mock()
    sink.flush.side_effect = flush
    adapter = CasmPopAdapter(None, {"model.name": "test"})
    adapter.add_observer(sink)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.start)
        try:
            assert entered.wait(5)
            adapter.cancel()
            adapter.cancel()
            model.cancel.assert_called_once()
            assert adapter.get_state()[0] == RunState.Running
            finish.set()
            assert flushing.wait(5)
            assert adapter.get_state()[0] == RunState.Running
        finally:
            finish.set()
            flush_release.set()
        future.result(timeout=5)
    assert adapter.get_state()[0] == RunState.Failed  # Legacy enum has no Cancelled.
    sink.flush.assert_called_once()
    adapter.cancel()
    model.cancel.assert_called_once()


def test_cancel_before_adapter_start_skips_model_and_flushes(monkeypatch):
    constructor = Mock()
    install_model(monkeypatch, constructor)
    sink = Mock()
    adapter = CasmPopAdapter(None, {"model.name": "test"})
    adapter.add_observer(sink)
    adapter.cancel()
    adapter.start()
    constructor.assert_not_called()
    sink.flush.assert_called_once()
    with pytest.raises(RuntimeError, match="already started"):
        adapter.start()


@pytest.mark.parametrize("phase", ["before_start", "constructor", "observer_registration"])
def test_multi_rank_startup_cancel_still_constructs_and_starts_model(monkeypatch, phase):
    comm = Mock()
    comm.Get_size.return_value = 2
    model, sink = Mock(), Mock()
    adapter = CasmPopAdapter(comm, {"model.name": "test"})
    adapter.add_observer(sink)

    def construct(actual_comm, params):
        assert actual_comm is comm
        if phase == "constructor":
            adapter.cancel()
            adapter.cancel()
        return model

    install_model(monkeypatch, construct)
    if phase == "before_start":
        adapter.cancel()
        adapter.cancel()
    elif phase == "observer_registration":
        model.add_observer.side_effect = lambda _: adapter.cancel()

    def start():
        model.cancel.assert_called_once_with()
        assert adapter.get_state()[0] == RunState.Running
        adapter.cancel()
        model.cancel.assert_called_once_with()

    model.start.side_effect = start
    adapter.start()
    model.start.assert_called_once_with()
    sink.flush.assert_called_once_with()
    assert adapter.get_state()[0] == RunState.Failed


def test_casmpop_cancel_before_first_tick_stops_without_advancing():
    from casmsocial.casmpop import CasmPop

    model = CasmPop.__new__(CasmPop)
    model._cancel_event = Event()
    model.size = 1
    model.rank = 0
    model.runner = Mock()
    model.runner.schedule.tick = 0
    model.cal = Mock()
    model.cancel()
    model.step()
    model.runner.schedule_stop.assert_called_once_with(0)
    model.cal.increment.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["constructor", "start", "flush"])
def test_adapter_failure_flushes_and_reports_failure(monkeypatch, failure_phase):
    model, sink = Mock(), Mock()
    constructor = Mock(return_value=model)
    target = {"constructor": constructor, "start": model.start, "flush": sink.flush}[failure_phase]
    target.side_effect = ValueError("test failure")
    install_model(monkeypatch, constructor)
    adapter = CasmPopAdapter(None, {"model.name": "test"})
    adapter.add_observer(sink)
    with pytest.raises(ValueError, match="test failure"):
        adapter.start()
    assert adapter.get_state()[0] == RunState.Failed
    sink.flush.assert_called_once()


def test_session_preserves_explicit_adapter_and_run_identity(monkeypatch):
    adapter = Mock()
    resolver = Mock(return_value=adapter)
    monkeypatch.setattr(grpc_runner, "get_comm", lambda: None)
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)
    broker = ObservationBroker()
    session = grpc_runner._RunSession(broker)
    assert session.cancel()
    session.run("run-1", b'{"runner.entry_point":"custom:Adapter","model.name":"test"}')
    params = resolver.call_args.args[1]
    assert params["runner.entry_point"] == "custom:Adapter"
    assert params["simulation.run_id"] == "run-1"
    adapter.cancel.assert_called_once()
    adapter.start.assert_called_once()
    assert broker.closed
    assert not session.cancel()


def test_completed_run_output_remains_retrievable_via_flight(monkeypatch, tmp_path):
    table = pa.table({"run_id": ["run-1"], "tick": [60], "agent_id": [1]})

    class Model:
        def add_observer(self, observer):
            self.observer = observer

        def get_observer_output_tables(self):
            return {"agent_log": table}

        def start(self):
            self.observer.on_end(self)

    install_model(monkeypatch, lambda *_: Model())
    control, flights = grpc_runner.start_runner(tmp_path)
    manifest = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())
    try:
        with grpc.insecure_channel(manifest["control"]["address"]) as channel:
            client = rpc.SimulatorControlStub(channel)
            client.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), timeout=2)
            wait_for_state(client, pb.RUN_STATE_COMPLETED)
            assert not client.Cancel(pb.CancelRequest(run_id="run-1"), timeout=2).acknowledged
        with flight.FlightClient(f"grpc://{manifest['flight']['address']}") as client:
            result = client.do_get(flight.Ticket(b"agent_log")).read_all()
        assert result.equals(table)
    finally:
        control.stop(0).wait()
        flights.shutdown()


def test_cancellation_does_not_hide_worker_error(monkeypatch):
    # Exercise the runtime's error path after a real adapter cancellation.
    from casmsim.grpc_runner import SimulatorControlServicer

    entered, release = Event(), Event()
    model = Mock()

    def start():
        entered.set()
        assert release.wait(5)
        raise ValueError("credential=secret")

    model.start.side_effect = start
    install_model(monkeypatch, lambda *_: model)
    broker = ObservationBroker()
    session = grpc_runner._RunSession(broker)
    server = SimulatorControlServicer(broker, session.run, session.cancel)
    context = Mock()
    server.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), context)
    try:
        assert entered.wait(5)
        assert server.Cancel(pb.CancelRequest(run_id="run-1"), context).acknowledged
    finally:
        release.set()
        server._worker.join(5)
    state = server.GetState(pb.GetStateRequest(run_id="run-1"), context)
    assert state.state == pb.RUN_STATE_FAILED
    assert "credential" not in state.status_message
    assert broker.closed
