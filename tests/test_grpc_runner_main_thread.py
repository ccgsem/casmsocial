"""Main-thread request execution across simulated MPI ranks."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import current_thread, main_thread
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from casmsim.observation_broker import ObservationBroker

from casmsocial import grpc_runner


def make_comm(rank, size=2):
    comm = MagicMock()
    comm.Get_rank.return_value = rank
    comm.Get_size.return_value = size
    return comm


@pytest.fixture
def submitted_run():
    """Keep a control callback waiting while the test drives the main thread."""
    handoff = grpc_runner._MainThreadRunHandoff()
    pool = ThreadPoolExecutor(max_workers=1)
    callbacks = []

    def submit(config_json):
        callback = pool.submit(handoff.submit, "run-1", config_json)
        callbacks.append(callback)
        handoff.wait_for_request(timeout=2)
        return handoff, callback

    yield submit
    if callbacks:
        try:
            handoff.complete()
        except RuntimeError:
            # The main-thread path normally completed the handoff already.
            pass
    pool.shutdown(wait=True)


@pytest.mark.parametrize("entry_point", [None, "custom:Adapter"])
def test_rank0_broadcasts_normalized_request_and_workers_start_on_main_thread(monkeypatch, submitted_run, entry_point):
    config = {"model.name": "test", "simulation.run_id": "untrusted", "observers.arrow_server.enabled": True}
    if entry_point is not None:
        config["runner.entry_point"] = entry_point
    handoff, callback = submitted_run(json.dumps(config).encode())
    root_comm, worker_comm = make_comm(0), make_comm(1)
    root_adapter, worker_adapter = MagicMock(), MagicMock()
    broker = ObservationBroker()
    requests = []

    def broadcast(message, root):
        assert current_thread() is main_thread()
        assert root == 0
        # Mimic MPI serialization rather than sharing mutable parameters.
        requests.append(deepcopy(message))

    root_comm.bcast.side_effect = broadcast

    def resolve(comm, params):
        assert current_thread() is main_thread()
        assert params == {
            "model.name": "test",
            "simulation.run_id": "run-1",
            "observers.arrow_server.enabled": False,
            "runner.entry_point": entry_point or "casmsocial.adapters.runner:CasmPopAdapter",
        }
        return root_adapter if comm is root_comm else worker_adapter

    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolve)

    def root_start():
        assert current_thread() is main_thread()
        assert not callback.done()
        sink = root_adapter.add_observer.call_args.args[0]
        sink.publish("agents", pa.table({"id": [1]}))
        sink.flush()
        # Flushing the broker alone must not release the control callback.
        assert broker.closed
        assert not callback.done()

    root_adapter.start.side_effect = root_start

    def wait_for_workers():
        assert current_thread() is main_thread()
        assert broker.closed
        assert not callback.done()

    root_comm.Barrier.side_effect = wait_for_workers

    def worker_start():
        assert current_thread() is main_thread()

    worker_adapter.start.side_effect = worker_start
    grpc_runner._rank0_main(root_comm, handoff, broker)
    callback.result(timeout=2)
    assert len(requests) == 1
    assert requests[0][0] == grpc_runner._MSG_START
    worker_comm.bcast.return_value = requests[0]
    grpc_runner._worker_main(worker_comm)
    worker_comm.bcast.assert_called_once_with(None, root=0)
    root_adapter.start.assert_called_once_with()
    worker_adapter.start.assert_called_once_with()
    worker_adapter.add_observer.assert_not_called()
    root_comm.Barrier.assert_called_once_with()
    worker_comm.Barrier.assert_called_once_with()
    assert broker.read("agents").batches[0].table.to_pydict() == {"id": [1]}


@pytest.mark.parametrize("with_comm", [False, True])
def test_single_rank_execution_skips_broadcast(monkeypatch, submitted_run, with_comm):
    handoff, callback = submitted_run(b'{"model.name":"test"}')
    comm = make_comm(0, size=1) if with_comm else None
    adapter = MagicMock()
    resolver = MagicMock(return_value=adapter)
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)
    broker = ObservationBroker()
    grpc_runner._rank0_main(comm, handoff, broker)
    callback.result(timeout=2)
    assert resolver.call_args.args[0] is comm
    adapter.start.assert_called_once_with()
    if comm is not None:
        comm.bcast.assert_not_called()
    assert broker.closed


@pytest.mark.parametrize("config", [b"[]", b"invalid-json"])
def test_invalid_request_releases_workers_and_fails_control_callback(monkeypatch, submitted_run, config):
    handoff, callback = submitted_run(config)
    root_comm, worker_comm = make_comm(0), make_comm(1)
    resolver = MagicMock()
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)
    broker = ObservationBroker()
    grpc_runner._rank0_main(root_comm, handoff, broker)
    with pytest.raises(ValueError):
        callback.result(timeout=2)
    root_comm.bcast.assert_called_once_with((grpc_runner._MSG_ABORT, None), root=0)
    worker_comm.bcast.return_value = root_comm.bcast.call_args.args[0]
    grpc_runner._worker_main(worker_comm)
    resolver.assert_not_called()
    assert broker.closed


@pytest.mark.parametrize("failure_phase", ["resolve", "start"])
def test_single_rank_execution_failure_propagates_to_control_callback(monkeypatch, submitted_run, failure_phase):
    handoff, callback = submitted_run(b'{"model.name":"test"}')
    adapter = MagicMock()
    error = ValueError("model execution failed")
    resolver = MagicMock(return_value=adapter)
    if failure_phase == "resolve":
        resolver.side_effect = error
    else:
        adapter.start.side_effect = error
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)
    broker = ObservationBroker()
    grpc_runner._rank0_main(None, handoff, broker)
    with pytest.raises(ValueError, match="model execution failed") as caught:
        callback.result(timeout=2)
    assert caught.value is error
    assert broker.closed
    assert not handoff.cancel()


def test_main_thread_execution_delivers_startup_cancellation(monkeypatch, submitted_run):
    handoff, callback = submitted_run(b'{"model.name":"test"}')
    assert handoff.cancel()
    adapter = MagicMock()
    monkeypatch.setattr(grpc_runner, "resolve_adapter", MagicMock(return_value=adapter))
    grpc_runner._rank0_main(None, handoff, ObservationBroker())
    callback.result(timeout=2)
    assert adapter.method_calls[0][0] == "add_observer"
    assert adapter.method_calls[1][0] == "cancel"
    assert adapter.method_calls[2][0] == "start"
    adapter.cancel.assert_called_once_with()


def test_broker_shutdown_failure_reaches_control_callback(monkeypatch, submitted_run):
    handoff, callback = submitted_run(b'{"model.name":"test"}')
    monkeypatch.setattr(grpc_runner, "resolve_adapter", MagicMock(return_value=MagicMock()))
    broker = MagicMock()
    broker.close.side_effect = ValueError("broker shutdown failed")
    grpc_runner._rank0_main(None, handoff, broker)
    with pytest.raises(ValueError, match="broker shutdown failed"):
        callback.result(timeout=2)


def test_background_execution_is_rejected_before_any_mpi_access():
    comm = make_comm(0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        callback = pool.submit(grpc_runner._rank0_main, comm, grpc_runner._MainThreadRunHandoff(), ObservationBroker())
        with pytest.raises(RuntimeError, match="process main thread"):
            callback.result(timeout=2)
    assert comm.mock_calls == []
