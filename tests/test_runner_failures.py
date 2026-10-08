"""Failure boundaries around startup, handoff shutdown, and MPI execution."""

from concurrent.futures import ThreadPoolExecutor
from threading import current_thread, main_thread
from unittest.mock import Mock

import pytest
from casmsim.grpc_runner import SimulatorControlServicer
from casmsim.observation_broker import ObservationBroker
from casmsim.proto import casm_runner_pb2 as pb

from casmsocial import grpc_runner


def make_comm(rank):
    comm = Mock()
    comm.Get_rank.return_value = rank
    comm.Get_size.return_value = 2
    return comm


@pytest.fixture
def pending_run():
    handoff = grpc_runner._MainThreadRunHandoff()
    with ThreadPoolExecutor(max_workers=1) as pool:
        callback = pool.submit(handoff.submit, "run-1", b'{"model.name":"test"}')
        handoff.wait_for_request(timeout=2)
        try:
            yield handoff, callback
        finally:
            handoff.shutdown(RuntimeError("test cleanup"))


def test_startup_interrupt_releases_workers_and_rejects_late_submission(monkeypatch):
    comm = make_comm(0)
    handoff = grpc_runner._MainThreadRunHandoff()
    monkeypatch.setattr(handoff, "wait_for_request", Mock(side_effect=KeyboardInterrupt()))
    broker = ObservationBroker()
    with pytest.raises(KeyboardInterrupt):
        grpc_runner._rank0_main(comm, handoff, broker)
    comm.bcast.assert_called_once_with((grpc_runner._MSG_ABORT, None), root=0)
    comm.Abort.assert_not_called()
    assert broker.closed
    assert not handoff.cancel()
    with pytest.raises(RuntimeError, match="Run interrupted"):
        handoff.submit("late-run", b"{}")


def test_shutdown_wakes_request_waiter_and_preserves_first_error():
    handoff = grpc_runner._MainThreadRunHandoff()
    first_error = ValueError("startup failed")
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(handoff.wait_for_request, timeout=2)
        handoff.shutdown(first_error)
        handoff.shutdown(RuntimeError("secondary failure"))
        with pytest.raises(ValueError, match="startup failed") as caught:
            waiter.result(timeout=2)
    assert caught.value is first_error


@pytest.mark.parametrize("phase", ["broadcast", "resolve", "start", "barrier"])
def test_rank0_failure_after_dispatch_releases_callback_then_aborts(monkeypatch, pending_run, phase):
    handoff, callback = pending_run
    comm = make_comm(0)
    broker = ObservationBroker()
    adapter = Mock()
    error = ValueError("injected collective execution failure")
    resolver = Mock(return_value=adapter)
    target = {"broadcast": comm.bcast, "resolve": resolver, "start": adapter.start, "barrier": comm.Barrier}[phase]
    target.side_effect = error
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)

    def abort(code):
        assert current_thread() is main_thread()
        assert broker.closed
        assert not handoff.cancel()

    comm.Abort.side_effect = abort
    with pytest.raises(SystemExit) as exit_error:
        grpc_runner._rank0_main(comm, handoff, broker)
    assert exit_error.value.code == 1
    with pytest.raises(ValueError) as callback_error:
        callback.result(timeout=2)
    assert callback_error.value is error
    comm.Abort.assert_called_once_with(1)
    assert comm.bcast.call_count == 1  # Never send another message after dispatch.


@pytest.mark.parametrize("phase", ["receive", "message", "resolve", "start", "barrier"])
def test_worker_failure_aborts_job_instead_of_leaving_peers_blocked(monkeypatch, phase):
    comm = make_comm(1)
    comm.bcast.return_value = (grpc_runner._MSG_START, {"runner.entry_point": "custom:Adapter"})
    adapter = Mock()
    resolver = Mock(return_value=adapter)
    if phase == "message":
        comm.bcast.return_value = ("invalid-message", None)
    else:
        target = {"receive": comm.bcast, "resolve": resolver, "start": adapter.start, "barrier": comm.Barrier}[phase]
        target.side_effect = ValueError("injected worker failure")
    monkeypatch.setattr(grpc_runner, "resolve_adapter", resolver)

    def abort(code):
        assert current_thread() is main_thread()

    comm.Abort.side_effect = abort
    with pytest.raises(SystemExit) as caught:
        grpc_runner._worker_main(comm)
    assert caught.value.code == 1
    comm.Abort.assert_called_once_with(1)


def test_failed_abort_message_falls_back_to_mpi_abort(monkeypatch):
    comm = make_comm(0)
    comm.bcast.side_effect = RuntimeError("broadcast failed")
    handoff = grpc_runner._MainThreadRunHandoff()
    monkeypatch.setattr(handoff, "wait_for_request", Mock(side_effect=KeyboardInterrupt()))
    broker = ObservationBroker()
    with pytest.raises(SystemExit):
        grpc_runner._rank0_main(comm, handoff, broker)
    comm.Abort.assert_called_once_with(1)
    assert broker.closed
    assert not handoff.cancel()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_single_rank_interrupt_fails_control_state_without_stranding_worker(monkeypatch, interrupt):
    handoff = grpc_runner._MainThreadRunHandoff()
    broker = ObservationBroker()
    servicer = SimulatorControlServicer(broker, handoff.submit, handoff.cancel)
    adapter = Mock()
    adapter.start.side_effect = interrupt("credential=secret")
    monkeypatch.setattr(grpc_runner, "resolve_adapter", Mock(return_value=adapter))
    context = Mock()
    servicer.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), context)
    try:
        with pytest.raises(interrupt):
            grpc_runner._rank0_main(None, handoff, broker)
    finally:
        handoff.shutdown(RuntimeError("test cleanup"))
        servicer._worker.join(timeout=2)
    assert not servicer._worker.is_alive()
    state = servicer.GetState(pb.GetStateRequest(run_id="run-1"), context)
    assert state.state == pb.RUN_STATE_FAILED
    assert "credential" not in state.status_message
    assert broker.closed
