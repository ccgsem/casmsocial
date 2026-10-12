"""Rank-aware CASMSocial launcher for the CASMSim gRPC/Flight runtime.

Launch with ``python -m casmsocial.grpc_runner --run-dir PATH`` or
``mpirun -n N python -m casmsocial.grpc_runner --run-dir PATH``. Only rank 0
opens transport listeners. All ranks execute the submitted model on their main
thread; CASMSim's callback waits for rank 0 to finish execution and flushing.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, current_thread, main_thread
from typing import TYPE_CHECKING

import pyarrow as pa
from casmsim.flight_server import start_broker_flight_server
from casmsim.grpc_runner import ENDPOINT_FILENAME, resolve_adapter, start_control_server
from casmsim.mpi_lifecycle import get_comm
from casmsim.observation_broker import ObservationBroker
from casmsim.protocols import RunnerModelAdapter

if TYPE_CHECKING:
    from mpi4py import MPI

_CASMPOP_ENTRY_POINT = "casmsocial.adapters.runner:CasmPopAdapter"
_MSG_START = "start"
_MSG_ABORT = "abort"


@dataclass(frozen=True)
class _RunRequest:
    """One control-plane request waiting for main-thread execution."""

    run_id: str
    config_json: bytes


class _MainThreadRunHandoff:
    """Bridge CASMSim's worker callback to a rank's main thread.

    ``submit`` is suitable for CASMSim's ``start_run`` callback. It records the
    single accepted request, wakes ``wait_for_request``, and does not return
    until main-thread execution calls ``complete``. Any execution exception is
    then re-raised on the callback thread so CASMSim can report the run failed.

    Cancellation may arrive before the main thread has constructed an adapter.
    The request is latched and delivered exactly once when ``set_adapter`` is
    called.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_ready = Event()
        self._completed = Event()
        self._request: _RunRequest | None = None
        self._adapter: RunnerModelAdapter | None = None
        self._cancel_requested = False
        self._cancel_delivered = False
        self._finished = False
        self._error: BaseException | None = None

    def submit(self, run_id: str, config_json: bytes) -> None:
        """Publish one run request and block until main-thread completion."""
        with self._lock:
            if self._finished:
                if self._error is not None:
                    raise self._error
                raise RuntimeError("run already completed")
            if self._request is not None:
                raise RuntimeError("run request already submitted")
            self._request = _RunRequest(run_id, config_json)
            self._request_ready.set()
        self._completed.wait()
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def wait_for_request(self, timeout: float | None = None) -> _RunRequest:
        """Wait for CASMSim to submit the request consumed by the main thread."""
        if not self._request_ready.wait(timeout):
            raise TimeoutError("timed out waiting for a run request")
        with self._lock:
            if self._error is not None:
                raise self._error
            assert self._request is not None
            return self._request

    def set_adapter(self, adapter: RunnerModelAdapter) -> None:
        """Expose the active adapter and deliver any latched cancellation."""
        deliver_cancel = False
        with self._lock:
            if self._finished:
                raise RuntimeError("run already completed")
            if self._adapter is not None:
                raise RuntimeError("active adapter already set")
            self._adapter = adapter
            if self._cancel_requested and not self._cancel_delivered:
                self._cancel_delivered = True
                deliver_cancel = True
        if deliver_cancel:
            adapter.cancel()

    def cancel(self) -> bool:
        """Latch cancellation and signal the active adapter at most once."""
        adapter: RunnerModelAdapter | None = None
        with self._lock:
            if self._finished:
                return False
            self._cancel_requested = True
            if self._adapter is not None and not self._cancel_delivered:
                self._cancel_delivered = True
                adapter = self._adapter
        if adapter is not None:
            adapter.cancel()
        return True

    def complete(self, error: BaseException | None = None) -> None:
        """Release the callback, propagating ``error`` when one was supplied."""
        with self._lock:
            if self._request is None:
                raise RuntimeError("cannot complete before a run is submitted")
            if self._finished:
                raise RuntimeError("run already completed")
            self._finish_locked(error)

    def shutdown(self, error: BaseException) -> None:
        """Release pending or late callbacks, including before submission."""
        with self._lock:
            if not self._finished:
                self._finish_locked(error)

    def _finish_locked(self, error: BaseException | None) -> None:
        # CASMSim catches Exception, not BaseException. Preserve ordinary
        # failures, but translate process interrupts for its callback thread.
        if error is not None and not isinstance(error, Exception):
            interrupted = RuntimeError(f"Run interrupted ({type(error).__name__})")
            interrupted.__cause__ = error
            error = interrupted
        self._adapter = None
        self._error = error
        self._finished = True
        self._request_ready.set()
        self._completed.set()


class _BrokerSink:
    """Expose the public observation protocol without runtime-private imports."""

    def __init__(self, broker: ObservationBroker) -> None:
        self._broker = broker

    def publish(self, channel: str, table: pa.Table) -> None:
        self._broker.publish(channel, table)

    def flush(self) -> None:
        self._broker.close()


def _prepare_run_params(run_id: str, config_json: bytes) -> dict:
    """Normalize the control-plane request before sharing it with workers."""
    params = json.loads(config_json)
    if not isinstance(params, dict):
        raise ValueError("config_json must encode a JSON object of model parameters")
    if ("model.plugins" in params or "model.name" in params) and "runner.entry_point" not in params:
        params["runner.entry_point"] = _CASMPOP_ENTRY_POINT
    params["simulation.run_id"] = run_id
    params["observers.arrow_server.enabled"] = False
    return params


def _require_main_thread() -> None:
    """Reject background execution before accessing an MPI communicator."""
    if current_thread() is not main_thread():
        raise RuntimeError("MPI runner execution must run on the process main thread")


def _abort_collective_run(comm: MPI.Comm, error: BaseException) -> None:
    """Terminate an unrecoverable MPI job from the failing rank's main thread."""
    _require_main_thread()
    logging.getLogger(__name__).error(
        "MPI runner failed; aborting all ranks",
        exc_info=(type(error), error, error.__traceback__),
    )
    try:
        comm.Abort(1)
    finally:
        # MPI_Abort normally never returns. Prevent normal execution if a
        # communicator implementation returns or raises instead of exiting.
        raise SystemExit(1) from error


def _release_waiting_workers(comm: MPI.Comm | None, size: int) -> None:
    """Release workers that have not yet received a run to execute."""
    if size > 1:
        try:
            comm.bcast((_MSG_ABORT, None), root=0)
        except BaseException as error:
            _abort_collective_run(comm, error)


def _run_on_rank(
    comm: MPI.Comm | None,
    params: dict,
    *,
    broker: ObservationBroker | None = None,
    handoff: _MainThreadRunHandoff | None = None,
) -> None:
    """Resolve and start an adapter on this rank's main thread."""
    _require_main_thread()
    adapter = resolve_adapter(comm, params)
    if broker is not None:
        adapter.add_observer(_BrokerSink(broker))
    if handoff is not None:
        handoff.set_adapter(adapter)
    adapter.start()
    if comm is not None and comm.Get_size() > 1:
        # Rank 0's control callback must not report a terminal state while
        # workers are still executing or flushing their model output.
        comm.Barrier()


def _rank0_main(
    comm: MPI.Comm | None,
    handoff: _MainThreadRunHandoff,
    broker: ObservationBroker,
) -> None:
    """Submit one collective run from rank 0's main thread.

    Pre-dispatch failures release waiting workers and fail the control callback.
    Once collective execution is dispatched, a rank-local exception aborts the
    MPI job because another rank may be blocked in a different collective.
    """
    _require_main_thread()
    if comm is not None and comm.Get_rank() != 0:
        raise RuntimeError("rank-0 execution requires rank 0")
    size = comm.Get_size() if comm is not None else 1
    error: BaseException | None = None
    dispatched = False
    abort_required = False
    try:
        request = handoff.wait_for_request()
        params = _prepare_run_params(request.run_id, request.config_json)
        if size > 1:
            # A partial/failed broadcast is already an unsafe collective; do
            # not attempt a second, differently ordered abort-message bcast.
            dispatched = True
            comm.bcast((_MSG_START, params), root=0)
        _run_on_rank(comm, params, broker=broker, handoff=handoff)
    except BaseException as exc:
        error = exc
        if dispatched:
            abort_required = True
        else:
            _release_waiting_workers(comm, size)
    finally:
        try:
            broker.close()
        except BaseException as exc:
            if error is None:
                error = exc
        finally:
            if error is not None:
                handoff.shutdown(error)
            else:
                handoff.complete()
    if abort_required:
        _abort_collective_run(comm, error)
    if error is not None and not isinstance(error, Exception):
        raise error


def _worker_main(comm: MPI.Comm) -> None:
    """Receive rank 0's request and participate without a transport sink."""
    _require_main_thread()
    if comm.Get_rank() == 0:
        raise RuntimeError("worker execution requires a nonzero rank")
    try:
        kind, params = comm.bcast(None, root=0)
        if kind == _MSG_ABORT:
            return
        if kind != _MSG_START:
            raise ValueError(f"unexpected runner message: {kind!r}")
        _run_on_rank(comm, params)
    except BaseException as error:
        _abort_collective_run(comm, error)


class _RunSession:
    """Own one adapter and latch cancellation while it is being constructed."""

    def __init__(self, broker: ObservationBroker) -> None:
        self._broker = broker
        self._lock = Lock()
        self._adapter: RunnerModelAdapter | None = None
        self._cancel_requested = False
        self._started = False
        self._finished = False

    def cancel(self) -> bool:
        with self._lock:
            if self._finished:
                return False
            if not self._cancel_requested:
                if self._adapter is not None:
                    self._adapter.cancel()
                self._cancel_requested = True
            return True

    def run(self, run_id: str, config_json: bytes) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("session already started")
            self._started = True
        try:
            params = _prepare_run_params(run_id, config_json)
            adapter = resolve_adapter(get_comm(), params)
            adapter.add_observer(_BrokerSink(self._broker))
            with self._lock:
                self._adapter = adapter
                if self._cancel_requested:
                    adapter.cancel()
            adapter.start()
        finally:
            with self._lock:
                self._adapter = None
                self._finished = True
            self._broker.close()


def run_casmsocial_model(run_id: str, config_json: bytes, broker: ObservationBroker) -> None:
    """Launch a CASMSocial model through CASMSim's generic adapter protocol."""
    _RunSession(broker).run(run_id, config_json)


def start_runner(
    run_dir: Path,
    *,
    broker: ObservationBroker | None = None,
    handoff: _MainThreadRunHandoff | None = None,
):
    """Start rank 0's control and Flight endpoints.

    The CLI supplies a handoff for execution on its main thread. Embedded
    single-process callers may omit it to retain the callback-driven session.
    """
    _require_main_thread()
    comm = get_comm()
    if comm is not None:
        if comm.Get_rank() != 0:
            raise RuntimeError("only rank 0 may start runner endpoints")
        if comm.Get_size() > 1 and handoff is None:
            raise RuntimeError("multi-rank runner endpoints require a main-thread handoff")
    if broker is None:
        broker = ObservationBroker()
    if handoff is None:
        session = _RunSession(broker)
        start_run, cancel_run = session.run, session.cancel
    else:
        start_run, cancel_run = handoff.submit, handoff.cancel
    control = start_control_server(
        run_dir,
        broker,
        start_run,
        cancel_run=cancel_run,
    )
    flights = None
    try:
        flights = start_broker_flight_server(run_dir, broker)
        manifest_path = run_dir / ENDPOINT_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["flight"] = {"address": f"127.0.0.1:{flights.port}", "protocol": "arrow.flight"}
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        manifest_path.chmod(0o600)
    except BaseException:
        try:
            control.stop(0).wait()
        finally:
            if flights is not None:
                flights.shutdown()
        raise
    return control, flights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    _require_main_thread()
    comm = get_comm()
    if comm is not None and comm.Get_rank() != 0:
        _worker_main(comm)
        return 0
    broker = ObservationBroker()
    handoff = _MainThreadRunHandoff()
    try:
        control, flights = start_runner(args.run_dir, broker=broker, handoff=handoff)
    except BaseException as error:
        # Workers entered their broadcast path before rank 0 opened listeners.
        # They must be released if listener or manifest initialization fails.
        handoff.shutdown(error)
        try:
            broker.close()
        finally:
            _release_waiting_workers(comm, comm.Get_size() if comm is not None else 1)
        raise
    try:
        _rank0_main(comm, handoff, broker)
        # Keep completed observations available until the supervisor stops the
        # process, matching the existing single-run transport lifecycle.
        control.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            control.stop(0).wait()
        finally:
            flights.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
