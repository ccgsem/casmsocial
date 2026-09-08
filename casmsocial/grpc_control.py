"""Loopback gRPC control adapter for a single CASMSocial runner."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from concurrent import futures
from pathlib import Path
from threading import Event, Lock

import grpc
import pyarrow as pa
from pyarrow import ipc

from casmsocial.observation_broker import ObservationBroker, ObservationCursorExpiredError
from casmsocial.proto import casm_runner_pb2 as pb2, casm_runner_pb2_grpc as pb2_grpc

ENDPOINT_FILENAME = "runner_endpoints.json"


def secure_run_directory(path: Path) -> None:
    """Create a local run directory and require owner-only permissions."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"run directory must be owner-only: {path}")


class SimulatorControlServicer(pb2_grpc.SimulatorControlServicer):
    """Atomically accepts one run and exposes its broker-backed observations.

    In the multi-rank runner, the model is driven from rank 0's main thread
    rather than a daemon thread, so that all MPI collective calls stay on the
    main thread (MPI_THREAD_FUNNELED safety).  ``Start`` therefore only stashes
    the request and fires ``_start_event``; ``wait_for_run`` lets the main
    thread block until a run arrives.  ``complete_run`` is called by the main
    thread once the model finishes.

    For single-rank deployments the same flow applies — the main thread runs
    the model without issuing any MPI collectives.
    """

    def __init__(self, broker: ObservationBroker) -> None:
        self._broker = broker
        self._lock = Lock()
        self._run_id: str | None = None
        self._state = pb2.RUN_STATE_INITIALIZING
        self._model = None  # set via _set_model once the model is instantiated
        self._pending_run: tuple[str, bytes] | None = None
        self._start_event = Event()

    def _set_model(self, model) -> None:
        """Store a reference to the running model for cooperative cancellation.

        Called from rank 0's main thread before ``model.start()``.
        """
        with self._lock:
            self._model = model

    def wait_for_run(self) -> tuple[str, bytes]:
        """Block until a ``Start`` RPC arrives and return ``(run_id, config_json)``.

        Must be called from rank 0's main thread.
        """
        self._start_event.wait()
        assert self._pending_run is not None  # set before event is fired
        return self._pending_run

    def complete_run(self, *, success: bool) -> None:
        """Transition to terminal state and close the broker.

        Called from rank 0's main thread after the model finishes.
        """
        with self._lock:
            if self._state == pb2.RUN_STATE_RUNNING:
                self._state = pb2.RUN_STATE_COMPLETED if success else pb2.RUN_STATE_FAILED
        self._broker.close()

    # ------------------------------------------------------------------
    # gRPC RPC handlers (called from gRPC thread-pool threads)
    # ------------------------------------------------------------------

    def Start(self, request, context):
        if not request.run_id or not request.config_json:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id and config_json are required")
        with self._lock:
            if self._run_id is not None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "this process already accepted a run")
            self._run_id = request.run_id
            self._state = pb2.RUN_STATE_RUNNING
            self._pending_run = (request.run_id, request.config_json)
        # Signal the main thread.  Set event after releasing the lock so the
        # main thread never races on _pending_run.
        self._start_event.set()
        return pb2.StartResponse(run_id=request.run_id)

    def Cancel(self, request, context):
        """Request cooperative cancellation of the running simulation.

        Sets a flag checked at the top of each model tick.  The model will stop
        cleanly after the current tick completes.  Returns ``acknowledged=True``
        once the flag is set; returns ``acknowledged=False`` only when no run is
        active or the model reference is not yet available.
        """
        with self._lock:
            if self._state != pb2.RUN_STATE_RUNNING or self._model is None:
                return pb2.CancelResponse(acknowledged=False)
            self._state = pb2.RUN_STATE_CANCELLED
            model = self._model
        model.cancel()
        return pb2.CancelResponse(acknowledged=True)

    def GetState(self, request, context):
        with self._lock:
            if request.run_id != self._run_id:
                context.abort(grpc.StatusCode.NOT_FOUND, "unknown run_id")
            return pb2.StateResponse(run_id=request.run_id, state=self._state)

    def StreamObs(self, request, context) -> Iterator[pb2.ObsBatch]:
        with self._lock:
            if request.run_id != self._run_id:
                context.abort(grpc.StatusCode.NOT_FOUND, "unknown run_id")
        try:
            result = self._broker.read(request.channel, start_batch_id=request.start_tick)
        except ObservationCursorExpiredError as error:
            context.abort(grpc.StatusCode.OUT_OF_RANGE, str(error))
        for batch in result.batches:
            if not context.is_active():
                return
            sink = pa.BufferOutputStream()
            with ipc.new_stream(sink, batch.table.schema) as writer:
                writer.write_table(batch.table)
            yield pb2.ObsBatch(channel=batch.channel, tick=batch.batch_id, arrow_ipc=sink.getvalue().to_pybytes())


def start_control_server(
    run_dir: Path,
    broker: ObservationBroker,
) -> tuple[object, SimulatorControlServicer]:
    """Start a loopback-only control server and write its endpoint manifest.

    Returns ``(grpc_server, servicer)``.  The caller is responsible for calling
    ``servicer.wait_for_run()`` and ``servicer.complete_run()`` from the main
    thread to drive the run lifecycle.
    """
    secure_run_directory(run_dir)
    servicer = SimulatorControlServicer(broker)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_SimulatorControlServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if not port:
        raise RuntimeError("could not bind loopback gRPC control listener")
    server.start()
    (run_dir / ENDPOINT_FILENAME).write_text(
        json.dumps({"control": {"address": f"127.0.0.1:{port}", "protocol": "casm.runner.v1"}}) + "\n"
    )
    os.chmod(run_dir / ENDPOINT_FILENAME, 0o600)
    return server, servicer
