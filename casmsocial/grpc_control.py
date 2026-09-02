"""Loopback gRPC control adapter for a single CASMSocial runner."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from concurrent import futures
from pathlib import Path
from threading import Lock, Thread

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
    """Atomically accepts one run and exposes its broker-backed observations."""

    def __init__(self, broker: ObservationBroker, start_run: Callable[[str, bytes], None]) -> None:
        self._broker = broker
        self._start_run = start_run
        self._lock = Lock()
        self._run_id: str | None = None
        self._state = pb2.RUN_STATE_INITIALIZING
        self._worker: Thread | None = None

    def Start(self, request, context):
        if not request.run_id or not request.config_json:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id and config_json are required")
        with self._lock:
            if self._run_id is not None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "this process already accepted a run")
            self._run_id = request.run_id  # Reserve before calling user code: fixes concurrent Start races.
            self._state = pb2.RUN_STATE_RUNNING
            self._worker = Thread(
                target=self._run,
                args=(request.run_id, request.config_json),
                name=f"casmsocial-run-{request.run_id}",
                daemon=True,
            )
            self._worker.start()
        return pb2.StartResponse(run_id=request.run_id)

    def _run(self, run_id: str, config_json: bytes) -> None:
        """Run the model off the gRPC request thread and expose terminal state."""
        try:
            self._start_run(run_id, config_json)
        except Exception:
            with self._lock:
                if self._state != pb2.RUN_STATE_CANCELLED:
                    self._state = pb2.RUN_STATE_FAILED
            self._broker.close()
            return
        with self._lock:
            if self._state == pb2.RUN_STATE_RUNNING:
                self._state = pb2.RUN_STATE_COMPLETED
        self._broker.close()

    def Cancel(self, request, context):
        """Report that cooperative model cancellation is not yet supported.

        CASMSocial's current model lifecycle has no cancellation hook. Returning
        ``acknowledged=False`` prevents callers from treating a still-running
        simulation as cancelled.
        """
        with self._lock:
            return pb2.CancelResponse(acknowledged=False)

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


def start_control_server(run_dir: Path, broker: ObservationBroker, start_run: Callable[[str, bytes], None]):
    """Start a loopback-only control server and write its endpoint manifest."""
    secure_run_directory(run_dir)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_SimulatorControlServicer_to_server(SimulatorControlServicer(broker, start_run), server)
    port = server.add_insecure_port("127.0.0.1:0")
    if not port:
        raise RuntimeError("could not bind loopback gRPC control listener")
    server.start()
    (run_dir / ENDPOINT_FILENAME).write_text(json.dumps({"control": {"address": f"127.0.0.1:{port}", "protocol": "casm.runner.v1"}}) + "\n")
    os.chmod(run_dir / ENDPOINT_FILENAME, 0o600)
    return server
