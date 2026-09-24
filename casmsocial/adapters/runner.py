"""Adapter that runs CASMSocial ``CasmPop`` models through CASMSim."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pyarrow as pa
from casmsim.protocols import ObservationAdapter
from casmsim.run_state import RunState

if TYPE_CHECKING:
    from mpi4py import MPI


class _ObservationBridge:
    """Forward completed model-owned observer tables to the runtime broker."""

    step_priority = 100

    def __init__(self, observer: ObservationAdapter) -> None:
        self._observer = observer
        self._tick = 0
        self._flushed = False
        self._last_published: dict[str, pa.Table] = {}

    def initialize(self, model) -> None:  # noqa: ANN001
        pass

    def on_step(self, model) -> None:  # noqa: ANN001
        tables: dict[str, pa.Table] = model.get_observer_output_tables()
        for channel, table in tables.items():
            self._observer.publish(channel, table)
            self._last_published[channel] = table
        self._tick += 1

    def on_end(self, model) -> None:  # noqa: ANN001
        if self._flushed:
            return
        for channel, table in model.get_observer_output_tables().items():
            previous = self._last_published.get(channel)
            # Model loggers expose their latest snapshot again at shutdown.
            # Retain genuinely new terminal output, but do not replay a batch
            # already published at the last step. Step batches are not deduped.
            if previous is table or (previous is not None and previous.equals(table)):
                continue
            self._observer.publish(channel, table)
            self._last_published[channel] = table
        self.flush()

    def flush(self) -> None:
        """Close the output sink once, including when startup is cancelled."""
        if not self._flushed:
            self._observer.flush()
            self._flushed = True

    def get_output_tables(self, model) -> dict:  # noqa: ANN001
        """The bridge is a sink and does not contribute model output tables."""
        return {}

    @property
    def tick(self) -> int:
        return self._tick


class CasmPopAdapter:
    """CASMSim adapter for models registered in the CASMSocial factory."""

    def __init__(self, comm: MPI.Comm, params: dict) -> None:  # type: ignore[name-defined]
        self._comm = comm
        self._params = dict(params)
        self._observer: ObservationAdapter | None = None
        self._bridge: _ObservationBridge | None = None
        self._lock = threading.Lock()
        self._state = RunState.Pending
        self._cancelled = False
        self._model = None
        self._started = False
        self._finished = False

    def add_observer(self, observer: ObservationAdapter) -> None:
        self._observer = observer

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("adapter already started")
            self._started = True
            self._state = RunState.Running
            cancelled = self._cancelled
        succeeded = False
        try:
            if not cancelled:
                from casmsocial.__main__ import load_builtin_models
                from casmsocial.factory import Models, load_models

                params = self._params
                params["observers.arrow_server.enabled"] = False
                load_builtin_models()
                if plugins := params.get("model.plugins", []):
                    load_models(plugins)
                model = Models.create_model(params["model.name"])(self._comm, params)
                if self._observer is not None:
                    self._bridge = _ObservationBridge(self._observer)
                    model.add_observer(self._bridge)
                with self._lock:
                    self._model = model
                    cancelled = self._cancelled
                # A later cancel() signals this model even in the gap before
                # start(). CasmPop retains that signal until its first tick.
                if not cancelled:
                    model.start()
            succeeded = True
        finally:
            flushed = False
            try:
                if self._bridge is not None:
                    self._bridge.flush()
                elif self._observer is not None:
                    self._observer.flush()
                flushed = True
            finally:
                with self._lock:
                    self._model = None
                    self._finished = True
                    # CASMSim's legacy adapter enum has no Cancelled member.
                    # Keep its failure mapping, but only after shutdown/flush.
                    # The gRPC servicer independently reports CANCELLED when
                    # the requested cooperative stop returns without error.
                    self._state = (
                        RunState.Completed if succeeded and flushed and not self._cancelled else RunState.Failed
                    )

    def cancel(self) -> None:
        with self._lock:
            if self._finished or self._cancelled:
                return
            if self._model is not None:
                # Model.cancel() is a non-blocking, thread-safe signal; never
                # call model.start() or flush the sink on the request thread.
                self._model.cancel()
            self._cancelled = True

    def get_state(self) -> tuple[RunState, float, int]:
        with self._lock:
            state = self._state
        tick = self._bridge.tick if self._bridge is not None else 0
        return state, float(tick), tick


__all__ = ["CasmPopAdapter", "_ObservationBridge"]
