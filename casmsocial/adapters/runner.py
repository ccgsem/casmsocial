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

    def initialize(self, model) -> None:  # noqa: ANN001
        pass

    def on_step(self, model) -> None:  # noqa: ANN001
        tables: dict[str, pa.Table] = model.get_observer_output_tables()
        for channel, table in tables.items():
            self._observer.publish(channel, table)
        self._tick += 1

    def on_end(self, model) -> None:  # noqa: ANN001
        for channel, table in model.get_observer_output_tables().items():
            self._observer.publish(channel, table)
        self._observer.flush()

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

    def add_observer(self, observer: ObservationAdapter) -> None:
        self._observer = observer

    def start(self) -> None:
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
            if self._cancelled:
                self._state = RunState.Failed
                if self._observer is not None:
                    self._observer.flush()
                return
            self._state = RunState.Running
        try:
            model.start()
        except Exception:
            with self._lock:
                self._state = RunState.Failed
            if self._observer is not None:
                self._observer.flush()
            raise
        with self._lock:
            if self._state == RunState.Running:
                self._state = RunState.Completed

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._state == RunState.Running:
                self._state = RunState.Failed

    def get_state(self) -> tuple[RunState, float, int]:
        with self._lock:
            state = self._state
        tick = self._bridge.tick if self._bridge is not None else 0
        return state, float(tick), tick


__all__ = ["CasmPopAdapter", "_ObservationBridge"]
