"""CASMSocial launcher for the standalone CASMSim gRPC/Flight runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock

import pyarrow as pa
from casmsim.flight_server import start_broker_flight_server
from casmsim.grpc_runner import ENDPOINT_FILENAME, resolve_adapter, start_control_server
from casmsim.mpi_lifecycle import get_comm
from casmsim.observation_broker import ObservationBroker
from casmsim.protocols import RunnerModelAdapter

_CASMPOP_ENTRY_POINT = "casmsocial.adapters.runner:CasmPopAdapter"


class _BrokerSink:
    """Expose the public observation protocol without runtime-private imports."""

    def __init__(self, broker: ObservationBroker) -> None:
        self._broker = broker

    def publish(self, channel: str, table: pa.Table) -> None:
        self._broker.publish(channel, table)

    def flush(self) -> None:
        self._broker.close()


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
            params = json.loads(config_json)
            if not isinstance(params, dict):
                raise ValueError("config_json must encode a JSON object of model parameters")
            if ("model.plugins" in params or "model.name" in params) and "runner.entry_point" not in params:
                params["runner.entry_point"] = _CASMPOP_ENTRY_POINT
            params["simulation.run_id"] = run_id
            params["observers.arrow_server.enabled"] = False
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


def start_runner(run_dir: Path):
    """Start loopback-only control and Arrow Flight endpoints."""
    broker = ObservationBroker()
    session = _RunSession(broker)
    control = start_control_server(
        run_dir,
        broker,
        session.run,
        cancel_run=session.cancel,
    )
    flights = start_broker_flight_server(run_dir, broker)
    manifest_path = run_dir / ENDPOINT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["flight"] = {"address": f"127.0.0.1:{flights.port}", "protocol": "arrow.flight"}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    return control, flights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    control, flights = start_runner(args.run_dir)
    try:
        control.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        control.stop(0).wait()
        flights.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
