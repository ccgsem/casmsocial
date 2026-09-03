"""Loopback runner hosting CASMSocial gRPC control and Arrow Flight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from casmsocial.__main__ import load_builtin_models
from casmsocial.factory import Models
from casmsocial.flight_broker import start_broker_flight_server
from casmsocial.grpc_control import ENDPOINT_FILENAME, start_control_server
from casmsocial.observation_broker import ObservationBroker
from casmsocial.repast_observation_broker import RepastObservationBrokerAdapter


def run_submitted_model(run_id: str, config_json: bytes, broker: ObservationBroker) -> None:
    """Run one submitted model and publish its model-owned observer tables."""
    params = json.loads(config_json)
    if not isinstance(params, dict):
        raise ValueError("config_json must encode a JSON object of model parameters")
    params = dict(params)
    params["simulation.run_id"] = run_id
    # The broker-backed Flight server is the live observation transport.
    params["observers.arrow_server.enabled"] = False

    load_builtin_models()
    model = Models.create_model(params["model.name"])(MPI.COMM_WORLD, params)
    model.add_observer(RepastObservationBrokerAdapter(broker))
    model.start()


def start_runner(run_dir: Path):
    """Start loopback-only endpoints and return their server handles."""
    broker = ObservationBroker()
    control = start_control_server(
        run_dir,
        broker,
        lambda run_id, config_json: run_submitted_model(run_id, config_json, broker),
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
