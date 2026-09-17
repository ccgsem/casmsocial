"""CASMSocial launcher for the standalone CASMSim gRPC/Flight runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmsim.flight_server import start_broker_flight_server
from casmsim.grpc_runner import ENDPOINT_FILENAME, run_submitted_model, start_control_server
from casmsim.observation_broker import ObservationBroker

_CASMPOP_ENTRY_POINT = "casmsocial.adapters.runner:CasmPopAdapter"


def run_casmsocial_model(run_id: str, config_json: bytes, broker: ObservationBroker) -> None:
    """Launch a CASMSocial model through CASMSim's generic adapter protocol."""
    params = json.loads(config_json)
    if ("model.plugins" in params or "model.name" in params) and "runner.entry_point" not in params:
        params["runner.entry_point"] = _CASMPOP_ENTRY_POINT
        config_json = json.dumps(params).encode()
    run_submitted_model(run_id, config_json, broker)


def start_runner(run_dir: Path):
    """Start loopback-only control and Arrow Flight endpoints."""
    broker = ObservationBroker()
    control = start_control_server(
        run_dir,
        broker,
        lambda run_id, config_json: run_casmsocial_model(run_id, config_json, broker),
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
