"""Send a Start RPC to a running casmsocial gRPC runner.

Reads the control endpoint from runner_endpoints.json in the run directory.

Usage::

    uv run python scripts/send_start.py --run-dir /tmp/runs/smoke-2rank \\
        --scenario wake_county_heat --run-id smoke-01

    # Override individual params
    uv run python scripts/send_start.py --run-dir /tmp/runs/smoke-2rank \\
        --scenario wake_county_heat --run-id smoke-01 \\
        --param stop.at=24
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import grpc

from casmsocial.grpc_control import ENDPOINT_FILENAME
from casmsocial.proto import casm_runner_pb2 as pb2, casm_runner_pb2_grpc as pb2_grpc

# ---------------------------------------------------------------------------
# Built-in scenario parameter sets (for quick smoke-testing without casmdb)
# ---------------------------------------------------------------------------

BUILTIN_SCENARIOS: dict[str, dict] = {
    "wake_county_heat": {
        "model.name": "casmsocial.citysim.citysocialmodel.CitySocialModel",
        "places.table": "wake_county_heat.places",
        "households.table": "wake_county_heat.hh_1000_households",
        "persons.table": "wake_county_heat.persons_1000_households",
        "activities.table": "wake_county_heat.activities_1000_households",
        "contacts.enabled": False,
        "communication.enabled": False,
    },
}


def _parse_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(f"--param must be key=value, got: {raw!r}")
    key, _, val = raw.partition("=")
    for coerce in (int, float):
        try:
            return key.strip(), coerce(val)
        except ValueError:
            pass
    if val.lower() in ("true", "yes"):
        return key.strip(), True
    if val.lower() in ("false", "no"):
        return key.strip(), False
    return key.strip(), val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send a Start RPC to a running casmsocial gRPC runner.")
    parser.add_argument("--run-dir", required=True, metavar="PATH", help="Run directory containing runner_endpoints.json.")
    parser.add_argument("--scenario", default="wake_county_heat", choices=list(BUILTIN_SCENARIOS), help="Built-in scenario to use.")
    parser.add_argument("--run-id", default=None, help="Run ID (default: random UUID).")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", dest="params")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    endpoint_file = run_dir / ENDPOINT_FILENAME
    if not endpoint_file.exists():
        print(f"ERROR: {endpoint_file} not found — is the runner started?", file=sys.stderr)
        return 1

    endpoints = json.loads(endpoint_file.read_text())
    address = endpoints["control"]["address"]

    params = dict(BUILTIN_SCENARIOS[args.scenario])
    for raw in args.params:
        k, v = _parse_override(raw)
        params[k] = v

    run_id = args.run_id or str(uuid.uuid4())

    print(f"Connecting to {address} …")
    with grpc.insecure_channel(address) as channel:
        stub = pb2_grpc.SimulatorControlStub(channel)
        resp = stub.Start(pb2.StartRequest(
            run_id=run_id,
            config_json=json.dumps(params).encode(),
        ))
    print(f"Started run_id={resp.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
