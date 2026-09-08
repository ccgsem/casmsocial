"""Stream observation batches from a running casmsocial gRPC runner.

Connects to the control endpoint written by grpc_runner, calls StreamObs for
the requested channel, and prints a summary of each received Arrow IPC batch.

Usage::

    # Stream agent_log from a running 2-rank smoke test
    uv run python scripts/stream_obs.py \\
        --run-dir /tmp/runs/smoke-2rank \\
        --run-id smoke-2rank-01 \\
        --channel agent_log

    # Stream from tick 5 onward, print full schema
    uv run python scripts/stream_obs.py \\
        --run-dir /tmp/runs/smoke-2rank \\
        --run-id smoke-2rank-01 \\
        --channel agent_log \\
        --start-tick 5 \\
        --schema

Available channels (default observers):
    agent_log, social_interactions, schedule_occupancy, behavior_log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import grpc
import pyarrow as pa
from pyarrow import ipc

from casmsocial.grpc_control import ENDPOINT_FILENAME
from casmsocial.proto import casm_runner_pb2 as pb2, casm_runner_pb2_grpc as pb2_grpc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream observation batches from a casmsocial gRPC runner.")
    parser.add_argument("--run-dir", required=True, metavar="PATH")
    parser.add_argument("--run-id", required=True, metavar="RUN_ID")
    parser.add_argument("--channel", default="agent_log", metavar="CHANNEL",
                        help="Observation channel to stream (default: agent_log).")
    parser.add_argument("--start-tick", type=int, default=0, metavar="TICK",
                        help="First batch ID to receive (default: 0).")
    parser.add_argument("--schema", action="store_true",
                        help="Print Arrow schema on first batch.")
    parser.add_argument("--rows", type=int, default=0, metavar="N",
                        help="Print first N data rows of each batch (default: 0).")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    endpoint_file = run_dir / ENDPOINT_FILENAME
    if not endpoint_file.exists():
        print(f"ERROR: {endpoint_file} not found — is the runner started?", file=sys.stderr)
        return 1

    endpoints = json.loads(endpoint_file.read_text())
    address = endpoints["control"]["address"]

    print(f"Connecting to {address} (channel={args.channel!r}, start_tick={args.start_tick}) …")
    total_batches = 0
    total_rows = 0

    with grpc.insecure_channel(address) as channel:
        stub = pb2_grpc.SimulatorControlStub(channel)
        request = pb2.StreamObsRequest(
            run_id=args.run_id,
            channel=args.channel,
            start_tick=args.start_tick,
        )
        try:
            for obs in stub.StreamObs(request):
                buf = pa.BufferReader(obs.arrow_ipc)
                with ipc.open_stream(buf) as reader:
                    table = reader.read_all()

                if total_batches == 0 and args.schema:
                    print(f"\nSchema:\n{table.schema}\n")

                total_batches += 1
                total_rows += len(table)
                print(f"  batch tick={obs.tick:4d}  rows={len(table):6,}  cumulative_rows={total_rows:8,}")

                if args.rows > 0:
                    import polars as pl
                    df = pl.from_arrow(table).head(args.rows)
                    print(df)

        except grpc.RpcError as exc:
            print(f"gRPC error: {exc.code()} — {exc.details()}", file=sys.stderr)
            return 1

    print(f"\nDone. Received {total_batches} batches, {total_rows:,} total rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
