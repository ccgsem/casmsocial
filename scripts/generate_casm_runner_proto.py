#!/usr/bin/env python3
"""Generate checked-in Python stubs for the CASM runner protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTO_DIRECTORY = REPOSITORY_ROOT / "proto" / "casm_runner" / "v1"
PROTO_SOURCE = PROTO_DIRECTORY / "casm_runner.proto"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "casmsocial" / "casmsim" / "proto"


def generate(output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> None:
    """Generate protobuf and gRPC Python modules into ``output_directory``."""
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={PROTO_DIRECTORY}",
            f"--python_out={output_directory}",
            f"--grpc_python_out={output_directory}",
            str(PROTO_SOURCE),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )

    protobuf_module = output_directory / "casm_runner_pb2.py"
    protobuf_module.write_text(
        protobuf_module.read_text().replace(
            "# NO CHECKED-IN PROTOBUF GENCODE\n",
            "# Checked-in generated code; regenerate with scripts/generate_casm_runner_proto.py.\n",
        )
    )
    grpc_module = output_directory / "casm_runner_pb2_grpc.py"
    grpc_module.write_text(
        grpc_module.read_text().replace(
            "import casm_runner_pb2 as casm__runner__pb2",
            "from casmsocial.casmsim.proto import casm_runner_pb2 as casm__runner__pb2",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    generate(args.output_directory)


if __name__ == "__main__":
    main()
