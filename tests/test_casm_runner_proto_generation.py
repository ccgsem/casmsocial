"""Keep checked-in runner stubs synchronized with their .proto source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY_ROOT / "scripts" / "generate_casm_runner_proto.py"
CHECKED_IN_OUTPUT = REPOSITORY_ROOT / "casmsocial" / "casmsim" / "proto"


def test_checked_in_stubs_match_proto_source(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-directory", str(tmp_path)],
        check=True,
        cwd=REPOSITORY_ROOT,
    )

    for filename in ("casm_runner_pb2.py", "casm_runner_pb2_grpc.py"):
        assert (tmp_path / filename).read_text() == (CHECKED_IN_OUTPUT / filename).read_text()
