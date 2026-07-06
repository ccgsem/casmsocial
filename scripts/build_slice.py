"""Regenerate Python Ice stubs from the slice IDL files.

Usage::

    uv run python scripts/build_slice.py

Stubs are written to the repo root so that ``import arrowservice`` resolves
from any working directory inside the project.

Run this whenever ``ArrowService.ice`` changes. Keep it byte-identical to
casmservice's copy of the same file (casmsocial and casmservice each keep
their own copy of the IDL -- there is no shared package between the two
repos, only the wire contract). The generated ``arrowservice/`` package is
committed to the repository so that importing casmsocial does not require
``slice2py`` to be installed at runtime -- only the optional ``service``
extra (``zeroc-ice``) is needed to actually start the server.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SLICE_DIR = REPO_ROOT / "slice"

# (slice_file, output_package_dir)
# slice2py writes <module>/<module>/*.py, so output to REPO_ROOT so that
# `import arrowservice` resolves from the project root.
TARGETS = [
    (SLICE_DIR / "arrowservice" / "ArrowService.ice", REPO_ROOT),
]


def find_slice2py() -> str:
    tool = shutil.which("slice2py")
    if tool is None:
        print(
            "ERROR: slice2py not found on PATH.\n"
            "Install ZeroC Ice tools: brew install zeroc-ice (macOS) "
            "or apt-get install zeroc-ice-all-dev (Ubuntu).",
            file=sys.stderr,
        )
        sys.exit(1)
    return tool


def build(slice2py: str, ice_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        slice2py,
        f"-I{SLICE_DIR}",
        "--output-dir", str(out_dir),
        "--build", "all",
        str(ice_file),
    ]
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    if result.stdout:
        print(result.stdout)


def main() -> None:
    slice2py = find_slice2py()
    print(f"Using {slice2py}")
    for ice_file, out_dir in TARGETS:
        print(f"\nBuilding {ice_file.name} → {out_dir.relative_to(REPO_ROOT)}/")
        build(slice2py, ice_file, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
