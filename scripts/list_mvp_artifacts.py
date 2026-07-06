"""List generated MVP artifact paths for local review and CI upload."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts.write_mvp_manifest import mvp_artifact_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--without-manifest",
        action="store_true",
        help="Exclude output/mvp_manifest.json from the listed paths.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for artifact_path in mvp_artifact_paths(include_manifest=not args.without_manifest):
        print(artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
