"""Verify the Colorado dataset builder's release-review state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from casmsocial.datasets.colorado_front_range.release_readiness import evaluate_release_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestations", type=Path)
    parser.add_argument(
        "--expect-status",
        choices=("review_required", "ready", "machine_control_failed"),
        default="review_required",
    )
    args = parser.parse_args()
    try:
        attestations = yaml.safe_load(args.attestations.read_text(encoding="utf-8")) if args.attestations else None
        result = evaluate_release_readiness(attestations)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"Release-readiness verification failed: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != args.expect_status:
        print(f"Expected status {args.expect_status!r}, found {result['status']!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
