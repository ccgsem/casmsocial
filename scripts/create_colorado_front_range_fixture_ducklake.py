#!/usr/bin/env python3
"""Materialize a validated local Colorado Front Range runtime into DuckLake.

The Colorado builder writes identifier-bearing runtime products locally.  This
script is the final local-only handoff from that product to the DuckLake
catalog consumed by ``config/colorado_front_range_fixture.yaml``.  It never
creates population data and it deliberately refuses an incomplete or stale
runtime export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmsocial.datasets.colorado_front_range.sources import sha256_file
from casmsocial.ducklake_utils import get_ducklake_connection

TABLES = ("activities", "persons", "hh", "places", "social_networks")
SCHEMA_NAME = "colorado_front_range"
PARTITION_TABLE = "partitions.colorado_front_range_place_partitions"


def _runtime_inputs(runtime_dir: Path) -> tuple[dict[str, object], Path, dict[str, Path]]:
    """Return the verified CASMSocial table exports from a runtime product."""
    runtime_dir = runtime_dir.expanduser().resolve()
    manifest_path = runtime_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValueError("Colorado runtime manifest must have status 'passed'")

    input_dir = runtime_dir / "casmsocial"
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Colorado runtime manifest is missing output checksums")
    required = {name: input_dir / f"{name}.parquet" for name in TABLES}
    for name, path in required.items():
        output = outputs.get(f"casmsocial/{name}.parquet")
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(output, dict) or output.get("sha256") != sha256_file(path):
            raise ValueError(f"Colorado runtime table does not match its manifest: {name}")
    return manifest, input_dir, required


def materialize_fixture(
    runtime_dir: Path,
    ducklake_path: Path,
    partition_ranks: int | None = None,
) -> dict[str, int]:
    """Load a validated Colorado runtime export into a local DuckLake catalog."""
    _, _, required = _runtime_inputs(runtime_dir)
    ducklake_path = ducklake_path.expanduser()
    if partition_ranks is not None and partition_ranks < 1:
        raise ValueError("partition_ranks must be positive")
    ranks = partition_ranks or 1

    connection = get_ducklake_connection(ducklake_path)
    try:
        connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}"')
        row_counts: dict[str, int] = {}
        for name, path in required.items():
            connection.execute(
                f'CREATE OR REPLACE TABLE "{SCHEMA_NAME}"."{name}" AS SELECT * FROM read_parquet(?)',
                [str(path)],
            )
            row_counts[name] = connection.execute(
                f'SELECT count(*) FROM "{SCHEMA_NAME}"."{name}"'
            ).fetchone()[0]
        connection.execute("CREATE SCHEMA IF NOT EXISTS partitions")
        connection.execute(
            f"CREATE OR REPLACE TABLE {PARTITION_TABLE} AS "
            f"SELECT 1::INTEGER AS imputation, {ranks}::INTEGER AS n_ranks, "
            f"CAST(hash(sp_id) % {ranks} AS INTEGER) AS rank, sp_id::BIGINT AS place_id "
            f'FROM "{SCHEMA_NAME}"."places"'
        )
        row_counts["place_partitions"] = connection.execute(
            f"SELECT count(*) FROM {PARTITION_TABLE}"
        ).fetchone()[0]
    finally:
        connection.close()
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="Local output directory from the Colorado profile runtime builder.",
    )
    parser.add_argument("--ducklake-path", type=Path, required=True)
    parser.add_argument("--partition-ranks", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(materialize_fixture(args.runtime_dir, args.ducklake_path, args.partition_ranks), sort_keys=True))


if __name__ == "__main__":
    main()
