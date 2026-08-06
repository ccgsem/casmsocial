#!/usr/bin/env python3
"""Materialize a local Colorado Front Range fixture into a DuckLake catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from casmsocial.ducklake_utils import get_ducklake_connection

TABLES = ("activities", "persons", "hh", "places")
PARTITION_TABLE = "partitions.colorado_front_range_fixture_place_partitions"


def materialize_fixture(
    input_dir: Path,
    social_networks: Path,
    ducklake_path: Path,
    schema_name: str = "colorado_front_range_fixture",
    partition_ranks: int | None = None,
) -> dict[str, int]:
    """Load exported schedule inputs and endpoint-complete ties into DuckLake."""
    input_dir = input_dir.expanduser()
    social_networks = social_networks.expanduser()
    ducklake_path = ducklake_path.expanduser()
    required = {name: input_dir / f"{name}.parquet" for name in TABLES}
    missing = [str(path) for path in [*required.values(), social_networks] if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if partition_ranks is not None and partition_ranks < 1:
        raise ValueError("partition_ranks must be positive")

    connection = get_ducklake_connection(ducklake_path)
    try:
        connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
        row_counts: dict[str, int] = {}
        for name, path in required.items():
            connection.execute(
                f'CREATE OR REPLACE TABLE "{schema_name}"."{name}" AS SELECT * FROM read_parquet(?)',
                [str(path)],
            )
            row_counts[name] = connection.execute(
                f'SELECT count(*) FROM "{schema_name}"."{name}"'
            ).fetchone()[0]
        connection.execute(
            f'CREATE OR REPLACE TABLE "{schema_name}"."social_networks" AS SELECT * FROM read_parquet(?)',
            [str(social_networks)],
        )
        row_counts["social_networks"] = connection.execute(
            f'SELECT count(*) FROM "{schema_name}"."social_networks"'
        ).fetchone()[0]
        if partition_ranks:
            connection.execute("CREATE SCHEMA IF NOT EXISTS partitions")
            connection.execute(
                f"CREATE OR REPLACE TABLE {PARTITION_TABLE} AS "
                f"SELECT 1::INTEGER AS imputation, {partition_ranks}::INTEGER AS n_ranks, "
                f"CAST(hash(sp_id) % {partition_ranks} AS INTEGER) AS rank, sp_id::BIGINT AS place_id "
                f'FROM "{schema_name}"."places"'
            )
            row_counts["place_partitions"] = connection.execute(
                f"SELECT count(*) FROM {PARTITION_TABLE}"
            ).fetchone()[0]
    finally:
        connection.close()
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--social-networks", type=Path, required=True)
    parser.add_argument("--ducklake-path", type=Path, required=True)
    parser.add_argument("--schema-name", default="colorado_front_range_fixture")
    parser.add_argument("--partition-ranks", type=int)
    args = parser.parse_args()
    print(materialize_fixture(
        args.input_dir, args.social_networks, args.ducklake_path,
        args.schema_name, args.partition_ranks,
    ))


if __name__ == "__main__":
    main()
