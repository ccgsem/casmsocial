"""Create the tiny DuckLake dataset used by the MVP example."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from casmsocial.ducklake_utils import get_ducklake_connection

DEFAULT_DUCKLAKE_PATH = Path("examples/mvp/mvp.ducklake")


def create_mvp_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create or replace the self-contained MVP tables on an open DuckDB connection."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS casmsocial_mvp")
    conn.execute("CREATE SCHEMA IF NOT EXISTS partitions")
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.places (
            sp_id BIGINT,
            rank INTEGER,
            place_type VARCHAR,
            place_name VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.persons (
            sp_id BIGINT,
            sp_hh_id BIGINT,
            sp_work_id BIGINT,
            sp_school_id BIGINT
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.hh (
            sp_id BIGINT,
            hh_size INTEGER,
            hh_income DOUBLE,
            hh_type VARCHAR,
            hh_race VARCHAR,
            hh_age INTEGER
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.activities (
            sp_persons_id BIGINT,
            activity_id INTEGER,
            activity_sequence INTEGER,
            starttime_min INTEGER,
            endtime_min INTEGER,
            sp_act_id BIGINT
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.road_nodes (
            node_id BIGINT,
            x DOUBLE,
            y DOUBLE
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.road_edges (
            edge_id BIGINT,
            from_node_id BIGINT,
            to_node_id BIGINT,
            length_m DOUBLE,
            travel_time_min DOUBLE,
            mode VARCHAR,
            road_type VARCHAR
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE casmsocial_mvp.place_road_snap (
            place_id BIGINT,
            road_node_id BIGINT
        )
        """)
    conn.execute("""
        CREATE OR REPLACE TABLE partitions.mvp_two_rank_place_partitions (
            imputation INTEGER,
            n_ranks INTEGER,
            rank INTEGER,
            place_id BIGINT
        )
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.places VALUES
            (100, 0, 'Household', 'home-a', 38.90, -77.04),
            (200, 0, 'Household', 'home-b', 38.91, -77.05),
            (300, 0, 'Workplace', 'work', 38.92, -77.06)
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.persons VALUES
            (1, 100, 300, NULL),
            (2, 200, 300, NULL)
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.hh VALUES
            (100, 1, 75000.0, 'single', 'unknown', 35),
            (200, 1, 82000.0, 'single', 'unknown', 41)
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.activities VALUES
            (1, 0, 0, 0, 480, 100),
            (1, 1, 1, 540, 1020, 300),
            (1, 0, 2, 1080, 1439, 100),
            (2, 0, 0, 0, 480, 200),
            (2, 1, 1, 540, 1020, 300),
            (2, 0, 2, 1080, 1439, 200)
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.road_nodes VALUES
            (1, -77.04, 38.90),
            (2, -77.05, 38.91),
            (3, -77.06, 38.92)
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.road_edges VALUES
            (1, 1, 3, 5000.0, 12.0, 'drive', 'local'),
            (2, 3, 1, 5000.0, 12.0, 'drive', 'local'),
            (3, 2, 3, 4200.0, 10.0, 'drive', 'local'),
            (4, 3, 2, 4200.0, 10.0, 'drive', 'local'),
            (5, 1, 2, 1500.0, 4.0, 'drive', 'local'),
            (6, 2, 1, 1500.0, 4.0, 'drive', 'local')
        """)
    conn.execute("""
        INSERT INTO casmsocial_mvp.place_road_snap VALUES
            (100, 1),
            (200, 2),
            (300, 3)
        """)
    conn.execute("""
        INSERT INTO partitions.mvp_two_rank_place_partitions VALUES
            (1, 2, 0, 100),
            (1, 2, 1, 200),
            (1, 2, 0, 300)
        """)


def create_mvp_ducklake(
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    database_name: str = "insights_ducklake",
) -> None:
    """Create or replace the sample DMV tables consumed by config/mvp.yaml."""
    ducklake_path = ducklake_path.expanduser()
    ducklake_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    try:
        create_mvp_tables(conn)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ducklake-path",
        type=Path,
        default=DEFAULT_DUCKLAKE_PATH,
        help="Directory for the DuckLake metadata and storage files.",
    )
    parser.add_argument(
        "--database-name",
        default="insights_ducklake",
        help="DuckLake catalog name to attach.",
    )
    args = parser.parse_args()

    create_mvp_ducklake(args.ducklake_path, database_name=args.database_name)
    print(f"Created MVP DuckLake at {args.ducklake_path}")


if __name__ == "__main__":
    main()
