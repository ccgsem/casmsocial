"""
DuckLake Utility Functions

Common utilities for working with DuckLake databases.
"""

import pathlib

import duckdb


def add_ducklake_insights_secrets(conn: duckdb.DuckDBPyConnection, ducklake_path: pathlib.Path) -> None:
    """Add DuckLake insights secrets to the DuckDB connection.
    Args:
        conn (duckdb.DuckDBPyConnection): DuckDB connection object.
    """
    ducklake_path.mkdir(parents=True, exist_ok=True)
    # catalog_path = "".join(["ducklake:sqlite:", str(ducklake_path / "metadata.sqlite")])
    # database_name = "insights_ducklake"
    # data_url = "".join(["file://", str(ducklake_path / "storage")])
    conn.execute(
        """
    CREATE SECRET IF NOT EXISTS insights_ducklake_secret (
        TYPE ducklake,
        METADATA_PATH )
        VALUES ('file://{}/metadata.sqlite');"""
    )


def get_ducklake_connection(ducklake_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    """Get a DuckLake connection using a context manager.
    Args:
        ducklake_path (pathlib.Path): Path to the DuckLake database directory.
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection object.
    """
    # change to ducklake_path
    ducklake_path.mkdir(exist_ok=True)
    catalog_path = "".join(["ducklake:sqlite:", str(ducklake_path / "metadata.sqlite")])
    data_url = "".join(["file://", str(ducklake_path / "storage")])
    database_name = "insights_ducklake"

    # create duckdb connection
    conn = duckdb.connect()
    conn.execute(
        """
    INSTALL sqlite;
    INSTALL ducklake;
    LOAD ducklake;
    INSTALL airport FROM community;
    LOAD airport;
    INSTALL spatial;
    LOAD spatial;
    """
    )

    # Attach datalake
    query_string = f"""
    ATTACH '{catalog_path}' AS {database_name}
        (DATA_PATH '{data_url}', OVERRIDE_DATA_PATH true);
    USE {database_name};
    """

    conn.execute(query_string)

    return conn
