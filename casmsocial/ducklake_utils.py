"""
DuckLake Utility Functions

Common utilities for working with DuckLake databases.
"""

import pathlib

import duckdb


def get_ducklake_connection(
    ducklake_path: pathlib.Path, database_name: str = "insights_ducklake"
) -> duckdb.DuckDBPyConnection:
    """Get a DuckLake connection using a context manager.
    Args:
        ducklake_path (pathlib.Path): Path to the DuckLake database directory.
        database_name (str): Name of the DuckLake database to attach.
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection object.
    """
    # change to ducklake_path
    ducklake_path.mkdir(exist_ok=True)
    catalog_path = "".join(["ducklake:sqlite:", str(ducklake_path / "metadata.sqlite")])
    data_url = "".join(["file://", str(ducklake_path / "storage")])

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
