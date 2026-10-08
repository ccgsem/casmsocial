"""
DuckLake Utility Functions

Common utilities for working with DuckLake databases.

A DuckLake has two parts: a *catalog* (metadata tables in a SQL database) and
a *data path* (Parquet files). By default casmsocial keeps both under one
directory: ``<ducklake_path>/metadata.sqlite`` and ``<ducklake_path>/storage``.

The catalog can instead live elsewhere by passing a full DuckLake catalog URI,
for example a DuckDB catalog served over DuckDB's Quack protocol, which lets
several processes commit to the same lake concurrently::

    CASMSOCIAL_DUCKLAKE_URI=ducklake:quack:lakehost:9494
    CASMSOCIAL_DUCKLAKE_QUACK_TOKEN=...

The Parquet files still live under ``<ducklake_path>/storage``; every process
that attaches the lake must be able to read (and, to write, create) files there.

Quack clients use plain HTTP only for local servers (``localhost``,
``127.0.0.1``, ``::1``) and HTTPS otherwise, so a remote catalog server needs a
TLS-terminating proxy in front of it. Quack is a beta DuckDB extension
(DuckDB >= 1.5.3).
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping

import duckdb

CATALOG_URI_ENV = "CASMSOCIAL_DUCKLAKE_URI"
QUACK_TOKEN_ENV = "CASMSOCIAL_DUCKLAKE_QUACK_TOKEN"


class DuckLakeCatalogError(ValueError):
    """Raised for an unusable DuckLake catalog URI or its credentials."""


def ducklake_catalog_uri_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """Return ``$CASMSOCIAL_DUCKLAKE_URI``, or ``None`` when unset or blank."""
    env = os.environ if environ is None else environ
    value = env.get(CATALOG_URI_ENV, "").strip()
    return value or None


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_quack(catalog_uri: str) -> bool:
    return catalog_uri.startswith("ducklake:quack:")


def resolve_catalog_uri(ducklake_path: pathlib.Path, catalog_uri: str | None = None) -> str:
    """Return the DuckLake catalog URI to attach.

    Without ``catalog_uri`` this is the default SQLite catalog inside
    ``ducklake_path``. An explicit URI must use the ``ducklake:`` scheme, e.g.
    ``ducklake:sqlite:/data/lake/metadata.sqlite``, ``ducklake:/data/catalog.duckdb``,
    ``ducklake:postgres:dbname=lake`` or ``ducklake:quack:host:9494``.
    """
    if catalog_uri is None:
        return "ducklake:sqlite:" + str(ducklake_path / "metadata.sqlite")
    catalog_uri = catalog_uri.strip()
    if not catalog_uri.startswith("ducklake:") or catalog_uri == "ducklake:":
        msg = f"DuckLake catalog URI must start with 'ducklake:', got {catalog_uri!r}"
        raise DuckLakeCatalogError(msg)
    return catalog_uri


def _quack_secret_sql(catalog_uri: str, environ: Mapping[str, str]) -> str:
    token = environ.get(QUACK_TOKEN_ENV, "")
    if not token:
        msg = f"{QUACK_TOKEN_ENV} must be set to attach a Quack-served DuckLake catalog ({catalog_uri})"
        raise DuckLakeCatalogError(msg)
    server = catalog_uri.removeprefix("ducklake:")
    return (
        "CREATE OR REPLACE SECRET casmsocial_ducklake_quack "
        f"(TYPE quack, TOKEN {_sql_string(token)}, SCOPE {_sql_string(server)});"
    )


def get_ducklake_connection(
    ducklake_path: pathlib.Path,
    database_name: str = "insights_ducklake",
    *,
    catalog_uri: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with the DuckLake attached and selected.

    Args:
        ducklake_path (pathlib.Path): DuckLake directory; its ``storage``
            subdirectory is the Parquet data path, and it holds the default
            SQLite catalog.
        database_name (str): Name of the DuckLake database to attach.
        catalog_uri (str | None): Full DuckLake catalog URI overriding the
            default SQLite catalog (see :func:`resolve_catalog_uri`). A
            ``ducklake:quack:`` catalog reads its token from
            ``$CASMSOCIAL_DUCKLAKE_QUACK_TOKEN``.
        environ (Mapping[str, str] | None): Environment to read Quack settings
            from (defaults to ``os.environ``).
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection object.
    """
    env = os.environ if environ is None else environ
    ducklake_path = ducklake_path.expanduser().resolve()
    ducklake_path.mkdir(parents=True, exist_ok=True)
    (ducklake_path / "storage").mkdir(exist_ok=True)
    catalog_path = resolve_catalog_uri(ducklake_path, catalog_uri)
    data_url = "".join(["file://", str(ducklake_path / "storage")])
    secret_sql = _quack_secret_sql(catalog_path, env) if _is_quack(catalog_path) else ""

    # create duckdb connection
    conn = duckdb.connect()
    try:
        conn.execute("""
        INSTALL sqlite;
        INSTALL ducklake;
        LOAD ducklake;
        INSTALL spatial;
        LOAD spatial;
        """)
        if secret_sql:
            conn.execute(secret_sql)

        # Attach datalake
        query_string = f"""
        ATTACH {_sql_string(catalog_path)} AS {database_name}
            (DATA_PATH {_sql_string(data_url)}, OVERRIDE_DATA_PATH true, AUTOMATIC_MIGRATION true);
        USE {database_name};
        """

        conn.execute(query_string)
    except BaseException:
        conn.close()
        raise

    return conn
