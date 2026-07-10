"""Materialize the Wake County Heat test fixture into DuckLake."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import yaml

from casmsocial.ducklake_utils import get_ducklake_connection

DEFAULT_FIXTURE_PATH = Path("testdata/wake_county_heat_1000_households")
DEFAULT_DUCKLAKE_PATH = Path("data/datalakehouse")
DEFAULT_DATABASE_NAME = "insights_ducklake"
DEFAULT_TARGET = "local"
DEFAULT_S3_SECRET_NAME = "casmsocial_ducklake_fixture_s3"

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TARGETS = ("local", "postgres-s3")


class WakeCountyHeatFixtureError(ValueError):
    """Raised when the Wake County Heat fixture cannot be loaded safely."""


@dataclass(frozen=True)
class FixtureTable:
    """A table entry from the Wake County Heat fixture manifest."""

    name: str
    source_table: str
    path: Path
    rows: int
    sha256: str
    columns: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FixtureManifest:
    """Validated Wake County Heat fixture metadata."""

    fixture_path: Path
    schema_name: str
    tables: tuple[FixtureTable, ...]


@dataclass(frozen=True)
class MaterializationResult:
    """Summary of a DuckLake materialization."""

    target: str
    ducklake_path: Path | None
    database_name: str
    schema_name: str
    tables: Mapping[str, int]
    catalog: str
    data_path: str


@dataclass(frozen=True)
class PostgresS3DuckLakeConfig:
    """Connection settings for a DuckLake catalog in Postgres and data in S3."""

    postgres_connection: str
    s3_data_path: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None
    s3_url_style: str = "path"
    s3_use_ssl: bool = False
    s3_secret_name: str = DEFAULT_S3_SECRET_NAME


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WakeCountyHeatFixtureError(message)


def _quote_identifier(identifier: str) -> str:
    _require(IDENTIFIER_PATTERN.match(identifier) is not None, f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _qualified_name(schema_name: str, table_name: str) -> str:
    return f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _redact_postgres_connection(postgres_connection: str) -> str:
    return re.sub(r"(password=)[^\s]+", r"\1<redacted>", postgres_connection)


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    _require(path.exists(), f"Manifest path does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), f"Manifest must be a YAML mapping: {path}")
    return cast(Mapping[str, Any], data)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value != "", f"{label} must be a non-empty string")
    return value


def _int(value: Any, label: str) -> int:
    _require(isinstance(value, int), f"{label} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_fixture_file(fixture_path: Path, relative_file: str) -> Path:
    path = (fixture_path / relative_file).resolve()
    fixture_root = fixture_path.resolve()
    _require(path.is_relative_to(fixture_root), f"Fixture table path escapes fixture directory: {relative_file}")
    _require(path.exists(), f"Fixture table file does not exist: {path}")
    _require(path.is_file(), f"Fixture table path is not a file: {path}")
    return path


def _column_signature_from_manifest(table_entry: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    columns = table_entry.get("columns")
    _require(isinstance(columns, list) and len(columns) > 0, f"{table_entry.get('name')} columns must be a list")
    signature = []
    for index, column in enumerate(columns):
        column_entry = _mapping(column, f"{table_entry.get('name')} columns[{index}]")
        signature.append((
            _string(column_entry.get("name"), f"{table_entry.get('name')} columns[{index}].name"),
            _string(column_entry.get("type"), f"{table_entry.get('name')} columns[{index}].type"),
        ))
    return tuple(signature)


def _describe_parquet(conn: duckdb.DuckDBPyConnection, path: Path) -> tuple[tuple[str, str], ...]:
    rows = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _describe_table(conn: duckdb.DuckDBPyConnection, qualified_name: str) -> tuple[tuple[str, str], ...]:
    rows = conn.execute(f"DESCRIBE SELECT * FROM {qualified_name}").fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _parquet_row_count(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    row = conn.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()
    return int(row[0]) if row is not None else 0


def _table_row_count(conn: duckdb.DuckDBPyConnection, qualified_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {qualified_name}").fetchone()
    return int(row[0]) if row is not None else 0


def _install_and_load_extensions(conn: duckdb.DuckDBPyConnection, extensions: tuple[str, ...]) -> None:
    for extension in extensions:
        conn.execute(f"INSTALL {extension}")
        conn.execute(f"LOAD {extension}")


def _create_s3_secret(conn: duckdb.DuckDBPyConnection, config: PostgresS3DuckLakeConfig) -> None:
    _quote_identifier(config.s3_secret_name)
    _require(config.s3_url_style in {"path", "vhost"}, "s3_url_style must be 'path' or 'vhost'")

    options = [
        "TYPE s3",
        "PROVIDER config",
        f"KEY_ID {_sql_literal(config.s3_access_key_id)}",
        f"SECRET {_sql_literal(config.s3_secret_access_key)}",
        f"REGION {_sql_literal(config.s3_region)}",
        f"URL_STYLE {_sql_literal(config.s3_url_style)}",
        f"USE_SSL {str(config.s3_use_ssl).lower()}",
    ]
    if config.s3_endpoint:
        options.append(f"ENDPOINT {_sql_literal(config.s3_endpoint)}")

    conn.execute(f"CREATE OR REPLACE SECRET {_quote_identifier(config.s3_secret_name)} ({', '.join(options)})")


def get_ducklake_postgres_s3_connection(
    config: PostgresS3DuckLakeConfig,
    database_name: str = DEFAULT_DATABASE_NAME,
) -> duckdb.DuckDBPyConnection:
    """Attach a DuckLake with a Postgres catalog and S3-compatible data path."""
    _quote_identifier(database_name)
    _require(config.postgres_connection.strip() != "", "postgres_connection must not be empty")
    _require(config.s3_data_path.startswith("s3://"), "s3_data_path must start with s3://")

    conn = duckdb.connect()
    _install_and_load_extensions(conn, ("postgres", "httpfs", "ducklake"))
    _create_s3_secret(conn, config)

    catalog = f"ducklake:postgres:{config.postgres_connection}"
    conn.execute(
        f"""
        ATTACH {_sql_literal(catalog)} AS {_quote_identifier(database_name)}
            (DATA_PATH {_sql_literal(config.s3_data_path)}, OVERRIDE_DATA_PATH true, AUTOMATIC_MIGRATION true);
        USE {_quote_identifier(database_name)};
        """
    )
    return conn


def read_fixture_manifest(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> FixtureManifest:
    """Read and structurally validate the fixture manifest."""
    fixture_path = fixture_path.expanduser()
    manifest = _read_yaml_mapping(fixture_path / "manifest.yaml")
    source = _mapping(manifest.get("source"), "source")
    schema_name = _string(source.get("schema"), "source.schema")
    _quote_identifier(schema_name)

    table_entries = manifest.get("tables")
    _require(isinstance(table_entries, list) and len(table_entries) > 0, "tables must be a non-empty list")

    tables = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(table_entries):
        entry = _mapping(raw_entry, f"tables[{index}]")
        name = _string(entry.get("name"), f"tables[{index}].name")
        _quote_identifier(name)
        _require(name not in seen_names, f"Duplicate table entry: {name}")
        seen_names.add(name)

        relative_file = _string(entry.get("file"), f"tables[{index}].file")
        tables.append(
            FixtureTable(
                name=name,
                source_table=_string(entry.get("source_table"), f"tables[{index}].source_table"),
                path=_safe_fixture_file(fixture_path, relative_file),
                rows=_int(entry.get("rows"), f"tables[{index}].rows"),
                sha256=_string(entry.get("sha256"), f"tables[{index}].sha256"),
                columns=_column_signature_from_manifest(entry),
            )
        )

    return FixtureManifest(
        fixture_path=fixture_path,
        schema_name=schema_name,
        tables=tuple(tables),
    )


def validate_fixture(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> FixtureManifest:
    """Validate fixture checksums, row counts, and DuckDB-visible schemas."""
    fixture = read_fixture_manifest(fixture_path)
    conn = duckdb.connect(":memory:")
    try:
        for table in fixture.tables:
            actual_sha256 = _sha256(table.path)
            _require(
                actual_sha256 == table.sha256,
                f"{table.name} checksum mismatch: {actual_sha256} != {table.sha256}",
            )
            actual_rows = _parquet_row_count(conn, table.path)
            _require(actual_rows == table.rows, f"{table.name} row mismatch: {actual_rows} != {table.rows}")
            actual_columns = _describe_parquet(conn, table.path)
            _require(actual_columns == table.columns, f"{table.name} schema mismatch")
    finally:
        conn.close()
    return fixture


def materialize_fixture(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    database_name: str = DEFAULT_DATABASE_NAME,
) -> MaterializationResult:
    """Create or replace the fixture tables in a local DuckLake."""
    fixture = validate_fixture(fixture_path)
    ducklake_path = ducklake_path.expanduser()
    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    catalog_path = f"ducklake:sqlite:{ducklake_path / 'metadata.sqlite'}"
    data_path = "file://" + str(ducklake_path / "storage")
    loaded_counts = _load_fixture_tables(conn, fixture)

    return MaterializationResult(
        target="local",
        ducklake_path=ducklake_path,
        database_name=database_name,
        schema_name=fixture.schema_name,
        tables=loaded_counts,
        catalog=catalog_path,
        data_path=data_path,
    )


def materialize_postgres_s3_fixture(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    config: PostgresS3DuckLakeConfig | None = None,
    database_name: str = DEFAULT_DATABASE_NAME,
) -> MaterializationResult:
    """Create or replace the fixture tables in DuckLake backed by Postgres and S3."""
    _require(config is not None, "postgres-s3 target requires a Postgres/S3 configuration")
    fixture = validate_fixture(fixture_path)
    conn = get_ducklake_postgres_s3_connection(config, database_name=database_name)
    loaded_counts = _load_fixture_tables(conn, fixture)

    return MaterializationResult(
        target="postgres-s3",
        ducklake_path=None,
        database_name=database_name,
        schema_name=fixture.schema_name,
        tables=loaded_counts,
        catalog=f"ducklake:postgres:{_redact_postgres_connection(config.postgres_connection)}",
        data_path=config.s3_data_path,
    )


def _load_fixture_tables(conn: duckdb.DuckDBPyConnection, fixture: FixtureManifest) -> dict[str, int]:
    loaded_counts: dict[str, int] = {}
    try:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(fixture.schema_name)}")
        for table in fixture.tables:
            qualified_name = _qualified_name(fixture.schema_name, table.name)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE {qualified_name} AS
                SELECT *
                FROM read_parquet(?)
                """,
                [str(table.path)],
            )
            loaded_rows = _table_row_count(conn, qualified_name)
            _require(loaded_rows == table.rows, f"{table.name} loaded row mismatch: {loaded_rows} != {table.rows}")
            loaded_columns = _describe_table(conn, qualified_name)
            _require(loaded_columns == table.columns, f"{table.name} loaded schema mismatch")
            loaded_counts[table.name] = loaded_rows
    finally:
        conn.close()
    return loaded_counts


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _postgres_s3_config_from_args(args: argparse.Namespace) -> PostgresS3DuckLakeConfig:
    postgres_connection = args.postgres_connection or os.environ.get("CASMSOCIAL_DUCKLAKE_POSTGRES_CONNECTION")
    s3_data_path = args.s3_data_path or os.environ.get("CASMSOCIAL_DUCKLAKE_S3_DATA_PATH")
    s3_access_key_id = args.s3_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_access_key = args.s3_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")

    _require(
        postgres_connection is not None and postgres_connection.strip() != "",
        "--postgres-connection or CASMSOCIAL_DUCKLAKE_POSTGRES_CONNECTION is required for postgres-s3",
    )
    _require(
        s3_data_path is not None and s3_data_path.strip() != "",
        "--s3-data-path or CASMSOCIAL_DUCKLAKE_S3_DATA_PATH is required for postgres-s3",
    )
    _require(
        s3_access_key_id is not None and s3_access_key_id.strip() != "",
        "--s3-access-key-id or AWS_ACCESS_KEY_ID is required for postgres-s3",
    )
    _require(
        s3_secret_access_key is not None and s3_secret_access_key.strip() != "",
        "--s3-secret-access-key or AWS_SECRET_ACCESS_KEY is required for postgres-s3",
    )

    return PostgresS3DuckLakeConfig(
        postgres_connection=postgres_connection,
        s3_data_path=s3_data_path,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_region=args.s3_region or os.environ.get("AWS_REGION", "us-east-1"),
        s3_endpoint=args.s3_endpoint or os.environ.get("CASMSOCIAL_DUCKLAKE_S3_ENDPOINT"),
        s3_url_style=args.s3_url_style or os.environ.get("CASMSOCIAL_DUCKLAKE_S3_URL_STYLE", "path"),
        s3_use_ssl=args.s3_use_ssl
        if args.s3_use_ssl is not None
        else _env_bool("CASMSOCIAL_DUCKLAKE_S3_USE_SSL", False),
        s3_secret_name=args.s3_secret_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default=DEFAULT_TARGET,
        help="DuckLake deployment target.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Fixture directory containing manifest.yaml and table Parquet files.",
    )
    parser.add_argument(
        "--ducklake-path",
        type=Path,
        default=DEFAULT_DUCKLAKE_PATH,
        help="Local DuckLake directory to create or update.",
    )
    parser.add_argument(
        "--database-name",
        default=DEFAULT_DATABASE_NAME,
        help="DuckLake database name to attach.",
    )
    parser.add_argument(
        "--postgres-connection",
        help=(
            "Postgres connection string for the DuckLake catalog, for example "
            "'host=ducklake-postgres port=5432 dbname=casmsocial_ducklake user=casmsocial password=casmsocial'."
        ),
    )
    parser.add_argument(
        "--s3-data-path",
        help="S3 URI for DuckLake data files, for example s3://casmsocial-ducklake/wake_county_heat.",
    )
    parser.add_argument("--s3-access-key-id", help="S3 access key ID. Defaults to AWS_ACCESS_KEY_ID.")
    parser.add_argument("--s3-secret-access-key", help="S3 secret access key. Defaults to AWS_SECRET_ACCESS_KEY.")
    parser.add_argument("--s3-region", help="S3 region. Defaults to AWS_REGION or us-east-1.")
    parser.add_argument(
        "--s3-endpoint",
        help="S3-compatible endpoint host[:port], for example ducklake-minio:9000.",
    )
    parser.add_argument(
        "--s3-url-style",
        choices=("path", "vhost"),
        help="S3 URL style. Defaults to CASMSOCIAL_DUCKLAKE_S3_URL_STYLE or path.",
    )
    parser.add_argument(
        "--s3-use-ssl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use HTTPS for S3-compatible storage. Defaults to CASMSOCIAL_DUCKLAKE_S3_USE_SSL or false.",
    )
    parser.add_argument(
        "--s3-secret-name",
        default=DEFAULT_S3_SECRET_NAME,
        help="DuckDB secret name used for S3 credentials.",
    )
    args = parser.parse_args()

    try:
        if args.target == "local":
            result = materialize_fixture(
                fixture_path=args.fixture_path,
                ducklake_path=args.ducklake_path,
                database_name=args.database_name,
            )
        else:
            result = materialize_postgres_s3_fixture(
                fixture_path=args.fixture_path,
                config=_postgres_s3_config_from_args(args),
                database_name=args.database_name,
            )
    except WakeCountyHeatFixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    table_counts = ", ".join(f"{name}={rows}" for name, rows in result.tables.items())
    print(
        "Materialized Wake County Heat fixture: "
        f"target={result.target} catalog={result.catalog} data_path={result.data_path} "
        f"schema={result.schema_name} ({table_counts})"
    )


if __name__ == "__main__":
    main()
