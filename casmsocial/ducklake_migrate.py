"""Move a DuckLake's catalog from SQLite to a DuckDB file (e.g. to serve it over Quack).

A DuckLake keeps all of its metadata (schemas, tables, snapshots, the list of
Parquet files and their statistics) in ``ducklake_*`` tables whose layout is the
same in every catalog database. Moving the catalog therefore copies only those
metadata rows; the Parquet files under ``<ducklake_path>/storage`` are never
touched and the old ``metadata.sqlite`` is only read.

Steps:

1. Attach the SQLite lake (upgrading its metadata to the installed DuckLake
   version) and flush inlined rows into Parquet. SQLite stores inlined rows
   with text types, so they are flushed rather than copied.
2. Fingerprint every table (row count plus an order-independent hash of all
   rows) and record the latest snapshot.
3. Let DuckLake create an empty DuckDB catalog with its own column types, then
   copy every ``ducklake_*`` metadata table into it in one transaction.
4. Re-attach the new catalog as a DuckLake, recompute the fingerprints and
   compare. The target is written to a temporary file and only moved into
   place when everything matches.

No process may write to the lake while this runs; the helper aborts if the
source's latest snapshot changes during the copy.

Usage::

    python -m casmsocial.ducklake_migrate --ducklake-path /data/lake --to /data/lake/catalog.duckdb

Then serve ``catalog.duckdb`` over Quack and set ``CASMSOCIAL_DUCKLAKE_URI``
(see :mod:`casmsocial.ducklake_utils`). Unset it to fall back to the SQLite
catalog, which stays valid until the first write through the new one.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import duckdb
import typer
from loguru import logger

_INLINED_PREFIX = "ducklake_inlined_data_"
_INLINED_REGISTRY = "ducklake_inlined_data_tables"


class DuckLakeMigrationError(RuntimeError):
    """Raised when a catalog cannot be moved safely; the target is not created."""


@dataclass(frozen=True)
class TableFingerprint:
    rows: int
    row_hash: int


@dataclass
class MigrationReport:
    source: pathlib.Path
    target: pathlib.Path
    flushed: bool
    latest_snapshot: int
    snapshots: int
    metadata_tables: int
    tables: dict[str, TableFingerprint] = field(default_factory=dict)

    def summary(self) -> str:
        rows = sum(fp.rows for fp in self.tables.values())
        return (
            f"Moved DuckLake catalog {self.source} -> {self.target}: {len(self.tables)} tables, {rows} rows, "
            f"{self.snapshots} snapshots (latest {self.latest_snapshot}), {self.metadata_tables} metadata tables copied"
        )


def _sql_string(value: str | pathlib.Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@contextmanager
def _connection() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL sqlite; INSTALL ducklake; LOAD ducklake; LOAD sqlite;")
        try:  # tables with GEOMETRY columns need spatial to be read and hashed
            conn.execute("INSTALL spatial; LOAD spatial;")
        except duckdb.Error as exc:
            logger.debug(f"spatial extension unavailable: {exc}")
        yield conn
    finally:
        conn.close()


def _attach_lake(conn: duckdb.DuckDBPyConnection, catalog_uri: str, alias: str, storage: pathlib.Path) -> None:
    conn.execute(
        f"ATTACH {_sql_string(catalog_uri)} AS {alias} "
        f"(DATA_PATH {_sql_string('file://' + str(storage))}, OVERRIDE_DATA_PATH true, AUTOMATIC_MIGRATION true)"
    )


def _one(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> tuple:
    row = conn.execute(sql, params or []).fetchone()
    if row is None:  # pragma: no cover - aggregate queries always return a row
        msg = f"query returned no row: {sql}"
        raise DuckLakeMigrationError(msg)
    return row


def _fingerprints(conn: duckdb.DuckDBPyConnection, alias: str) -> dict[str, TableFingerprint]:
    tables = conn.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() WHERE database_name = ? ORDER BY 1, 2", [alias]
    ).fetchall()
    result: dict[str, TableFingerprint] = {}
    for schema, table in tables:
        qualified = f"{alias}.{_ident(schema)}.{_ident(table)}"
        rows, row_hash = _one(conn, f"SELECT count(*), coalesce(bit_xor(hash(t)), 0) FROM {qualified} t")
        result[f"{schema}.{table}"] = TableFingerprint(int(rows), int(row_hash))
    return result


def _snapshot_state(conn: duckdb.DuckDBPyConnection, alias: str) -> tuple[int, int]:
    count, latest = _one(conn, f"SELECT count(*), max(snapshot_id) FROM ducklake_snapshots({_sql_string(alias)})")
    return int(count), int(latest)


def _table_names(conn: duckdb.DuckDBPyConnection, database: str) -> set[str]:
    rows = conn.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = ?", [database]).fetchall()
    return {row[0] for row in rows}


def _columns(conn: duckdb.DuckDBPyConnection, database: str, table: str) -> list[str]:
    rows = conn.execute(
        "SELECT column_name FROM duckdb_columns() WHERE database_name = ? AND table_name = ? ORDER BY column_index",
        [database, table],
    ).fetchall()
    return [row[0] for row in rows]


def migrate_sqlite_catalog_to_duckdb(
    ducklake_path: pathlib.Path, target: pathlib.Path, *, flush: bool = True
) -> MigrationReport:
    """Copy the SQLite catalog of the lake at ``ducklake_path`` into a new DuckDB file ``target``.

    Raises :class:`DuckLakeMigrationError` (leaving no target behind) if the
    target exists, the source has inlined rows that were not flushed, the
    source changed during the copy, or the copied lake does not match.
    """
    ducklake_path = ducklake_path.expanduser().resolve()
    target = target.expanduser().resolve()
    source = ducklake_path / "metadata.sqlite"
    storage = ducklake_path / "storage"
    if not source.is_file():
        msg = f"no SQLite DuckLake catalog at {source}"
        raise DuckLakeMigrationError(msg)
    if target.exists():
        msg = f"target {target} already exists; choose a new path"
        raise DuckLakeMigrationError(msg)
    partial = target.with_name(target.name + ".partial")
    for leftover in (partial, partial.with_name(partial.name + ".wal")):
        leftover.unlink(missing_ok=True)

    try:
        with _connection() as conn:
            # 1. Upgrade + flush, then fingerprint the source as a DuckLake.
            _attach_lake(conn, f"ducklake:sqlite:{source}", "src_lake", storage)
            if flush:
                conn.execute("CALL ducklake_flush_inlined_data('src_lake')")
            before = _fingerprints(conn, "src_lake")
            snapshots, latest = _snapshot_state(conn, "src_lake")
            conn.execute("DETACH src_lake")

            # 2. Empty DuckDB catalog with DuckLake's own schema.
            _attach_lake(conn, f"ducklake:{partial}", "new_lake", storage)
            conn.execute("DETACH new_lake")

            # 3. Copy metadata rows.
            conn.execute(f"ATTACH {_sql_string(source)} AS src (TYPE sqlite, READ_ONLY)")
            conn.execute(f"ATTACH {_sql_string(partial)} AS dst")
            src_tables = _table_names(conn, "src")
            dst_tables = sorted(t for t in _table_names(conn, "dst") if t.startswith("ducklake_"))
            for table in sorted(src_tables):
                if table.startswith(_INLINED_PREFIX) and table != _INLINED_REGISTRY:
                    pending = _one(conn, f"SELECT count(*) FROM src.{_ident(table)}")[0]
                    if pending:
                        msg = f"{table} still holds {pending} inlined rows; run without --no-flush"
                        raise DuckLakeMigrationError(msg)
            missing = sorted(t for t in src_tables if t.startswith("ducklake_") and not t.startswith(_INLINED_PREFIX))
            missing = [t for t in missing if t not in dst_tables]
            if missing:
                msg = f"target catalog lacks metadata tables {missing}; DuckLake versions differ"
                raise DuckLakeMigrationError(msg)

            copied = 0
            conn.execute("BEGIN")
            for table in dst_tables:
                conn.execute(f"DELETE FROM dst.main.{_ident(table)}")
                if table == _INLINED_REGISTRY or table not in src_tables:
                    # Inlined tables are empty after the flush; DuckLake recreates them
                    # with proper DuckDB types the next time it inlines rows.
                    continue
                dst_cols, src_cols = _columns(conn, "dst", table), set(_columns(conn, "src", table))
                if lost := sorted(src_cols - set(dst_cols)):
                    msg = f"{table}: source columns {lost} have no place in the target; DuckLake versions differ"
                    raise DuckLakeMigrationError(msg)
                cols = ", ".join(_ident(c) for c in dst_cols if c in src_cols)
                conn.execute(f"INSERT INTO dst.main.{_ident(table)} ({cols}) SELECT {cols} FROM src.{_ident(table)}")
                copied += 1
            conn.execute("COMMIT")
            conn.execute("DETACH src")
            conn.execute("DETACH dst")

            # 4. Verify the source did not move and the new catalog matches.
            _attach_lake(conn, f"ducklake:sqlite:{source}", "src_lake", storage)
            if _snapshot_state(conn, "src_lake") != (snapshots, latest):
                msg = "the source lake changed during the copy; stop all writers and retry"
                raise DuckLakeMigrationError(msg)
            conn.execute("DETACH src_lake")
            _attach_lake(conn, f"ducklake:{partial}", "new_lake", storage)
            after = _fingerprints(conn, "new_lake")
            new_state = _snapshot_state(conn, "new_lake")
            conn.execute("DETACH new_lake")
            if after != before or new_state != (snapshots, latest):
                diff = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
                msg = (
                    f"copied lake does not match the source (tables differing: {diff}; "
                    f"snapshots {new_state} vs {(snapshots, latest)})"
                )
                raise DuckLakeMigrationError(msg)
        os.replace(partial, target)
    except BaseException:
        for leftover in (partial, partial.with_name(partial.name + ".wal")):
            leftover.unlink(missing_ok=True)
        raise

    report = MigrationReport(source, target, flush, latest, snapshots, copied, before)
    logger.info(report.summary())
    return report


def main(
    ducklake_path: pathlib.Path = typer.Option(
        ..., "--ducklake-path", help="Lake directory holding metadata.sqlite and storage/."
    ),
    to: pathlib.Path = typer.Option(..., "--to", help="New DuckDB catalog file to create, e.g. <lake>/catalog.duckdb."),
    flush: bool = typer.Option(
        True, "--flush/--no-flush", help="Flush inlined rows into Parquet first (modifies the source lake)."
    ),
) -> None:
    """Move a DuckLake's SQLite catalog into a new DuckDB catalog file."""
    try:
        report = migrate_sqlite_catalog_to_duckdb(ducklake_path, to, flush=flush)
    except DuckLakeMigrationError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(report.summary())
    for name, fp in sorted(report.tables.items()):
        typer.echo(f"  {name}: {fp.rows} rows")


if __name__ == "__main__":
    typer.run(main)
