"""Tests for moving a DuckLake catalog from SQLite to DuckDB (casmsocial.ducklake_migrate)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from casmsocial.ducklake_migrate import DuckLakeMigrationError, migrate_sqlite_catalog_to_duckdb
from casmsocial.ducklake_utils import get_ducklake_connection


def _loads(*extensions: str) -> bool:
    try:
        conn = duckdb.connect()
        try:
            for name in extensions:
                conn.execute(f"LOAD {name}")
        finally:
            conn.close()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _loads("ducklake", "sqlite_scanner", "spatial"), reason="ducklake/sqlite/spatial extensions not installed"
)


def _build_lake(path: Path) -> dict:
    """A small SQLite-catalog lake exercising the metadata the migration must carry."""
    conn = get_ducklake_connection(path, database_name="lake")
    try:
        conn.execute("CREATE SCHEMA wake")
        conn.execute("""CREATE TABLE wake.places AS
                        SELECT range AS sp_id, ST_Point(range, range * 2) AS geom, [range, range + 1] AS ids,
                               {'kind': 'home', 'n': range} AS info
                        FROM range(5000)""")
        conn.execute("CREATE TABLE main.runs (run_id VARCHAR, ticks INT, ok BOOLEAN, started TIMESTAMP)")
        conn.execute("INSERT INTO main.runs VALUES ('r1', 10, true, TIMESTAMP '2026-10-01 08:00')")  # inlined
        first = conn.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
        conn.execute("INSERT INTO main.runs VALUES ('r2', 20, false, TIMESTAMP '2026-10-02 09:30')")  # inlined
        conn.execute("DELETE FROM wake.places WHERE sp_id % 10 = 0")  # delete files
        conn.execute("ALTER TABLE wake.places ADD COLUMN rank INT")  # schema change
        conn.execute("UPDATE wake.places SET rank = sp_id % 4 WHERE sp_id < 100")
        return {"first_runs_snapshot": first}
    finally:
        conn.close()


def _rows(conn, sql):
    return conn.execute(sql).fetchall()


def test_moves_catalog_and_preserves_data_history_and_writes(tmp_path):
    lake = tmp_path / "lake"
    info = _build_lake(lake)
    with get_ducklake_connection(lake, database_name="lake") as conn:
        expected_places = _rows(conn, "SELECT count(*), sum(sp_id), sum(rank), sum(ST_X(geom)) FROM wake.places")
        expected_runs = _rows(conn, "FROM main.runs ORDER BY run_id")
    parquet_before = sorted(p.relative_to(lake) for p in (lake / "storage").rglob("*.parquet"))

    report = migrate_sqlite_catalog_to_duckdb(lake, lake / "catalog.duckdb")

    assert report.target == (lake / "catalog.duckdb").resolve()
    assert set(report.tables) == {"wake.places", "main.runs"}
    assert report.tables["main.runs"].rows == 2
    assert not (lake / "catalog.duckdb.partial").exists()

    catalog = f"ducklake:{lake / 'catalog.duckdb'}"
    conn = get_ducklake_connection(lake, database_name="lake", catalog_uri=catalog)
    try:
        assert (
            _rows(conn, "SELECT count(*), sum(sp_id), sum(rank), sum(ST_X(geom)) FROM wake.places") == expected_places
        )
        assert _rows(conn, "FROM main.runs ORDER BY run_id") == expected_runs
        # Time travel to before the second insert still works on the moved catalog.
        assert _rows(conn, f"SELECT run_id FROM main.runs AT (VERSION => {info['first_runs_snapshot']})") == [("r1",)]
        # New small writes inline again (into a freshly created, properly typed table) and large ones go to Parquet.
        conn.execute("INSERT INTO main.runs VALUES ('r3', 30, true, TIMESTAMP '2026-10-03 07:15')")
        conn.execute("INSERT INTO wake.places (sp_id, rank) SELECT range + 10000, 1 FROM range(2000)")
        assert _rows(conn, "SELECT count(*), max(ticks) FROM main.runs") == [(3, 30)]
        assert _rows(conn, "SELECT count(*) FROM wake.places WHERE sp_id >= 10000") == [(2000,)]
    finally:
        conn.close()

    # Parquet files that existed before were neither moved nor rewritten by the migration itself.
    parquet_after = {p.relative_to(lake) for p in (lake / "storage").rglob("*.parquet")}
    assert set(parquet_before) <= parquet_after
    # The SQLite catalog is still a readable, unmodified lake (apart from the flush).
    with get_ducklake_connection(lake, database_name="lake") as old:
        assert _rows(old, "SELECT count(*) FROM main.runs") == [(2,)]


def test_refuses_existing_target(tmp_path):
    lake = tmp_path / "lake"
    _build_lake(lake)
    (lake / "catalog.duckdb").write_bytes(b"")
    with pytest.raises(DuckLakeMigrationError, match="already exists"):
        migrate_sqlite_catalog_to_duckdb(lake, lake / "catalog.duckdb")


def test_refuses_missing_source(tmp_path):
    with pytest.raises(DuckLakeMigrationError, match="no SQLite DuckLake catalog"):
        migrate_sqlite_catalog_to_duckdb(tmp_path, tmp_path / "catalog.duckdb")


def test_refuses_unflushed_inlined_rows_and_leaves_no_target(tmp_path):
    lake = tmp_path / "lake"
    _build_lake(lake)
    with pytest.raises(DuckLakeMigrationError, match="inlined rows"):
        migrate_sqlite_catalog_to_duckdb(lake, lake / "catalog.duckdb", flush=False)
    assert not (lake / "catalog.duckdb").exists()
    assert not (lake / "catalog.duckdb.partial").exists()


def test_detects_a_mismatch_and_leaves_no_target(tmp_path, monkeypatch):
    import casmsocial.ducklake_migrate as migrate

    lake = tmp_path / "lake"
    _build_lake(lake)
    real = migrate._fingerprints
    calls = {"n": 0}

    def tampered(conn, alias):
        calls["n"] += 1
        result = real(conn, alias)
        if alias == "new_lake":
            result["main.runs"] = migrate.TableFingerprint(rows=-1, row_hash=0)
        return result

    monkeypatch.setattr(migrate, "_fingerprints", tampered)
    with pytest.raises(DuckLakeMigrationError, match="does not match"):
        migrate_sqlite_catalog_to_duckdb(lake, lake / "catalog.duckdb")
    assert not (lake / "catalog.duckdb").exists()
    assert not (lake / "catalog.duckdb.partial").exists()


def test_cli_reports_and_exits_nonzero_on_failure(tmp_path):
    import typer
    from typer.testing import CliRunner

    import casmsocial.ducklake_migrate as migrate

    lake = tmp_path / "lake"
    _build_lake(lake)
    app = typer.Typer()
    app.command()(migrate.main)
    ok = CliRunner().invoke(app, ["--ducklake-path", str(lake), "--to", str(lake / "catalog.duckdb")])
    assert ok.exit_code == 0, ok.output
    assert "wake.places: 4500 rows" in ok.output
    again = CliRunner().invoke(app, ["--ducklake-path", str(lake), "--to", str(lake / "catalog.duckdb")])
    assert again.exit_code == 1


@pytest.mark.skipif(not _loads("quack", "httpfs"), reason="quack/httpfs extensions not installed")
def test_moved_catalog_can_be_served_over_quack(tmp_path):
    import socket
    import subprocess
    import sys
    import textwrap

    from casmsocial.ducklake_utils import QUACK_TOKEN_ENV

    lake = tmp_path / "lake"
    _build_lake(lake)
    migrate_sqlite_catalog_to_duckdb(lake, lake / "catalog.duckdb")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        uri = f"quack:localhost:{sock.getsockname()[1]}"
    token = "lake-token-0123456789"
    server = subprocess.Popen(  # noqa: S603 - fixed interpreter and test-authored script
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import time, duckdb
                con = duckdb.connect({str(lake / 'catalog.duckdb')!r})
                con.execute("LOAD quack")
                con.execute("CALL quack_serve(?, token => ?)", [{uri!r}, {token!r}])
                print("ready", flush=True)
                while True:
                    time.sleep(1)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert server.stdout.readline().strip() == "ready"
        conn = get_ducklake_connection(
            lake, database_name="lake", catalog_uri=f"ducklake:{uri}", environ={QUACK_TOKEN_ENV: token}
        )
        try:
            assert _rows(conn, "SELECT count(*) FROM wake.places") == [(4500,)]
            conn.execute("INSERT INTO main.runs VALUES ('r4', 40, true, TIMESTAMP '2026-10-04 06:00')")
            assert _rows(conn, "SELECT count(*) FROM main.runs") == [(3,)]
        finally:
            conn.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
