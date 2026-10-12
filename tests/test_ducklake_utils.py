"""Tests for DuckLake catalog selection in casmsocial.ducklake_utils."""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import pytest

from casmsocial.ducklake_utils import (
    CATALOG_URI_ENV,
    QUACK_TOKEN_ENV,
    DuckLakeCatalogError,
    ducklake_catalog_uri_from_env,
    get_ducklake_connection,
    resolve_catalog_uri,
)

TOKEN = "lake-token-0123456789"


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


requires_ducklake = pytest.mark.skipif(
    not _loads("ducklake", "sqlite_scanner", "spatial"), reason="ducklake/sqlite/spatial extensions not installed"
)
requires_quack = pytest.mark.skipif(
    not _loads("ducklake", "sqlite_scanner", "spatial", "quack", "httpfs"),
    reason="ducklake/quack/httpfs extensions not installed",
)


def test_default_catalog_is_sqlite_inside_the_lake(tmp_path):
    assert resolve_catalog_uri(tmp_path) == f"ducklake:sqlite:{tmp_path / 'metadata.sqlite'}"


@pytest.mark.parametrize(
    "uri",
    [
        "ducklake:quack:localhost:9494",
        "ducklake:/data/catalog.duckdb",
        "ducklake:postgres:dbname=lake",
        "  ducklake:x  ",
    ],
)
def test_explicit_catalog_uri_is_used(tmp_path, uri):
    assert resolve_catalog_uri(tmp_path, uri) == uri.strip()


@pytest.mark.parametrize("uri", ["quack:localhost:9494", "ducklake:", "/data/metadata.sqlite"])
def test_invalid_catalog_uri_is_rejected(tmp_path, uri):
    with pytest.raises(DuckLakeCatalogError, match="ducklake:"):
        resolve_catalog_uri(tmp_path, uri)


def test_catalog_uri_from_env():
    assert ducklake_catalog_uri_from_env({}) is None
    assert ducklake_catalog_uri_from_env({CATALOG_URI_ENV: "  "}) is None
    assert ducklake_catalog_uri_from_env({CATALOG_URI_ENV: "ducklake:quack:h:1"}) == "ducklake:quack:h:1"


def test_quack_catalog_requires_a_token_before_connecting(tmp_path, monkeypatch):
    def fail_connect(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("connected without a token")

    monkeypatch.setattr(duckdb, "connect", fail_connect)
    with pytest.raises(DuckLakeCatalogError, match=QUACK_TOKEN_ENV):
        get_ducklake_connection(tmp_path, catalog_uri="ducklake:quack:localhost:9494", environ={})


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.closed = False

    def execute(self, sql: str):
        self.statements.append(sql)
        return self

    def close(self) -> None:
        self.closed = True


def test_quack_catalog_registers_a_scoped_secret_before_attaching(tmp_path, monkeypatch):
    conn = _RecordingConnection()
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: conn)

    result = get_ducklake_connection(
        tmp_path, catalog_uri="ducklake:quack:lakehost:9494", environ={QUACK_TOKEN_ENV: "it's-a-token"}
    )

    assert result is conn
    secret = next(s for s in conn.statements if "SECRET" in s)
    install = next(i for i, s in enumerate(conn.statements) if "INSTALL quack" in s)
    assert install < conn.statements.index(secret)
    assert "TYPE quack" in secret and "SCOPE 'quack:lakehost:9494'" in secret
    assert "TOKEN 'it''s-a-token'" in secret  # quoted safely
    attach = conn.statements[-1]
    assert conn.statements.index(secret) < len(conn.statements) - 1
    assert "ATTACH 'ducklake:quack:lakehost:9494' AS insights_ducklake" in attach
    assert f"DATA_PATH 'file://{tmp_path.resolve() / 'storage'}'" in attach


def test_connection_is_closed_when_attach_fails(tmp_path, monkeypatch):
    class _Failing(_RecordingConnection):
        def execute(self, sql: str):
            if "ATTACH" in sql:
                raise duckdb.IOException("cannot reach catalog")
            return super().execute(sql)

    conn = _Failing()
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: conn)
    with pytest.raises(duckdb.IOException):
        get_ducklake_connection(tmp_path, catalog_uri="ducklake:/nowhere/catalog.duckdb")
    assert conn.closed


@requires_ducklake
def test_default_sqlite_catalog_still_works(tmp_path):
    conn = get_ducklake_connection(tmp_path / "lake")
    try:
        conn.execute("CREATE TABLE t AS SELECT range AS i FROM range(5)")
        assert conn.execute("SELECT sum(i) FROM t").fetchone()[0] == 10
    finally:
        conn.close()
    assert (tmp_path / "lake" / "metadata.sqlite").exists()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def quack_catalog(tmp_path):
    """A DuckDB catalog file served over Quack by a separate process."""
    catalog = tmp_path / "catalog.duckdb"
    duckdb.connect(str(catalog)).close()
    uri = f"quack:localhost:{_free_port()}"
    server = subprocess.Popen(  # noqa: S603 - fixed interpreter and test-authored script
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time, duckdb
                con = duckdb.connect({str(catalog)!r})
                con.execute("LOAD quack")
                con.execute("CALL quack_serve(?, token => ?)", [{uri!r}, {TOKEN!r}])
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
        yield f"ducklake:{uri}"
    finally:
        server.terminate()
        server.wait(timeout=10)


@requires_quack
def test_quack_served_catalog_with_local_parquet_storage(tmp_path, quack_catalog):
    lake = tmp_path / "lake"
    env = {QUACK_TOKEN_ENV: TOKEN}

    writer = get_ducklake_connection(lake, catalog_uri=quack_catalog, environ=env)
    try:
        writer.execute("CREATE SCHEMA wake_county_heat")
        writer.execute("CREATE TABLE wake_county_heat.places AS SELECT range AS sp_id FROM range(1000)")
    finally:
        writer.close()

    reader = get_ducklake_connection(lake, catalog_uri=quack_catalog, environ=env)
    try:
        assert reader.execute("SELECT count(*), sum(sp_id) FROM wake_county_heat.places").fetchone() == (1000, 499500)
    finally:
        reader.close()

    assert list((lake / "storage").rglob("*.parquet")), "Parquet data should be written under <lake>/storage"
    assert not (lake / "metadata.sqlite").exists(), "the SQLite catalog must not be created"

    with pytest.raises(duckdb.Error):
        get_ducklake_connection(lake, catalog_uri=quack_catalog, environ={QUACK_TOKEN_ENV: "wrong-token-000000"})


def test_partitioner_cli_defaults_catalog_uri_from_env(tmp_path, monkeypatch):
    import typer
    from typer.testing import CliRunner

    import casmsocial.network_partitioner_ducklake as partitioner

    seen: dict = {}
    monkeypatch.setattr(partitioner, "partition_many_from_ducklake", lambda *a, **k: seen.update(k))
    monkeypatch.setenv(CATALOG_URI_ENV, "ducklake:quack:lakehost:9494")
    app = typer.Typer()
    app.command()(partitioner.main)
    result = CliRunner().invoke(
        app,
        ["--ducklake-path", str(tmp_path), "--schema", "s", "--imputations", "1", "--n-ranks", "2"],
    )
    assert result.exit_code == 0, result.output
    assert seen["catalog_uri"] == "ducklake:quack:lakehost:9494"


def test_model_reads_catalog_uri_from_env(tmp_path, monkeypatch):
    import casmsocial.casmpop as casmpop

    calls: list = []
    monkeypatch.setattr(casmpop, "get_ducklake_connection", lambda path, **kw: calls.append((path, kw)) or object())
    monkeypatch.setenv("CASMSOCIAL_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("CASMSOCIAL_DUCKLAKE_PATH", str(tmp_path / "lake"))
    monkeypatch.setenv(CATALOG_URI_ENV, "ducklake:quack:lakehost:9494")

    model = casmpop.CasmPop.__new__(casmpop.CasmPop)
    model._set_data_resources()

    assert calls == [(Path(tmp_path / "lake"), {"catalog_uri": "ducklake:quack:lakehost:9494"})]
