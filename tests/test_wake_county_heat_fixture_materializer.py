from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import scripts.materialize_wake_county_heat_fixture as materializer


class _NoCloseConnection:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def close(self) -> None:
        pass

    def execute(self, *args: object, **kwargs: object) -> duckdb.DuckDBPyConnection:
        return self._conn.execute(*args, **kwargs)


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None]] = []

    def execute(self, sql: str, params: object | None = None) -> _RecordingConnection:
        self.statements.append((sql, params))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return []

    def fetchone(self) -> tuple[int]:
        return (0,)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _describe_parquet(path: Path) -> list[dict[str, object]]:
    conn = duckdb.connect(":memory:")
    try:
        rows = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    finally:
        conn.close()
    return [{"name": row[0], "type": row[1], "nullable": row[2] == "YES"} for row in rows]


def _write_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "fixture"
    tables_path = fixture_path / "tables"
    tables_path.mkdir(parents=True)

    table_path = tables_path / "persons_1000_households.parquet"
    table = pa.Table.from_pylist([
        {"sp_id": 1, "sp_hh_id": 100, "Imputation": 1},
        {"sp_id": 2, "sp_hh_id": 100, "Imputation": 1},
    ])
    pq.write_table(table, table_path, compression="zstd")

    manifest = {
        "version": 1,
        "fixture_id": "test_fixture",
        "source": {"schema": "wake_county_heat"},
        "tables": [
            {
                "name": "persons_1000_households",
                "source_table": "wake_county_heat.persons_1000_households",
                "file": "tables/persons_1000_households.parquet",
                "format": "parquet",
                "compression": "zstd",
                "rows": 2,
                "size_bytes": table_path.stat().st_size,
                "sha256": _sha256(table_path),
                "columns": _describe_parquet(table_path),
            }
        ],
    }
    (fixture_path / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return fixture_path


def test_materialize_fixture_loads_manifest_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_path = _write_fixture(tmp_path)
    conn = duckdb.connect(":memory:")
    monkeypatch.setattr(materializer, "get_ducklake_connection", lambda *args, **kwargs: _NoCloseConnection(conn))

    try:
        result = materializer.materialize_fixture(fixture_path, tmp_path / "datalakehouse")

        assert result.schema_name == "wake_county_heat"
        assert result.tables == {"persons_1000_households": 2}
        assert conn.execute("SELECT COUNT(*) FROM wake_county_heat.persons_1000_households").fetchone() == (2,)
    finally:
        conn.close()


def test_validate_fixture_rejects_checksum_mismatch(tmp_path: Path) -> None:
    fixture_path = _write_fixture(tmp_path)
    manifest_path = fixture_path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"][0]["sha256"] = "bad"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(materializer.WakeCountyHeatFixtureError, match="checksum mismatch"):
        materializer.validate_fixture(fixture_path)


def test_committed_fixture_records_pending_data_approval_boundary() -> None:
    manifest_path = Path("testdata/wake_county_heat_1000_households/manifest.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    classification = manifest["data_classification"]

    assert classification["status"] == "pending_data_owner_approval"
    assert classification["approval"]["decision"] == "not recorded"
    assert classification["approval"]["evidence_uri"] == ""
    assert "github.com/ccgsem/casmsocial" in classification["current_sharing_boundary"]["known_repository_locations"]
    assert "Do not mirror" in classification["current_sharing_boundary"]["broader_redistribution"]


def test_get_ducklake_postgres_s3_connection_attaches_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _RecordingConnection()
    monkeypatch.setattr(materializer.duckdb, "connect", lambda: conn)

    result = materializer.get_ducklake_postgres_s3_connection(
        materializer.PostgresS3DuckLakeConfig(
            postgres_connection="host=ducklake-postgres port=5432 dbname=casmsocial user=casmsocial password=secret",
            s3_data_path="s3://casmsocial-ducklake/wake_county_heat",
            s3_access_key_id="casmsocial",
            s3_secret_access_key="minio-secret",
            s3_region="us-east-1",
            s3_endpoint="ducklake-minio:9000",
            s3_use_ssl=False,
        )
    )

    assert result is conn
    statements = "\n".join(statement for statement, _ in conn.statements)
    assert "INSTALL postgres" in statements
    assert "INSTALL httpfs" in statements
    assert "INSTALL ducklake" in statements
    assert "CREATE OR REPLACE SECRET" in statements
    assert "ENDPOINT 'ducklake-minio:9000'" in statements
    assert "USE_SSL false" in statements
    assert "ATTACH 'ducklake:postgres:host=ducklake-postgres" in statements
    assert "DATA_PATH 's3://casmsocial-ducklake/wake_county_heat'" in statements


def test_materialization_result_redacts_postgres_password(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_path = _write_fixture(tmp_path)
    conn = duckdb.connect(":memory:")
    monkeypatch.setattr(
        materializer, "get_ducklake_postgres_s3_connection", lambda *args, **kwargs: _NoCloseConnection(conn)
    )

    try:
        result = materializer.materialize_postgres_s3_fixture(
            fixture_path,
            materializer.PostgresS3DuckLakeConfig(
                postgres_connection="host=postgres dbname=casmsocial user=casmsocial password=secret",
                s3_data_path="s3://casmsocial-ducklake/wake_county_heat",
                s3_access_key_id="casmsocial",
                s3_secret_access_key="minio-secret",
            ),
        )

        assert result.catalog == "ducklake:postgres:host=postgres dbname=casmsocial user=casmsocial password=<redacted>"
    finally:
        conn.close()
