from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from casmsocial.datasets.colorado_front_range.sources import sha256_file
from scripts import create_colorado_front_range_fixture_ducklake as fixture


def _runtime_product(tmp_path: Path) -> Path:
    runtime_dir = tmp_path / "runtime"
    export_dir = runtime_dir / "casmsocial"
    export_dir.mkdir(parents=True)
    tables = {
        "activities": {"sp_persons_id": [1], "activity_id": [0], "sp_act_id": [100]},
        "persons": {"sp_id": [1], "sp_hh_id": [100]},
        "hh": {"sp_id": [100], "sp_home_id": [100], "hh_size": [1]},
        "places": {"sp_id": [100], "rank": [0], "place_type": ["home"]},
        "social_networks": {"person_id_a": [], "person_id_b": [], "network_kind": []},
    }
    outputs: dict[str, dict[str, object]] = {}
    for name, columns in tables.items():
        path = export_dir / f"{name}.parquet"
        pl.DataFrame(columns).write_parquet(path)
        outputs[f"casmsocial/{name}.parquet"] = {"sha256": sha256_file(path)}
    (runtime_dir / "manifest.json").write_text(
        json.dumps({"status": "passed", "outputs": outputs}), encoding="utf-8"
    )
    return runtime_dir


class _Result:
    def fetchone(self):
        return (1,)


class _Connection:
    def __init__(self):
        self.calls: list[tuple[str, list[str] | None]] = []

    def execute(self, query: str, parameters=None):
        self.calls.append((query, parameters))
        return _Result()

    def close(self):
        pass


def test_materialize_fixture_loads_a_verified_runtime_product(tmp_path: Path, monkeypatch):
    runtime_dir = _runtime_product(tmp_path)
    connection = _Connection()
    monkeypatch.setattr(fixture, "get_ducklake_connection", lambda path: connection)

    counts = fixture.materialize_fixture(runtime_dir, tmp_path / "ducklake", partition_ranks=2)

    assert counts == {
        "activities": 1,
        "persons": 1,
        "hh": 1,
        "places": 1,
        "social_networks": 1,
        "place_partitions": 1,
    }
    loaded = [parameters[0] for query, parameters in connection.calls if "read_parquet(?)" in query]
    assert loaded == [str(runtime_dir / "casmsocial" / f"{name}.parquet") for name in fixture.TABLES]
    assert any("colorado_front_range" in query for query, _ in connection.calls)
    assert any("n_ranks, CAST(hash(sp_id) % 2" in query for query, _ in connection.calls)


def test_materialize_fixture_rejects_a_runtime_table_with_a_bad_manifest_hash(tmp_path: Path):
    runtime_dir = _runtime_product(tmp_path)
    (runtime_dir / "casmsocial" / "persons.parquet").write_bytes(b"changed")

    with pytest.raises(ValueError, match="persons"):
        fixture.materialize_fixture(runtime_dir, tmp_path / "ducklake")
