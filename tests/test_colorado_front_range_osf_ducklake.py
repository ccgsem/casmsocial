import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from casmsocial.datasets.colorado_front_range.osf_ducklake import (
    build_ducklake,
    validate_state_partitions,
)
from casmsocial.datasets.colorado_front_range.sources import sha256_file


def _write_state_partition(root: Path, state: str = "CO", *, unresolved_tie: bool = False) -> Path:
    state_dir = root / f"source_state={state}"
    state_dir.mkdir(parents=True)
    offset = 0 if state == "CO" else 1_000
    tables = {
        "places": pl.DataFrame({
            "sp_id": [offset + 10, offset + 100],
            "place_type": ["Household", "Workplace"],
            "longitude": [-104.9, -104.8],
            "latitude": [39.7, 39.8],
            "source_state": [state, state],
        }),
        "hh": pl.DataFrame({
            "sp_id": [offset + 10],
            "sp_home_id": [offset + 10],
            "hh_size": [2],
            "household_type": ["2"],
            "source_state": [state],
        }),
        "persons": pl.DataFrame(
            {
                "sp_id": [offset + 1, offset + 2],
                "sp_hh_id": [offset + 10, offset + 10],
                "sp_work_id": [offset + 100, None],
                "activity_assignment_kind": ["work", None],
                "age": [30.0, 12.0],
                "gender": ["male", "female"],
                "assigned": [1, 1],
                "urban": [1, 1],
                "household_type": ["2", "2"],
                "home_longitude": [-104.9, -104.9],
                "home_latitude": [39.7, 39.7],
                "source_state": [state, state],
            },
            schema={
                "sp_id": pl.Int64,
                "sp_hh_id": pl.Int64,
                "sp_work_id": pl.Int64,
                "activity_assignment_kind": pl.String,
                "age": pl.Float64,
                "gender": pl.String,
                "assigned": pl.Int64,
                "urban": pl.Int64,
                "household_type": pl.String,
                "home_longitude": pl.Float64,
                "home_latitude": pl.Float64,
                "source_state": pl.String,
            },
        ),
        "social_networks": pl.DataFrame({
            "person_id_a": [offset + 1],
            "person_id_b": [offset + 999 if unresolved_tie else offset + 2],
            "network_kind": ["household"],
            "source_state": [state],
        }),
    }
    manifest_tables = {}
    for name, frame in tables.items():
        path = state_dir / f"{name}.parquet"
        frame.write_parquet(path)
        manifest_tables[name] = {
            "path": str(path),
            "rows": frame.height,
            "sha256": sha256_file(path),
            "schema": {key: str(value) for key, value in frame.schema.items()},
        }
    manifest = {"schema_version": 1, "state": state, "tables": manifest_tables}
    (state_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return state_dir


def _query_counts(catalog: Path, data_path: Path) -> dict[str, int]:
    connection = duckdb.connect()
    try:
        connection.execute("LOAD ducklake")
        connection.execute(f"ATTACH 'ducklake:{catalog}' AS lake (DATA_PATH '{data_path}')")
        connection.execute("USE lake")
        return {
            name: connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in ("places", "hh", "persons", "social_networks")
        }
    finally:
        connection.close()


def test_build_ducklake_materializes_accepts_and_resumes_unchanged_sources(tmp_path: Path):
    source = tmp_path / "states"
    _write_state_partition(source)
    catalog = tmp_path / "lake" / "metadata.ducklake"
    data_path = tmp_path / "lake" / "files"

    manifest = build_ducklake(source, catalog, data_path)
    counts_after_reopen = _query_counts(catalog, data_path)
    resumed = build_ducklake(source, catalog, data_path)

    assert manifest["status"] == "passed"
    assert manifest["tables"] == {"places": 2, "hh": 1, "persons": 2, "social_networks": 1}
    assert manifest["acceptance"]["checks"]["activity_assignments_without_place"] == 0
    assert resumed["resumed"] is True
    assert counts_after_reopen == manifest["tables"]


def test_build_ducklake_combines_multiple_state_partitions(tmp_path: Path):
    source = tmp_path / "states"
    _write_state_partition(source, "CO")
    _write_state_partition(source, "VA")

    manifest = build_ducklake(
        source,
        tmp_path / "lake" / "metadata.ducklake",
        tmp_path / "lake" / "files",
    )

    assert manifest["states"] == ["CO", "VA"]
    assert manifest["tables"] == {"places": 4, "hh": 2, "persons": 4, "social_networks": 2}
    assert manifest["persons_by_state"] == [
        {"source_state": "CO", "persons": 2},
        {"source_state": "VA", "persons": 2},
    ]


def test_state_validation_rejects_table_changed_after_manifest(tmp_path: Path):
    source = tmp_path / "states"
    state_dir = _write_state_partition(source)
    with (state_dir / "persons.parquet").open("ab") as target:
        target.write(b"changed")

    with pytest.raises(ValueError, match="hash does not match"):
        validate_state_partitions(source)


def test_ducklake_acceptance_rejects_unresolved_social_endpoint(tmp_path: Path):
    source = tmp_path / "states"
    _write_state_partition(source, unresolved_tie=True)
    catalog = tmp_path / "lake" / "metadata.ducklake"
    data_path = tmp_path / "lake" / "files"

    with pytest.raises(ValueError, match="acceptance failed"):
        build_ducklake(source, catalog, data_path)
    assert not catalog.exists()
    assert not data_path.exists()


def test_state_validation_requires_manifested_partitions(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No state partitions"):
        validate_state_partitions(tmp_path)
