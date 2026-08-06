import json
from pathlib import Path

import duckdb
import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from casmsocial.datasets.colorado_front_range import build_profile_population, load_profile
from casmsocial.datasets.colorado_front_range.osf_ducklake import build_ducklake
from casmsocial.datasets.colorado_front_range.sources import sha256_file


def _write_state_partition(root: Path, *, include_social_ties: bool = True) -> None:
    state_dir = root / "source_state=CO"
    state_dir.mkdir(parents=True)
    coordinates = [(0.5, 0.5), (2.5, 0.5), (4.5, 0.5), (6.5, 0.5)] * 2
    person_ids = list(range(1, 9))
    home_ids = list(range(101, 109))
    work_ids = list(range(201, 209))
    tables = {
        "places": pl.DataFrame({
            "sp_id": home_ids + work_ids,
            "place_type": ["Household"] * 8 + ["Workplace"] * 8,
            "longitude": [point[0] for point in coordinates] * 2,
            "latitude": [point[1] for point in coordinates] * 2,
            "source_state": ["CO"] * 16,
        }),
        "hh": pl.DataFrame({
            "sp_id": home_ids,
            "sp_home_id": home_ids,
            "hh_size": [1] * 8,
            "household_type": ["1"] * 8,
            "source_state": ["CO"] * 8,
        }),
        "persons": pl.DataFrame({
            "sp_id": person_ids,
            "sp_hh_id": home_ids,
            "sp_work_id": work_ids,
            "activity_assignment_kind": ["work", "school", "work", "school"] * 2,
            "age": [34.0, 12.0, 45.0, 17.0, 70.0, 4.0, 28.0, 55.0],
            "gender": ["female", "male"] * 4,
            "assigned": [1] * 8,
            "urban": [1] * 8,
            "household_type": ["1"] * 8,
            "home_longitude": [point[0] for point in coordinates],
            "home_latitude": [point[1] for point in coordinates],
            "source_state": ["CO"] * 8,
        }),
        "social_networks": pl.DataFrame(
            {
                "person_id_a": [1, 3, 5, 7] if include_social_ties else [],
                "person_id_b": [2, 4, 6, 8] if include_social_ties else [],
                "network_kind": ["social"] * 4 if include_social_ties else [],
                "source_state": ["CO"] * 4 if include_social_ties else [],
            },
            schema={
                "person_id_a": pl.Int64,
                "person_id_b": pl.Int64,
                "network_kind": pl.String,
                "source_state": pl.String,
            },
        ),
    }
    manifested = {}
    for name, frame in tables.items():
        path = state_dir / f"{name}.parquet"
        frame.write_parquet(path)
        manifested[name] = {
            "path": str(path),
            "rows": frame.height,
            "sha256": sha256_file(path),
            "schema": {column: str(dtype) for column, dtype in frame.schema.items()},
        }
    (state_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "state": "CO", "tables": manifested}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_counties(path: Path) -> None:
    geoids = [
        "08013",
        "08001",
        "08005",
        "08014",
        "08019",
        "08031",
        "08035",
        "08039",
        "08047",
        "08059",
        "08093",
        "08069",
        "08123",
    ]
    inhabited = {
        "08013": box(0, 0, 1, 1),
        "08031": box(2, 0, 3, 1),
        "08069": box(4, 0, 5, 1),
        "08123": box(6, 0, 7, 1),
    }
    counties = gpd.GeoDataFrame(
        {
            "GEOID": geoids,
            "STATEFP": ["08"] * len(geoids),
        },
        geometry=[inhabited.get(geoid, box(20 + index, 0, 21 + index, 1)) for index, geoid in enumerate(geoids)],
        crs="EPSG:4326",
    )
    counties.to_file(path, driver="GPKG")


def _small_profile(person_limit: int = 6):
    profile = load_profile("example-1k")
    population = profile.population.model_copy(update={"person_limit": person_limit, "minimum_persons_per_cbsa": 1})
    return profile.model_copy(update={"profile_id": "test-profile", "population": population})


def _input_lake(tmp_path: Path, *, include_social_ties: bool = True) -> tuple[Path, Path, Path]:
    source = tmp_path / "states"
    _write_state_partition(source, include_social_ties=include_social_ties)
    catalog = tmp_path / "lake" / "metadata.ducklake"
    data_path = tmp_path / "lake" / "files"
    build_ducklake(source, catalog, data_path)
    counties = tmp_path / "counties.gpkg"
    _write_counties(counties)
    return catalog, data_path, counties


def test_build_profile_population_is_spatial_bounded_closed_and_resumable(tmp_path: Path):
    catalog, data_path, counties = _input_lake(tmp_path)
    output = tmp_path / "profile"

    manifest = build_profile_population(
        catalog,
        data_path,
        counties,
        _small_profile(),
        output,
        batch_size=2,
    )
    resumed = build_profile_population(
        catalog,
        data_path,
        counties,
        _small_profile(),
        output,
        batch_size=2,
    )

    assert manifest["status"] == "passed"
    assert manifest["counts"]["persons"] == 6
    assert set(manifest["persons_by_cbsa"]) == {"14540", "19740", "22660", "24540"}
    assert manifest["sampling"]["network_seed_ties"] > 0
    assert manifest["counts"]["social_networks"] > 0
    assert manifest["integrity"]["status"] == "passed"
    assert manifest["schedule_eligibility"]["status"] == "deferred_until_schedule_generation"
    assert set(manifest["tables"]) == {"places", "hh", "persons", "social_networks"}
    assert resumed["resumed"] is True

    people = set(pl.read_parquet(output / "persons.parquet")["sp_id"].to_list())
    ties = pl.read_parquet(output / "social_networks.parquet")
    assert set(ties["person_id_a"].to_list()) <= people
    assert set(ties["person_id_b"].to_list()) <= people


def test_full_profile_selects_every_person_in_boundary(tmp_path: Path):
    catalog, data_path, counties = _input_lake(tmp_path)

    manifest = build_profile_population(
        catalog,
        data_path,
        counties,
        load_profile("north-corridor-full"),
        tmp_path / "full-profile",
        batch_size=3,
    )

    assert manifest["counts"]["persons"] == 8
    assert manifest["sampling"]["method"] == "full_boundary_population"


def test_smoke_profile_rejects_source_without_sampleable_social_tie(tmp_path: Path):
    catalog, data_path, counties = _input_lake(tmp_path, include_social_ties=False)

    with pytest.raises(ValueError, match="runtime smoke requires at least one"):
        build_profile_population(
            catalog,
            data_path,
            counties,
            _small_profile(),
            tmp_path / "profile",
            batch_size=2,
        )


def test_planned_profile_requires_explicit_override(tmp_path: Path):
    catalog, data_path, counties = _input_lake(tmp_path)

    with pytest.raises(ValueError, match="is planned; pass --allow-planned"):
        build_profile_population(
            catalog,
            data_path,
            counties,
            load_profile("six-metro-full"),
            tmp_path / "planned-profile",
        )


def test_profile_builder_rejects_unaccepted_catalog(tmp_path: Path):
    catalog = tmp_path / "metadata.ducklake"
    duckdb.connect(catalog).close()
    counties = tmp_path / "counties.gpkg"
    _write_counties(counties)

    with pytest.raises(FileNotFoundError):
        build_profile_population(
            catalog,
            tmp_path / "files",
            counties,
            _small_profile(),
            tmp_path / "profile",
        )
