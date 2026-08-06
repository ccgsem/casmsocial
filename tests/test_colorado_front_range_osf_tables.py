from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest

from casmsocial.datasets.colorado_front_range import osf_tables
from casmsocial.datasets.colorado_front_range.osf_tables import (
    assignment_kind,
    build_state_tables,
    person_rows,
    scoped_id,
    social_network_rows,
    workplace_rows,
)


def _population_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["p1", "p2"],
        "age": [30.0, 12.0],
        "gender": ["male", "female"],
        "assigned": [1, 1],
        "hhold": ["coh1", "coh1"],
        "htype": ["2", "2"],
        "wp": ["cow1", "cos1"],
        "urban": [1, 1],
        "long": [-104.9, -104.9],
        "lat": [39.7, 39.7],
    })


def test_rows_use_deterministic_state_scoped_identifiers():
    persons = person_rows("CO", _population_frame())
    workplaces = workplace_rows("CO", pl.DataFrame({"wp": ["cow1"], "long": [-104.8], "lat": [39.8]}))

    assert persons["sp_id"].to_list()[0] == scoped_id("CO", "person", "p1")
    assert persons["sp_hh_id"].n_unique() == 1
    assert persons["activity_assignment_kind"].to_list() == ["work", "school"]
    assert workplaces["sp_id"].to_list() == [scoped_id("CO", "workplace", "cow1")]
    assert scoped_id("CO", "person", "p1") != scoped_id("VA", "person", "p1")
    assert assignment_kind("24001000100d7") == "daycare"


def test_social_network_rows_are_timeless_canonical_person_pairs(tmp_path: Path):
    archive = tmp_path / "co.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("co/social_networks/co_household_network.csv", "p1,p2\np2,p1\n")
        output.writestr("co/social_networks/co_daycare_network.csv", "p1\n")
        output.writestr("co/social_networks/co_school_network.csv", "p2\n")
        output.writestr("co/social_networks/co_work_network.csv", ",source,target\n0,p1,p2\n")

    networks = social_network_rows(archive, "CO")
    assert networks.height == 1
    assert networks["network_kind"].to_list() == ["household"]
    assert "hour" not in networks.columns
    assert networks.select((pl.col("person_id_a") < pl.col("person_id_b")).all()).item()


def test_build_state_tables_emits_four_manifested_endpoint_complete_tables(tmp_path: Path, monkeypatch):
    population_archive = tmp_path / "co.zip"
    with ZipFile(population_archive, "w") as output:
        output.writestr("unsafe/../../co_population.gpkg", "population")
        output.writestr("co/co_workplace.gpkg", "workplace")
        output.writestr("co/social_networks/co_household_network.csv", "p1,p2,missing\n")
        output.writestr("co/social_networks/co_daycare_network.csv", "p1\n")
        output.writestr("co/social_networks/co_school_network.csv", "p2\n")
        output.writestr("co/social_networks/co_work_network.csv", ",source,target\n0,p1,p2\n")
    education_archive = tmp_path / "co_edu_sites.zip"
    with ZipFile(education_archive, "w") as output:
        output.writestr("co_school_id.gpkg", "school")
        output.writestr("co_daycare_id.gpkg", "daycare")

    monkeypatch.setattr(osf_tables, "_feature_count", lambda path: 2)

    def batches(path, columns, batch_size, *, read_geometry=False):
        if path.name == "population.gpkg":
            yield _population_frame()
        elif path.name == "workplace.gpkg":
            yield pl.DataFrame({"wp": ["cow1"], "long": [-104.8], "lat": [39.8]})
        elif path.name == "school.gpkg":
            yield pl.DataFrame({"eduID": ["cos1"], "longitude": [-104.7], "latitude": [39.6]})
        else:
            yield pl.DataFrame({"eduID": ["cod1"], "longitude": [-104.6], "latitude": [39.5]})

    monkeypatch.setattr(osf_tables, "_iter_geopackage", batches)
    manifest = build_state_tables(
        population_archive,
        "CO",
        tmp_path / "tables",
        education_archive=education_archive,
        batch_size=1,
    )
    state_dir = tmp_path / "tables" / "source_state=CO"

    assert set(manifest["tables"]) == {"places", "hh", "persons", "social_networks"}
    assert {name: table["rows"] for name, table in manifest["tables"].items()} == {
        "places": 4,
        "hh": 1,
        "persons": 2,
        "social_networks": 1,
    }
    assert manifest["social_networks"]["excluded_unresolved_endpoint_rows"] == 1
    assert manifest["social_networks"]["cross_state_ties"] is False
    assert manifest["social_networks"]["excluded_source_networks"] == {
        "work": "non-person-source work memberships; not person-person social ties"
    }
    person_ids = set(pl.read_parquet(state_dir / "persons.parquet")["sp_id"])
    ties = pl.read_parquet(state_dir / "social_networks.parquet")
    assert set(ties["person_id_a"]).issubset(person_ids)
    assert set(ties["person_id_b"]).issubset(person_ids)
    assert (state_dir / "manifest.json").is_file()


def test_build_state_tables_reads_real_geopackages_when_builder_extra_is_installed(tmp_path: Path):
    geopandas = pytest.importorskip("geopandas")
    geometry = pytest.importorskip("shapely.geometry")
    source_dir = tmp_path / "geopackages"
    source_dir.mkdir()
    population_path = source_dir / "co_population.gpkg"
    workplace_path = source_dir / "co_workplace.gpkg"
    school_path = source_dir / "co_school_id.gpkg"
    daycare_path = source_dir / "co_daycare_id.gpkg"
    geopandas.GeoDataFrame(
        _population_frame().to_pandas(),
        geometry=[geometry.Point(-104.9, 39.7), geometry.Point(-104.9, 39.7)],
        crs="EPSG:4326",
    ).to_file(population_path)
    geopandas.GeoDataFrame(
        {"wp": ["cow1"], "long": [-104.8], "lat": [39.8]},
        geometry=[geometry.Point(-104.8, 39.8)],
        crs="EPSG:4326",
    ).to_file(workplace_path)
    geopandas.GeoDataFrame({"eduID": ["cos1"]}, geometry=[geometry.Point(-104.7, 39.6)], crs="EPSG:4326").to_file(
        school_path
    )
    geopandas.GeoDataFrame({"eduID": ["cod1"]}, geometry=[geometry.Point(-104.6, 39.5)], crs="EPSG:4326").to_file(
        daycare_path
    )

    population_archive = tmp_path / "co.zip"
    with ZipFile(population_archive, "w") as output:
        output.write(population_path, "co/co_population.gpkg")
        output.write(workplace_path, "co/co_workplace.gpkg")
        output.writestr("co/social_networks/co_household_network.csv", "p1,p2\n")
        output.writestr("co/social_networks/co_daycare_network.csv", "p1\n")
        output.writestr("co/social_networks/co_school_network.csv", "p2\n")
        output.writestr("co/social_networks/co_work_network.csv", ",source,target\n0,p1,p2\n")
    education_archive = tmp_path / "co_edu_sites.zip"
    with ZipFile(education_archive, "w") as output:
        output.write(school_path, school_path.name)
        output.write(daycare_path, daycare_path.name)

    manifest = build_state_tables(
        population_archive,
        "CO",
        tmp_path / "real-tables",
        education_archive=education_archive,
        batch_size=1,
    )
    assert {name: table["rows"] for name, table in manifest["tables"].items()} == {
        "places": 4,
        "hh": 1,
        "persons": 2,
        "social_networks": 1,
    }
