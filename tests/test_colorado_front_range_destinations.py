import json
from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point, box

from casmsocial.datasets.colorado_front_range import (
    BASE_CAPACITY,
    build_profile_destinations,
    load_profile,
)
from casmsocial.datasets.colorado_front_range.destination_supply import _category
from casmsocial.datasets.colorado_front_range.sources import sha256_file


def _profile():
    profile = load_profile("example-1k")
    population = profile.population.model_copy(update={"person_limit": 4, "minimum_persons_per_cbsa": 1})
    return profile.model_copy(update={"profile_id": "test-destination-profile", "population": population})


def _write_profile(root: Path) -> Path:
    root.mkdir()
    cbsa_codes = ["14540", "19740", "22660", "24540"]
    frames = {
        "places": pl.DataFrame({
            "sp_id": [101, 102, 103, 104],
            "place_type": ["Household"] * 4,
            "longitude": [0.5, 2.5, 4.5, 6.5],
            "latitude": [0.5] * 4,
            "source_state": ["CO"] * 4,
        }),
        "hh": pl.DataFrame({
            "sp_id": [11, 12, 13, 14],
            "sp_home_id": [101, 102, 103, 104],
            "hh_size": [1] * 4,
            "household_type": ["1"] * 4,
            "source_state": ["CO"] * 4,
        }),
        "persons": pl.DataFrame(
            {
                "sp_id": [1, 2, 3, 4],
                "sp_hh_id": [11, 12, 13, 14],
                "sp_work_id": [None] * 4,
                "activity_assignment_kind": [None] * 4,
                "age": [30.0] * 4,
                "gender": ["female", "male", "female", "male"],
                "assigned": [1] * 4,
                "urban": [1] * 4,
                "household_type": ["1"] * 4,
                "home_longitude": [0.5, 2.5, 4.5, 6.5],
                "home_latitude": [0.5] * 4,
                "source_state": ["CO"] * 4,
                "home_county_geoid": ["08013", "08031", "08069", "08123"],
                "home_cbsa_code": cbsa_codes,
                "age_group": ["25-64"] * 4,
            },
            schema_overrides={"sp_id": pl.Int64, "sp_hh_id": pl.Int64, "sp_work_id": pl.Int64},
        ),
        "social_networks": pl.DataFrame(
            schema={
                "person_id_a": pl.Int64,
                "person_id_b": pl.Int64,
                "network_kind": pl.String,
                "source_state": pl.String,
            }
        ),
    }
    tables = {}
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        frame.write_parquet(path)
        tables[name] = {"path": f"{name}.parquet", "rows": frame.height, "sha256": sha256_file(path)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "profile_id": _profile().profile_id,
                "profile_version": _profile().profile_version,
                "tables": tables,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _write_schedules(root: Path) -> Path:
    root.mkdir()
    rows = []
    for person_id, home_id in zip(range(1, 5), range(101, 105), strict=True):
        for day_type in ("weekday", "weekend"):
            for sequence, kind, start, end in (
                (0, "home", 0, 300),
                (1, "personal_care", 300, 400),
                (2, "shopping", 400, 500),
                (3, "travel", 500, 520),
                (4, "meal", 520, 600),
                (5, "home", 600, 1440),
            ):
                rows.append({
                    "person_id": person_id,
                    "day_type": day_type,
                    "activity_sequence": sequence,
                    "activity_kind": kind,
                    "start_minute": start,
                    "end_minute": end,
                    "place_id": None if kind == "travel" else home_id,
                    "location_source": "donor_travel_placeholder" if kind == "travel" else "home_anchor",
                    "schedule_source": "test",
                })
    schedule = pl.DataFrame(rows, schema_overrides={"person_id": pl.Int64, "place_id": pl.Int64})
    path = root / "daily_activities.parquet"
    schedule.write_parquet(path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "profile_id": _profile().profile_id,
                "profile_version": _profile().profile_version,
                "outputs": {"daily_activities": {"path": path.name, "sha256": sha256_file(path)}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


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
    gpd.GeoDataFrame(
        {"GEOID": geoids, "STATEFP": ["08"] * len(geoids)},
        geometry=[inhabited.get(geoid, box(20 + index, 0, 21 + index, 1)) for index, geoid in enumerate(geoids)],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")


def _write_osm_candidates(path: Path) -> None:
    categories = [
        {"shop": "supermarket"},
        {"shop": "convenience"},
        {"amenity": "restaurant"},
        {"amenity": "clinic"},
        {"leisure": "park"},
        {"amenity": "library"},
        {"amenity": "bank"},
    ]
    centers = [0.5, 2.5, 4.5, 6.5]
    rows = []
    geometries = []
    osm_id = 1
    for center in centers:
        for index, tags in enumerate(categories):
            rows.append({
                "osm_id": osm_id,
                "amenity": tags.get("amenity"),
                "shop": tags.get("shop"),
                "leisure": tags.get("leisure"),
                "tourism": None,
                "healthcare": None,
                "other_tags": None,
            })
            geometries.append(Point(center + index * 0.01, 0.5))
            osm_id += 1
    gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326").to_file(path, layer="points", driver="GPKG")


def test_osm_tag_mapping_and_base_capacity_contract():
    assert _category({"shop": "supermarket"}) == "shopping"
    assert _category({"amenity": "restaurant"}) == "meal"
    assert _category({"healthcare": "clinic"}) == "healthcare"
    assert _category({"leisure": "park"}) == "recreation"
    assert _category({"amenity": "library"}) == "social"
    assert _category({"amenity": "bank"}) == "errand"
    assert _category({"amenity": "bicycle_parking"}) is None
    assert BASE_CAPACITY["recreation"] == 125


def test_build_destinations_assigns_administrative_cbsas_and_event_places(tmp_path: Path):
    profile_dir = _write_profile(tmp_path / "profile")
    schedule_dir = _write_schedules(tmp_path / "schedules")
    counties = tmp_path / "counties.gpkg"
    candidates = tmp_path / "osm_candidates.gpkg"
    _write_counties(counties)
    _write_osm_candidates(candidates)
    output = tmp_path / "destinations"

    manifest = build_profile_destinations(
        profile_dir,
        schedule_dir,
        candidates,
        counties,
        _profile(),
        output,
        minimum_places_per_activity_kind=1,
    )
    resumed = build_profile_destinations(
        profile_dir,
        schedule_dir,
        candidates,
        counties,
        _profile(),
        output,
        minimum_places_per_activity_kind=1,
    )
    pois = pl.read_parquet(output / "osm_pois.parquet")
    supply = pl.read_parquet(output / "destination_supply.parquet")
    activities = pl.read_parquet(output / "daily_activities_with_destinations.parquet")

    assert manifest["status"] == "passed"
    assert manifest["assignment"]["acceptance"]["status"] == "passed"
    assert manifest["assignment"]["policy"] == "event_level_deterministic_home_grid_then_cbsa_fallback"
    assert manifest["routing_status"] == "ready_for_step_10_routing"
    assert manifest["governance"]["distribution_policy"] == "local_build_only"
    assert manifest["governance"]["redistribution_authorized"] is False
    assert manifest["governance"]["license"] == "ODbL-1.0"
    notice = (output / "OPENSTREETMAP_ATTRIBUTION.md").read_text()
    assert "© OpenStreetMap contributors" in notice
    assert "Local build only" in notice
    assert manifest["outputs"]["osm_attribution"]["sha256"] == sha256_file(output / "OPENSTREETMAP_ATTRIBUTION.md")
    assert resumed["resumed"] is True
    assert pois.height == 28
    assert set(pois["cbsa_code"]) == {"14540", "19740", "22660", "24540"}
    assert {"home", "shopping", "meal", "healthcare", "recreation", "social", "errand", "other"} <= set(
        supply["activity_kind"]
    )
    assert "travel" not in set(activities["activity_kind"])
    assert activities.filter(pl.col("activity_kind").is_in(["shopping", "meal"]))["place_id"].max() < 0
    assert set(activities["place_id"]) <= set(supply["place_id"])
    assert activities.filter(pl.col("activity_purpose") == "personal_care").select(
        "activity_kind", "location_source"
    ).unique().row(0) == ("home", "home_anchor")
    assert (
        activities.filter((pl.col("person_id") == 1) & (pl.col("activity_kind") == "shopping"))["place_id"].n_unique()
        == 2
    )

    (output / "OPENSTREETMAP_ATTRIBUTION.md").write_text("tampered\n")
    with pytest.raises(FileExistsError, match="not resumable"):
        build_profile_destinations(
            profile_dir,
            schedule_dir,
            candidates,
            counties,
            _profile(),
            output,
            minimum_places_per_activity_kind=1,
        )
