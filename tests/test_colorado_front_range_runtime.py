import json
from pathlib import Path

import polars as pl

from casmsocial.datasets.colorado_front_range import build_profile_runtime, load_profile
from casmsocial.datasets.colorado_front_range.sources import sha256_file
from casmsocial.ducklake_utils import get_ducklake_connection


def _profile():
    profile = load_profile("example-1k")
    population = profile.population.model_copy(update={"person_limit": 2, "minimum_persons_per_cbsa": 1})
    return profile.model_copy(update={"profile_id": "test-runtime-profile", "population": population})


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    profile_dir = tmp_path / "profile"
    destination_dir = tmp_path / "destinations"
    profile_dir.mkdir()
    destination_dir.mkdir()
    tables = {
        "places": pl.DataFrame({
            "sp_id": [101, 102],
            "place_type": ["Household"] * 2,
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "source_state": ["CO"] * 2,
        }),
        "hh": pl.DataFrame({
            "sp_id": [11, 12],
            "sp_home_id": [101, 102],
            "hh_size": [1, 1],
            "household_type": ["1", "1"],
            "source_state": ["CO", "CO"],
        }),
        "persons": pl.DataFrame(
            {
                "sp_id": [1, 2],
                "sp_hh_id": [11, 12],
                "sp_work_id": [None, None],
                "activity_assignment_kind": [None, None],
                "age": [30.0, 40.0],
                "gender": ["female", "male"],
                "assigned": [1, 1],
                "urban": [1, 1],
                "household_type": ["1", "1"],
                "home_longitude": [0.0, 1.0],
                "home_latitude": [0.0, 1.0],
                "source_state": ["CO", "CO"],
                "home_county_geoid": ["08013", "08031"],
                "home_cbsa_code": ["14540", "19740"],
                "age_group": ["25-64", "25-64"],
            },
            schema_overrides={"sp_id": pl.Int64, "sp_hh_id": pl.Int64, "sp_work_id": pl.Int64},
        ),
        "social_networks": pl.DataFrame({
            "person_id_a": [1],
            "person_id_b": [2],
            "network_kind": ["social"],
            "source_state": ["CO"],
        }),
    }
    manifested = {}
    for name, frame in tables.items():
        path = profile_dir / f"{name}.parquet"
        frame.write_parquet(path)
        manifested[name] = {"rows": frame.height, "sha256": sha256_file(path)}
    (profile_dir / "manifest.json").write_text(
        json.dumps({
            "status": "passed",
            "profile_id": _profile().profile_id,
            "profile_version": _profile().profile_version,
            "tables": manifested,
        })
    )

    supply = pl.DataFrame({
        "place_id": [101, 102, -1, -2],
        "activity_kind": ["home", "home", "shopping", "meal"],
        "latitude": [0.0, 1.0, 0.01, 0.02],
        "longitude": [0.0, 1.0, 0.01, 0.02],
        "cbsa_code": ["14540", "19740", "14540", "14540"],
        "base_capacity": [1, 1, 100, 100],
        "capacity": [1, 1, 1, 100],
        "supply_source": ["synthetic_home", "synthetic_home", "osm", "osm"],
    })
    activities = []
    for person, home in ((1, 101), (2, 102)):
        for day in ("weekday", "weekend"):
            activities.extend([
                (person, day, 0, "home", "home", 0, 300 if person == 1 else 2, home, "home_anchor", "test"),
                (
                    person,
                    day,
                    1,
                    "shopping",
                    "shopping",
                    300 if person == 1 else 2,
                    500,
                    -1,
                    "sampled_destination",
                    "test",
                ),
                (person, day, 2, "meal", "meal", 500, 700, -2, "sampled_destination", "test"),
                (person, day, 3, "home", "home", 700, 1440, home, "home_anchor", "test"),
            ])
    activity_frame = pl.DataFrame(
        activities,
        schema=[
            "person_id",
            "day_type",
            "activity_sequence",
            "activity_kind",
            "activity_purpose",
            "start_minute",
            "end_minute",
            "place_id",
            "location_source",
            "schedule_source",
        ],
        orient="row",
    ).with_columns(pl.col("person_id", "place_id").cast(pl.Int64))
    supply_path = destination_dir / "destination_supply.parquet"
    activity_path = destination_dir / "daily_activities_with_destinations.parquet"
    supply.write_parquet(supply_path)
    activity_frame.write_parquet(activity_path)
    (destination_dir / "manifest.json").write_text(
        json.dumps({
            "status": "passed",
            "profile_id": _profile().profile_id,
            "profile_version": _profile().profile_version,
            "outputs": {
                "destination_supply": {"sha256": sha256_file(supply_path)},
                "daily_activities": {"sha256": sha256_file(activity_path)},
            },
        })
    )
    return profile_dir, destination_dir


def test_build_runtime_routes_repairs_exports_catalogs_and_resumes(tmp_path: Path):
    profile_dir, destination_dir = _inputs(tmp_path)
    output = tmp_path / "runtime"
    manifest = build_profile_runtime(profile_dir, destination_dir, _profile(), output)
    resumed = build_profile_runtime(profile_dir, destination_dir, _profile(), output)

    assert manifest["status"] == "passed"
    assert manifest["acceptance"]["status"] == "passed"
    assert manifest["routing"]["retained_location_infeasible_trip"] > 0
    assert manifest["destination_resolution"] == "event_place_id"
    assert resumed["resumed"] is True
    exported = pl.read_parquet(output / "casmsocial" / "activities.parquet")
    assert exported.columns == [
        "sp_persons_id",
        "activity_id",
        "activity_sequence",
        "starttime_min",
        "endtime_min",
        "sp_act_id",
    ]
    connection = get_ducklake_connection(output / "ducklake")
    try:
        assert connection.execute("SELECT count(*) FROM colorado_front_range.persons").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM colorado_front_range.social_networks").fetchone()[0] == 1
    finally:
        connection.close()
