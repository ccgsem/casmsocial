import json
from pathlib import Path

import polars as pl
import pytest

from casmsocial.datasets.colorado_front_range import build_profile_schedules, load_profile
from casmsocial.datasets.colorado_front_range.atus import normalize_atus_donor_diaries
from casmsocial.datasets.colorado_front_range.sources import sha256_file


def _small_profile():
    profile = load_profile("example-1k")
    population = profile.population.model_copy(update={"person_limit": 5, "minimum_persons_per_cbsa": 1})
    return profile.model_copy(update={"profile_id": "test-schedule-profile", "population": population})


def _write_profile_product(root: Path) -> Path:
    profile = _small_profile()
    root.mkdir()
    frames = {
        "places": pl.DataFrame({
            "sp_id": [101, 102, 103, 104, 105, 201, 203, 204],
            "place_type": ["Household"] * 5 + ["Workplace", "School", "Daycare"],
            "longitude": [0.0] * 8,
            "latitude": [0.0] * 8,
            "source_state": ["CO"] * 8,
        }),
        "hh": pl.DataFrame({
            "sp_id": [11, 12, 13, 14, 15],
            "sp_home_id": [101, 102, 103, 104, 105],
            "hh_size": [1] * 5,
            "household_type": ["1"] * 5,
            "source_state": ["CO"] * 5,
        }),
        "persons": pl.DataFrame(
            {
                "sp_id": [1, 2, 3, 4, 5],
                "sp_hh_id": [11, 12, 13, 14, 15],
                "sp_work_id": [201, 999, 203, 204, None],
                "activity_assignment_kind": ["work", "work", "school", "daycare", None],
                "age": [34.0, 46.0, 9.0, 3.0, 12.0],
                "gender": ["female", "male", "female", "male", "female"],
                "assigned": [1] * 5,
                "urban": [1] * 5,
                "household_type": ["1"] * 5,
                "home_longitude": [0.0] * 5,
                "home_latitude": [0.0] * 5,
                "source_state": ["CO"] * 5,
                "home_county_geoid": ["08013"] * 5,
                "home_cbsa_code": ["14540"] * 5,
                "age_group": ["25-64", "25-64", "5-14", "0-4", "5-14"],
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
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "tables": tables,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def _write_atus_extracts(root: Path) -> tuple[Path, Path, Path]:
    respondents = root / "atusresp_2024.dat"
    activities = root / "atusact_2024.dat"
    roster = root / "atusrost_2024.dat"
    pl.DataFrame({
        "TUCASEID": [101, 102],
        "TULINENO": [1, 1],
        "TUDIARYDAY": [2, 7],
        "TUFINLWGT": [2.0, 3.0],
        "TELFS": [1, 5],
    }).write_csv(respondents)
    pl.DataFrame({
        "TUCASEID": [101] * 4 + [102] * 4,
        "TUACTIVITY_N": [1, 2, 3, 4] * 2,
        "TUSTARTTIM": ["04:00:00", "08:00:00", "16:00:00", "16:30:00"] * 2,
        "TUSTOPTIME": ["08:00:00", "16:00:00", "16:30:00", "04:00:00"] * 2,
        "TUTIER1CODE": [1, 4, 16, 11, 16, 4, 11, 16],
        "TEWHERE": [1, 2, 13, 1, 13, 2, 1, 13],
    }).write_csv(activities)
    pl.DataFrame({
        "TUCASEID": [101, 102],
        "TULINENO": [1, 1],
        "TEAGE": [34, 46],
        "TESEX": [2, 1],
    }).write_csv(roster)
    return respondents, activities, roster


def test_build_profile_schedules_covers_both_day_types_and_falls_back_unresolved_anchors(tmp_path: Path):
    profile_dir = _write_profile_product(tmp_path / "profile")
    respondents, activities, roster = _write_atus_extracts(tmp_path)
    output = tmp_path / "schedules"

    manifest = build_profile_schedules(
        profile_dir,
        respondents,
        activities,
        roster,
        _small_profile(),
        output,
        minimum_routable_minutes=5,
    )
    resumed = build_profile_schedules(
        profile_dir,
        respondents,
        activities,
        roster,
        _small_profile(),
        output,
        minimum_routable_minutes=5,
    )
    schedules = pl.read_parquet(output / "daily_activities.parquet")

    assert manifest["status"] == "passed"
    assert manifest["counts"]["persons"] == 5
    assert manifest["acceptance"]["status"] == "passed"
    assert manifest["routing_status"] == "deferred_until_destination_assignment_and_routing"
    assert resumed["resumed"] is True
    assert schedules.select("person_id", "day_type").unique().height == 10
    assert schedules.filter((pl.col("person_id") == 1) & (pl.col("activity_kind") == "work"))["place_id"].to_list() == [
        201,
        201,
    ]
    assert schedules.filter(
        (pl.col("person_id") == 1) & (pl.col("day_type") == "weekday") & (pl.col("activity_kind") == "home")
    ).is_empty()
    assert set(
        schedules.filter((pl.col("person_id") == 1) & (pl.col("location_source") == "home_anchor"))["place_id"]
    ) == {101}
    assert schedules.filter(
        (pl.col("person_id") == 1)
        & (pl.col("day_type") == "weekend")
        & (pl.col("location_source") == "boundary_travel_home_fallback")
    ).select("activity_kind", "start_minute", "end_minute", "place_id").rows() == [
        ("home", 0, 240, 101),
        ("home", 750, 1440, 101),
    ]
    assert schedules.filter((pl.col("person_id") == 2) & (pl.col("activity_kind") == "work")).is_empty()
    assert set(schedules.filter(pl.col("person_id") == 2)["place_id"].drop_nulls()) == {102}
    assert schedules.filter((pl.col("person_id") == 3) & (pl.col("activity_kind") == "school"))[
        "place_id"
    ].to_list() == [203]
    assert schedules.filter((pl.col("person_id") == 4) & (pl.col("activity_kind") == "daycare"))[
        "place_id"
    ].to_list() == [204]
    assert schedules.filter(pl.col("person_id") == 5).select("day_type", "activity_kind").rows() == [
        ("weekday", "home"),
        ("weekend", "home"),
    ]


def test_normalize_atus_diaries_repairs_gap_overlap_and_short_interval(tmp_path: Path):
    path = tmp_path / "donors.parquet"
    pl.DataFrame({
        "donor_id": ["2024:1"] * 5,
        "day_type": ["weekday"] * 5,
        "activity_sequence": [1, 2, 3, 4, 5],
        "activity_kind": ["home", "work", "travel", "social", "home"],
        "start_minute": [0, 110, 290, 400, 405],
        "end_minute": [100, 300, 400, 405, 1440],
        "atus_activity_code": [1, 5, 18, 12, 1],
        "atus_location_code": [1, 2, 13, 1, 1],
        "diary_weight": [1.0] * 5,
        "age": [34] * 5,
        "sex_code": [1] * 5,
        "labor_force_status": [1] * 5,
    }).write_parquet(path)

    manifest = normalize_atus_donor_diaries(path, tmp_path / "normalized", minimum_routable_minutes=10)
    result = pl.read_parquet(tmp_path / "normalized" / "atus_donor_activities_normalized.parquet")

    assert result["start_minute"].to_list() == [0, 100, 110, 300, 405]
    assert result["end_minute"].to_list() == [100, 110, 300, 405, 1440]
    assert manifest["normalization"]["compacted_short_intervals"] == 1


def test_schedule_builder_rejects_tampered_profile_table(tmp_path: Path):
    profile_dir = _write_profile_product(tmp_path / "profile")
    respondents, activities, roster = _write_atus_extracts(tmp_path)
    with (profile_dir / "persons.parquet").open("ab") as target:
        target.write(b"changed")

    with pytest.raises(ValueError, match="does not match its manifest: persons"):
        build_profile_schedules(
            profile_dir,
            respondents,
            activities,
            roster,
            _small_profile(),
            tmp_path / "schedules",
        )
