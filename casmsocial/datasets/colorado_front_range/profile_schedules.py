"""Build profile-scoped weekday and weekend activity schedules.

Derived from the mydatalakehouse Colorado Front Range schedule builder at
commit 00380e58c1a33449d07bd346ce6c0df3eb6ceaf1. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from casmsocial.datasets.colorado_front_range.atus import (
    assign_atus_donors,
    normalize_atus_donor_diaries,
    stage_atus_donor_diaries,
)
from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile
from casmsocial.datasets.colorado_front_range.sources import sha256_file

ACTIVITY_KINDS = {
    "home",
    "work",
    "school",
    "daycare",
    "travel",
    "shopping",
    "meal",
    "personal_care",
    "social",
    "recreation",
    "healthcare",
    "errand",
    "other",
}
OUTPUT_FILES = {
    "staged_donors": "donors/atus_donor_activities.parquet",
    "normalized_donors": "donors/atus_donor_activities_normalized.parquet",
    "donor_assignments": "atus_donor_assignments.parquet",
    "daily_activities": "daily_activities.parquet",
}
PROFILE_TABLES = ("places", "hh", "persons", "social_networks")
SCHEDULE_CONTRACT_VERSION = 2


def _sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def _write_json_atomic(path: Path, content: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_profile_product(profile_dir: Path, profile: ColoradoDatasetProfile) -> dict[str, object]:
    manifest_path = profile_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValueError("Profile population does not have an accepted manifest")
    if manifest.get("profile_id") != profile.profile_id or manifest.get("profile_version") != profile.profile_version:
        raise ValueError("Profile population manifest does not match the requested profile")
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Profile population manifest has no table inventory")
    for name in PROFILE_TABLES:
        table = tables.get(name)
        path = profile_dir / f"{name}.parquet"
        if not isinstance(table, dict) or not path.is_file() or table.get("sha256") != sha256_file(path):
            raise ValueError(f"Profile population table does not match its manifest: {name}")
    return manifest


def _input_fingerprint(
    profile_dir: Path,
    profile: ColoradoDatasetProfile,
    respondents: Path,
    activities: Path,
    roster: Path,
    source_year: int,
    minimum_routable_minutes: int,
) -> dict[str, object]:
    return {
        "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
        "profile_population": str(profile_dir),
        "profile_manifest_sha256": sha256_file(profile_dir / "manifest.json"),
        "profile": profile.model_dump(mode="json"),
        "atus": {
            "source_year": source_year,
            "respondents": {"path": str(respondents), "sha256": sha256_file(respondents)},
            "activities": {"path": str(activities), "sha256": sha256_file(activities)},
            "roster": {"path": str(roster), "sha256": sha256_file(roster)},
        },
        "minimum_routable_minutes": minimum_routable_minutes,
    }


def _resumable_output(output_dir: Path, fingerprint: dict[str, object]) -> dict[str, object] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed" or manifest.get("inputs") != fingerprint:
        return None
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for name, relative_path in OUTPUT_FILES.items():
        artifact = outputs.get(name)
        path = output_dir / relative_path
        if not isinstance(artifact, dict) or not path.is_file() or artifact.get("sha256") != sha256_file(path):
            return None
    return {**manifest, "resumed": True}


def _write_adult_matching_people(profile_dir: Path, destination: Path) -> int:
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT sp_id::BIGINT person_id, age::BIGINT age, "
            "CASE lower(gender) WHEN 'm' THEN 1 WHEN 'male' THEN 1 "
            "WHEN 'f' THEN 2 WHEN 'female' THEN 2 ELSE 0 END::BIGINT sex_code, "
            f"0::BIGINT labor_force_status FROM read_parquet('{_sql_path(profile_dir / 'persons.parquet')}') "
            f"WHERE age >= 15 ORDER BY sp_id) TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        return connection.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(destination)}')").fetchone()[0]
    finally:
        connection.close()


def _materialize_daily_activities(
    profile_dir: Path,
    normalized_donors: Path,
    assignments: Path,
    destination: Path,
) -> None:
    people_path = profile_dir / "persons.parquet"
    households_path = profile_dir / "hh.parquet"
    places_path = profile_dir / "places.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (WITH people AS (SELECT person.sp_id::BIGINT person_id, "
            "household.sp_home_id::BIGINT home_place_id, "
            "CASE WHEN anchor.sp_id IS NOT NULL THEN person.sp_work_id ELSE NULL END::BIGINT activity_place_id, "
            "person.activity_assignment_kind, person.age::BIGINT age "
            f"FROM read_parquet('{_sql_path(people_path)}') person "
            f"JOIN read_parquet('{_sql_path(households_path)}') household ON person.sp_hh_id = household.sp_id "
            f"LEFT JOIN read_parquet('{_sql_path(places_path)}') anchor ON person.sp_work_id = anchor.sp_id), "
            "adults AS (SELECT people.person_id, assignment.day_type, donor.activity_sequence::BIGINT activity_sequence, "
            "CASE WHEN donor.activity_kind = 'travel' AND (donor.start_minute = 0 OR donor.end_minute = 1440) "
            "THEN 'home' WHEN donor.activity_kind = 'work' AND NOT (people.activity_assignment_kind = 'work' "
            "AND people.activity_place_id IS NOT NULL) THEN 'home' "
            "WHEN donor.activity_kind = 'school' AND NOT (people.activity_assignment_kind = 'school' "
            "AND people.activity_place_id IS NOT NULL) THEN 'home' ELSE donor.activity_kind END activity_kind, "
            "donor.start_minute::BIGINT start_minute, donor.end_minute::BIGINT end_minute, "
            "CASE WHEN donor.activity_kind = 'travel' AND donor.start_minute <> 0 AND donor.end_minute <> 1440 THEN NULL "
            "WHEN donor.activity_kind = 'work' AND people.activity_assignment_kind = 'work' "
            "AND people.activity_place_id IS NOT NULL THEN people.activity_place_id "
            "WHEN donor.activity_kind = 'school' AND people.activity_assignment_kind = 'school' "
            "AND people.activity_place_id IS NOT NULL THEN people.activity_place_id "
            "ELSE people.home_place_id END::BIGINT place_id, "
            "CASE WHEN donor.activity_kind = 'travel' AND (donor.start_minute = 0 OR donor.end_minute = 1440) "
            "THEN 'boundary_travel_home_fallback' WHEN donor.activity_kind = 'travel' THEN 'donor_travel_placeholder' "
            "WHEN donor.activity_kind = 'work' AND people.activity_assignment_kind = 'work' "
            "AND people.activity_place_id IS NOT NULL THEN 'work_anchor' "
            "WHEN donor.activity_kind = 'school' AND people.activity_assignment_kind = 'school' "
            "AND people.activity_place_id IS NOT NULL THEN 'school_anchor' ELSE 'home_anchor' END location_source, "
            "'atus_donor' schedule_source FROM people "
            f"JOIN read_parquet('{_sql_path(assignments)}') assignment USING (person_id) "
            f"JOIN read_parquet('{_sql_path(normalized_donors)}') donor "
            "ON assignment.donor_id = donor.donor_id AND assignment.day_type = donor.day_type), "
            "children AS (SELECT * FROM people WHERE age BETWEEN 0 AND 14), "
            "child_rows AS (SELECT person_id, 'weekday' day_type, 0::BIGINT activity_sequence, 'home' activity_kind, "
            "0::BIGINT start_minute, CASE WHEN age <= 4 AND activity_assignment_kind = 'daycare' "
            "AND activity_place_id IS NOT NULL THEN 480 WHEN age BETWEEN 5 AND 14 "
            "AND activity_assignment_kind = 'school' AND activity_place_id IS NOT NULL THEN 510 ELSE 1440 END::BIGINT end_minute, "
            "home_place_id place_id, 'home_anchor' location_source, 'rule_based' schedule_source FROM children "
            "UNION ALL SELECT person_id, 'weekday', 1::BIGINT, CASE WHEN age <= 4 THEN 'daycare' ELSE 'school' END, "
            "CASE WHEN age <= 4 THEN 480 ELSE 510 END::BIGINT, CASE WHEN age <= 4 THEN 1020 ELSE 930 END::BIGINT, "
            "activity_place_id, CASE WHEN age <= 4 THEN 'daycare_anchor' ELSE 'school_anchor' END, 'school_calendar' "
            "FROM children WHERE (age <= 4 AND activity_assignment_kind = 'daycare' AND activity_place_id IS NOT NULL) "
            "OR (age BETWEEN 5 AND 14 AND activity_assignment_kind = 'school' AND activity_place_id IS NOT NULL) "
            "UNION ALL SELECT person_id, 'weekday', 2::BIGINT, 'home', "
            "CASE WHEN age <= 4 THEN 1020 ELSE 930 END::BIGINT, 1440::BIGINT, home_place_id, "
            "'home_anchor', 'rule_based' FROM children WHERE (age <= 4 AND activity_assignment_kind = 'daycare' "
            "AND activity_place_id IS NOT NULL) OR (age BETWEEN 5 AND 14 AND activity_assignment_kind = 'school' "
            "AND activity_place_id IS NOT NULL) UNION ALL SELECT person_id, 'weekend', 0::BIGINT, 'home', "
            "0::BIGINT, 1440::BIGINT, home_place_id, 'home_anchor', 'rule_based' FROM children) "
            "SELECT * FROM adults UNION ALL SELECT * FROM child_rows ORDER BY person_id, day_type, activity_sequence) "
            f"TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()


def _schedule_acceptance(
    profile_dir: Path,
    schedule_path: Path,
    expected_persons: int,
    require_exactly_one_weekday_home: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    connection = duckdb.connect()
    supported = ", ".join(f"'{kind}'" for kind in sorted(ACTIVITY_KINDS))
    try:
        schedule = f"read_parquet('{_sql_path(schedule_path)}')"
        places = f"read_parquet('{_sql_path(profile_dir / 'places.parquet')}')"
        checks = {
            "duplicate_sequence_rows": connection.execute(
                f"SELECT count(*) FROM (SELECT person_id, day_type, activity_sequence, count(*) records FROM {schedule} "
                "GROUP BY ALL HAVING records > 1)"
            ).fetchone()[0],
            "unsupported_day_types": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE day_type NOT IN ('weekday', 'weekend')"
            ).fetchone()[0],
            "unsupported_activity_kinds": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE activity_kind NOT IN ({supported})"
            ).fetchone()[0],
            "invalid_intervals": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE start_minute < 0 OR end_minute > 1440 "
                "OR start_minute >= end_minute"
            ).fetchone()[0],
            "travel_rows_with_place": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE activity_kind = 'travel' AND place_id IS NOT NULL"
            ).fetchone()[0],
            "boundary_travel_placeholders": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE activity_kind = 'travel' "
                "AND (start_minute = 0 OR end_minute = 1440)"
            ).fetchone()[0],
            "stationary_rows_without_place": connection.execute(
                f"SELECT count(*) FROM {schedule} WHERE activity_kind <> 'travel' AND place_id IS NULL"
            ).fetchone()[0],
            "stationary_rows_without_profile_place": connection.execute(
                f"SELECT count(*) FROM {schedule} activity ANTI JOIN {places} place ON activity.place_id = place.sp_id "
                "WHERE activity.activity_kind <> 'travel'"
            ).fetchone()[0],
            "missing_person_day_types": connection.execute(
                f"SELECT {expected_persons * 2} - count(*) FROM (SELECT DISTINCT person_id, day_type FROM {schedule})"
            ).fetchone()[0],
            "noncontiguous_plans": connection.execute(
                f"WITH ordered AS (SELECT *, lag(end_minute) OVER (PARTITION BY person_id, day_type "
                "ORDER BY activity_sequence) previous_end, row_number() OVER (PARTITION BY person_id, day_type "
                "ORDER BY activity_sequence) forward_rank, row_number() OVER (PARTITION BY person_id, day_type "
                f"ORDER BY activity_sequence DESC) reverse_rank FROM {schedule}) SELECT count(*) FROM ordered "
                "WHERE (forward_rank = 1 AND start_minute <> 0) OR (forward_rank > 1 AND start_minute <> previous_end) "
                "OR (reverse_rank = 1 AND end_minute <> 1440)"
            ).fetchone()[0],
            "persons_without_exactly_one_weekday_home": connection.execute(
                f"SELECT count(*) FROM (SELECT person_id, count(DISTINCT CASE WHEN day_type = 'weekday' "
                f"AND location_source = 'home_anchor' THEN place_id END) homes FROM {schedule} "
                "GROUP BY person_id) WHERE homes <> 1"
            ).fetchone()[0],
        }
        required_zero = [
            name
            for name in checks
            if require_exactly_one_weekday_home or name != "persons_without_exactly_one_weekday_home"
        ]
        counts_row = connection.execute(
            f"SELECT count(*) AS row_count, count(DISTINCT person_id) AS persons, "
            f"sum(CASE WHEN activity_kind = 'travel' THEN 1 ELSE 0 END) travel_placeholders FROM {schedule}"
        ).fetchone()
        counts = {
            "rows": counts_row[0],
            "persons": counts_row[1],
            "travel_placeholders": counts_row[2],
            "by_day_type": [
                {"day_type": row[0], "rows": row[1]}
                for row in connection.execute(
                    f"SELECT day_type, count(*) FROM {schedule} GROUP BY day_type ORDER BY day_type"
                ).fetchall()
            ],
            "by_schedule_source": [
                {"schedule_source": row[0], "rows": row[1]}
                for row in connection.execute(
                    f"SELECT schedule_source, count(*) FROM {schedule} GROUP BY schedule_source ORDER BY schedule_source"
                ).fetchall()
            ],
        }
    finally:
        connection.close()
    if counts["persons"] != expected_persons:
        raise ValueError(f"Schedule contains {counts['persons']} people, expected {expected_persons}")
    acceptance = {
        "status": "passed" if all(checks[name] == 0 for name in required_zero) else "failed",
        "required_zero": required_zero,
        "checks": checks,
    }
    if acceptance["status"] != "passed":
        raise ValueError(f"Profile schedule acceptance failed: {checks}")
    return acceptance, counts


def build_profile_schedules(
    profile_dir: Path,
    respondents_path: Path,
    activities_path: Path,
    roster_path: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
    *,
    source_year: int = 2024,
    minimum_routable_minutes: int = 10,
    allow_planned: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build an accepted and resumable weekday/weekend pre-routing schedule."""
    profile_dir = profile_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    respondents_path, activities_path, roster_path = (
        path.expanduser().resolve() for path in (respondents_path, activities_path, roster_path)
    )
    _validate_profile_product(profile_dir, profile)
    for path in (respondents_path, activities_path, roster_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if profile.release_status == "planned" and not allow_planned:
        raise ValueError(
            f"Profile {profile.profile_id} is planned; pass --allow-planned to build exploratory schedules"
        )
    fingerprint = _input_fingerprint(
        profile_dir,
        profile,
        respondents_path,
        activities_path,
        roster_path,
        source_year,
        minimum_routable_minutes,
    )
    if resumed := _resumable_output(output_dir, fingerprint):
        return resumed
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Schedule output exists but is not resumable; use --overwrite: {output_dir}")
    staging = output_dir.with_name(f"{output_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "donors").mkdir(parents=True)

    try:
        staged = stage_atus_donor_diaries(
            respondents_path,
            activities_path,
            roster_path,
            staging / "donors",
            source_year=source_year,
        )
        normalized = normalize_atus_donor_diaries(
            staging / OUTPUT_FILES["staged_donors"],
            staging / "donors",
            minimum_routable_minutes=minimum_routable_minutes,
        )
        matching_path = staging / "adult_matching_persons.parquet"
        adult_count = _write_adult_matching_people(profile_dir, matching_path)
        assignments = assign_atus_donors(
            matching_path,
            staging / OUTPUT_FILES["normalized_donors"],
            staging,
            random_seed=profile.schedules.donor_assignment_seed,
        )
        matching_path.unlink()
        schedule_path = staging / OUTPUT_FILES["daily_activities"]
        _materialize_daily_activities(
            profile_dir,
            staging / OUTPUT_FILES["normalized_donors"],
            staging / OUTPUT_FILES["donor_assignments"],
            schedule_path,
        )
        profile_manifest = json.loads((profile_dir / "manifest.json").read_text(encoding="utf-8"))
        expected_persons = int(profile_manifest["tables"]["persons"]["rows"])
        acceptance, counts = _schedule_acceptance(
            profile_dir,
            schedule_path,
            expected_persons,
            profile.population.require_exactly_one_weekday_home,
        )
        outputs = {
            name: {"path": relative_path, "sha256": sha256_file(staging / relative_path)}
            for name, relative_path in OUTPUT_FILES.items()
        }
        outputs["daily_activities"]["rows"] = counts["rows"]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "resumed": False,
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "release_status": profile.release_status,
            "diary_day_start_local_time": profile.schedules.diary_day_start_local_time,
            "inputs": fingerprint,
            "staging": staged,
            "normalization": normalized,
            "donor_assignment": {**assignments, "adult_persons": adult_count},
            "counts": counts,
            "acceptance": acceptance,
            "outputs": outputs,
            "routing_status": "deferred_until_destination_assignment_and_routing",
            "limitations": [
                "Day types are representative ATUS plans, not observed longitudinal trajectories.",
                "Travel intervals are placeholders until routing.",
                "Discretionary adult activities remain at home until destination assignment.",
                "Unresolved OSF work, school, and daycare anchors fall back to home.",
            ],
        }
        _write_json_atomic(staging / "manifest.json", manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
