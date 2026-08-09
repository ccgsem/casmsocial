"""Route, validate, export, and catalog a profile-scoped CASMSocial runtime.

Routing and export logic is derived from the mydatalakehouse Colorado Front
Range routing, feasibility, and CASMSocial export modules. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import duckdb
import polars as pl
import yaml

from casmsocial.datasets.colorado_front_range.profile_schedules import _validate_profile_product
from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile
from casmsocial.datasets.colorado_front_range.sources import sha256_file
from casmsocial.ducklake_utils import get_ducklake_connection

ACTIVITY_IDS = {
    "home": 0,
    "work": 1,
    "school": 2,
    "daycare": 3,
    "shopping": 4,
    "meal": 5,
    "personal_care": 6,
    "social": 7,
    "recreation": 8,
    "healthcare": 9,
    "errand": 10,
    "other": 11,
}
ANCHOR_COLUMNS = {
    "home": "sp_hh_id",
    "work": "sp_work_id",
    "school": "sp_school_id",
    "daycare": "sp_daycare_id",
    "shopping": "sp_shopping_id",
    "meal": "sp_meal_id",
    "personal_care": "sp_personal_care_id",
    "social": "sp_social_id",
    "recreation": "sp_recreation_id",
    "healthcare": "sp_healthcare_id",
    "errand": "sp_errand_id",
    "other": "sp_other_id",
}
RUNTIME_TABLES = ("activities", "persons", "hh", "places", "social_networks")
SCHEMA_NAME = "colorado_front_range"
PARTITION_TABLE = "partitions.colorado_front_range_place_partitions"


def _sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tree_sha256(root: Path) -> str:
    digest = __import__("hashlib").sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _validate_destination_product(destination_dir: Path, profile: ColoradoDatasetProfile) -> dict[str, object]:
    manifest_path = destination_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    required = {
        "destination_supply": "destination_supply.parquet",
        "daily_activities": "daily_activities_with_destinations.parquet",
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "passed"
        or manifest.get("profile_id") != profile.profile_id
        or manifest.get("profile_version") != profile.profile_version
    ):
        raise ValueError("Destination product does not have a matching accepted manifest")
    for name, relative in required.items():
        artifact = outputs.get(name, {}) if isinstance(outputs, dict) else {}
        path = destination_dir / relative
        if not isinstance(artifact, dict) or not path.is_file() or artifact.get("sha256") != sha256_file(path):
            raise ValueError(f"Destination product output does not match its manifest: {name}")
    return manifest


def _fingerprint(
    profile_dir: Path,
    destination_dir: Path,
    profile: ColoradoDatasetProfile,
) -> dict[str, object]:
    return {
        "runtime_contract_version": 2,
        "profile_manifest_sha256": sha256_file(profile_dir / "manifest.json"),
        "destination_manifest_sha256": sha256_file(destination_dir / "manifest.json"),
        "profile": profile.model_dump(mode="json"),
        "routing_algorithm": "partitioned_sequential_straight_line_retain_unreachable_v1",
        "runtime_day_type": profile.runtime.default_day_type,
    }


def _resumable(output_dir: Path, fingerprint: dict[str, object]) -> dict[str, object] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed" or manifest.get("inputs") != fingerprint:
        return None
    outputs = manifest.get("outputs", {})
    for relative in (
        "daily_activities_routed.parquet",
        "casmsocial/activities.parquet",
        "casmsocial/persons.parquet",
        "casmsocial/hh.parquet",
        "casmsocial/places.parquet",
        "casmsocial/social_networks.parquet",
        "casmsocial.yaml",
    ):
        artifact = outputs.get(relative, {}) if isinstance(outputs, dict) else {}
        path = output_dir / relative
        if not isinstance(artifact, dict) or not path.is_file() or artifact.get("sha256") != sha256_file(path):
            return None
    ducklake = output_dir / "ducklake"
    if not ducklake.is_dir() or manifest.get("ducklake", {}).get("tree_sha256") != _tree_sha256(ducklake):
        return None
    return {**manifest, "resumed": True}


def _minutes(
    first: tuple[float, float], second: tuple[float, float], profile: ColoradoDatasetProfile
) -> tuple[int, float]:
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    haversine = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    distance = 6371.0088 * 2 * math.asin(math.sqrt(haversine))
    minutes = math.ceil(distance / profile.routing.average_speed_kph * 60)
    return min(profile.routing.maximum_minutes, max(profile.routing.minimum_minutes, minutes)), distance


def _route_partition(
    source: Path,
    destination: Path,
    coordinates: dict[int, tuple[float, float]],
    profile: ColoradoDatasetProfile,
) -> dict[str, object]:
    frame = pl.read_parquet(source).sort("person_id", "day_type", "activity_sequence")
    rows: list[dict[str, object]] = []
    retained = travel_legs = 0
    distance_km = 0.0
    for _, plan in frame.group_by(["person_id", "day_type"], maintain_order=True):
        items = plan.sort("activity_sequence").to_dicts()
        routed = [items[0].copy()]
        for desired_source in items[1:]:
            desired = desired_source.copy()
            previous = routed[-1]
            if previous["place_id"] != desired["place_id"]:
                duration, distance = _minutes(
                    coordinates[previous["place_id"]], coordinates[desired["place_id"]], profile
                )
                available = int(previous["end_minute"]) - int(previous["start_minute"])
                if duration >= available:
                    desired["activity_purpose"] = desired.get("activity_purpose", desired["activity_kind"])
                    desired["place_id"] = previous["place_id"]
                    desired["activity_kind"] = previous["activity_kind"]
                    desired["location_source"] = "retained_location_infeasible_trip"
                    retained += 1
                else:
                    travel_start = int(desired["start_minute"]) - duration
                    previous["end_minute"] = travel_start
                    routed.append({
                        **previous,
                        "activity_kind": "travel",
                        "activity_purpose": "travel",
                        "start_minute": travel_start,
                        "end_minute": desired["start_minute"],
                        "place_id": None,
                        "location_source": "straight_line_proxy",
                        "schedule_source": "routing_proxy",
                    })
                    travel_legs += 1
                    distance_km += distance
            routed.append(desired)
        for sequence, row in enumerate(routed):
            row["activity_sequence"] = sequence
            rows.append(row)
    output = (
        pl
        .DataFrame(rows)
        .with_columns(
            pl.col("person_id").cast(pl.Int64),
            pl.col("activity_sequence").cast(pl.Int64),
            pl.col("start_minute").cast(pl.Int64),
            pl.col("end_minute").cast(pl.Int64),
            pl.col("place_id").cast(pl.Int64),
        )
        .sort("person_id", "day_type", "activity_sequence")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(destination, compression="zstd")
    return {
        "rows": output.height,
        "persons": output["person_id"].n_unique(),
        "travel_legs": travel_legs,
        "retained_location_infeasible_trip": retained,
        "total_straight_line_distance_km": round(distance_km, 3),
    }


def _source_partitions(source: Path, staging: Path, partitions: int) -> list[Path]:
    root = staging / "stationary_partitions"
    manifest_path = root / "manifest.json"
    fingerprint = {"source_sha256": sha256_file(source), "partitions": partitions}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [root / item["path"] for item in manifest.get("files", [])]
        if manifest.get("inputs") == fingerprint and all(
            path.is_file() and item["sha256"] == sha256_file(path)
            for path, item in zip(paths, manifest.get("files", []), strict=True)
        ):
            return paths
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT *, CAST(hash(person_id, 'routing') % {partitions} AS INTEGER) person_bucket "
            f"FROM read_parquet('{_sql_path(source)}')) TO '{_sql_path(root)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (person_bucket))"
        )
    finally:
        connection.close()
    paths = sorted(root.glob("person_bucket=*/*.parquet"))
    files = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in paths]
    _write_json(manifest_path, {"inputs": fingerprint, "files": files})
    return paths


def _route_partitions(
    sources: list[Path],
    staging: Path,
    supply_path: Path,
    profile: ColoradoDatasetProfile,
) -> tuple[list[Path], dict[str, object]]:
    supply_hash = sha256_file(supply_path)
    supply = pl.read_parquet(supply_path).select("place_id", "latitude", "longitude")
    coordinates = {row["place_id"]: (row["latitude"], row["longitude"]) for row in supply.iter_rows(named=True)}
    outputs = []
    reports = []
    root = staging / "routing_partitions"
    root.mkdir(exist_ok=True)
    rules = profile.routing.model_dump(mode="json")
    for index, source in enumerate(sources):
        output = root / f"part-{index:04d}.parquet"
        manifest_path = root / f"part-{index:04d}.manifest.json"
        inputs = {"source_sha256": sha256_file(source), "supply_sha256": supply_hash, "rules": rules}
        if manifest_path.is_file() and output.is_file():
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("inputs") == inputs and cached.get("output_sha256") == sha256_file(output):
                outputs.append(output)
                reports.append(cached["report"])
                continue
        report = _route_partition(source, output, coordinates, profile)
        _write_json(
            manifest_path,
            {"inputs": inputs, "output_sha256": sha256_file(output), "report": report},
        )
        outputs.append(output)
        reports.append(report)
    totals = {
        key: sum(report[key] for report in reports)
        for key in ("rows", "travel_legs", "retained_location_infeasible_trip", "total_straight_line_distance_km")
    }
    totals["partitions"] = len(outputs)
    return outputs, totals


def _combine_and_validate(
    parts: list[Path],
    supply_path: Path,
    destination: Path,
    profile: ColoradoDatasetProfile,
) -> tuple[dict[str, object], dict[str, object]]:
    paths = ", ".join(f"'{_sql_path(path)}'" for path in parts)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT * FROM read_parquet([{paths}]) ORDER BY person_id, day_type, activity_sequence) "
            f"TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        output = f"read_parquet('{_sql_path(destination)}')"
        supply = f"read_parquet('{_sql_path(supply_path)}')"
        checks = {
            "noncontiguous_transitions": connection.execute(
                f"WITH rows AS (SELECT *, lead(start_minute) OVER (PARTITION BY person_id, day_type "
                f"ORDER BY activity_sequence) next_start FROM {output}) SELECT count(*) FROM rows "
                "WHERE next_start IS NOT NULL AND end_minute <> next_start"
            ).fetchone()[0],
            "invalid_intervals": connection.execute(
                f"SELECT count(*) FROM {output} WHERE start_minute < 0 OR end_minute > 1440 OR start_minute >= end_minute"
            ).fetchone()[0],
            "unknown_stationary_places": connection.execute(
                f"SELECT count(*) FROM {output} activity ANTI JOIN {supply} place ON activity.place_id = place.place_id "
                "WHERE activity.activity_kind <> 'travel'"
            ).fetchone()[0],
            "mismatched_stationary_place_kinds": connection.execute(
                f"SELECT count(*) FROM {output} activity JOIN {supply} place ON activity.place_id = place.place_id "
                "WHERE activity.activity_kind <> 'travel' AND activity.activity_kind <> place.activity_kind"
            ).fetchone()[0],
            "travel_rows_with_places": connection.execute(
                f"SELECT count(*) FROM {output} WHERE activity_kind = 'travel' AND place_id IS NOT NULL"
            ).fetchone()[0],
            "capacity_exceedances": connection.execute(
                f"WITH events AS (SELECT day_type, place_id, start_minute event_minute, 1 delta FROM {output} "
                "WHERE activity_kind NOT IN ('travel', 'home') UNION ALL SELECT day_type, place_id, end_minute, -1 FROM "
                f"{output} WHERE activity_kind NOT IN ('travel', 'home')), minute_delta AS (SELECT day_type, place_id, "
                "event_minute, sum(delta) delta FROM events GROUP BY day_type, place_id, event_minute), occupancy AS "
                "(SELECT day_type, place_id, sum(delta) OVER (PARTITION BY day_type, place_id ORDER BY event_minute) "
                "occupancy FROM minute_delta), peaks AS (SELECT day_type, place_id, max(occupancy) peak FROM occupancy "
                f"GROUP BY day_type, place_id) SELECT count(*) FROM peaks "
                f"JOIN {supply} USING (place_id) WHERE peak > capacity"
            ).fetchone()[0],
        }
        limits = {
            "noncontiguous_transitions": profile.validation.maximum_noncontiguous_transitions,
            "invalid_intervals": profile.validation.maximum_invalid_intervals,
            "unknown_stationary_places": profile.validation.maximum_unknown_stationary_places,
            "capacity_exceedances": profile.validation.maximum_capacity_exceedances,
            "mismatched_stationary_place_kinds": 0,
            "travel_rows_with_places": 0,
        }
        counts = {
            "rows": connection.execute(f"SELECT count(*) FROM {output}").fetchone()[0],
            "persons": connection.execute(f"SELECT count(DISTINCT person_id) FROM {output}").fetchone()[0],
            "travel_legs": connection.execute(
                f"SELECT count(*) FROM {output} WHERE activity_kind = 'travel'"
            ).fetchone()[0],
        }
    finally:
        connection.close()
    status = "passed" if all(checks[name] <= limit for name, limit in limits.items()) else "failed"
    acceptance = {"status": status, "limits": limits, "checks": checks}
    if status != "passed":
        raise ValueError(f"Routed schedule acceptance failed: {acceptance}")
    return acceptance, counts


def _export_runtime_tables(
    routed_path: Path,
    supply_path: Path,
    profile_dir: Path,
    output_dir: Path,
    day_type: str,
) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        routed = _sql_path(routed_path)
        supply = _sql_path(supply_path)
        selected_sql = (
            f"SELECT * FROM read_parquet('{routed}') WHERE day_type = '{day_type}' AND activity_kind <> 'travel'"
        )
        connection.execute(f"CREATE TEMP VIEW selected AS {selected_sql}")
        unsupported = connection.execute(
            "SELECT string_agg(DISTINCT activity_kind, ',') FROM selected WHERE activity_kind NOT IN ("
            + ",".join(f"'{kind}'" for kind in ACTIVITY_IDS)
            + ")"
        ).fetchone()[0]
        if unsupported:
            raise ValueError(f"CASMSocial export does not support activity kinds: {unsupported}")
        cases = " ".join(f"WHEN '{kind}' THEN {value}" for kind, value in ACTIVITY_IDS.items())
        connection.execute(
            f"COPY (SELECT person_id::BIGINT sp_persons_id, CASE activity_kind {cases} END::INTEGER activity_id, "
            "row_number() OVER (PARTITION BY person_id ORDER BY activity_sequence) - 1 AS activity_sequence, "
            "start_minute::INTEGER starttime_min, end_minute::INTEGER endtime_min, place_id::BIGINT sp_act_id "
            f"FROM selected ORDER BY person_id, activity_sequence) TO '{_sql_path(output_dir / 'activities.parquet')}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        anchor_expressions = []
        for kind, column in ANCHOR_COLUMNS.items():
            anchor_expressions.append(
                f"first(place_id ORDER BY activity_sequence) FILTER (WHERE activity_kind = '{kind}')::BIGINT {column}"
            )
        connection.execute(
            f"COPY (SELECT person_id::BIGINT sp_id, {', '.join(anchor_expressions)} FROM selected GROUP BY person_id "
            f"ORDER BY person_id) TO '{_sql_path(output_dir / 'persons.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        people = _sql_path(output_dir / "persons.parquet")
        connection.execute(
            f"COPY (SELECT sp_hh_id::BIGINT sp_id, sp_hh_id::BIGINT sp_home_id, count(*)::INTEGER hh_size "
            f"FROM read_parquet('{people}') GROUP BY sp_hh_id ORDER BY sp_hh_id) "
            f"TO '{_sql_path(output_dir / 'hh.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY (SELECT source.place_id::BIGINT sp_id, 0::INTEGER rank, "
            "source.activity_kind place_type, 'colorado-' || source.activity_kind || '-' || source.place_id place_name, "
            f"source.latitude::DOUBLE latitude, source.longitude::DOUBLE longitude FROM read_parquet('{supply}') source "
            "JOIN (SELECT DISTINCT sp_act_id place_id FROM read_parquet('"
            f"{_sql_path(output_dir / 'activities.parquet')}')) used USING (place_id) ORDER BY source.place_id) "
            f"TO '{_sql_path(output_dir / 'places.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        source_ties = _sql_path(profile_dir / "social_networks.parquet")
        connection.execute(
            f"COPY (SELECT tie.* FROM read_parquet('{source_ties}') tie "
            f"JOIN read_parquet('{people}') first_person ON tie.person_id_a = first_person.sp_id "
            f"JOIN read_parquet('{people}') second_person ON tie.person_id_b = second_person.sp_id "
            f"ORDER BY person_id_a, person_id_b, network_kind) TO '{_sql_path(output_dir / 'social_networks.parquet')}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
    activity_map = {str(value): key for key, value in ACTIVITY_IDS.items()}
    (output_dir / "activity_id_map.json").write_text(
        json.dumps(activity_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        name: {
            "path": f"casmsocial/{name}.parquet",
            "rows": pl.scan_parquet(output_dir / f"{name}.parquet").select(pl.len()).collect().item(),
            "sha256": sha256_file(output_dir / f"{name}.parquet"),
        }
        for name in RUNTIME_TABLES
    }


def _materialize_ducklake(input_dir: Path, destination: Path, partition_ranks: int) -> dict[str, object]:
    connection = get_ducklake_connection(destination)
    try:
        connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}"')
        counts = {}
        for name in RUNTIME_TABLES:
            connection.execute(
                f'CREATE OR REPLACE TABLE "{SCHEMA_NAME}"."{name}" AS SELECT * FROM read_parquet(?)',
                [str(input_dir / f"{name}.parquet")],
            )
            counts[name] = connection.execute(f'SELECT count(*) FROM "{SCHEMA_NAME}"."{name}"').fetchone()[0]
        connection.execute("CREATE SCHEMA IF NOT EXISTS partitions")
        connection.execute(
            f"CREATE OR REPLACE TABLE {PARTITION_TABLE} AS SELECT 1::INTEGER imputation, "
            f"{partition_ranks}::INTEGER n_ranks, CAST(hash(sp_id) % {partition_ranks} AS INTEGER) rank, "
            f"sp_id::BIGINT place_id FROM {SCHEMA_NAME}.places"
        )
        counts["place_partitions"] = connection.execute(f"SELECT count(*) FROM {PARTITION_TABLE}").fetchone()[0]
    finally:
        connection.close()
    return {
        "schema": SCHEMA_NAME,
        "partition_table": PARTITION_TABLE,
        "partition_ranks": partition_ranks,
        "counts": counts,
    }


def _write_runtime_config(path: Path, profile: ColoradoDatasetProfile) -> None:
    config = {
        "model.name": profile.runtime.model,
        "random.seed": profile.population.seed,
        "start.datetime": "2026-06-01 00:00:00",
        "duration.hours": profile.runtime.default_duration_hours or 24,
        "timezone": "America/Denver",
        "time.step.minutes": 60,
        "places.table": f"{SCHEMA_NAME}.places",
        "households.table": f"{SCHEMA_NAME}.hh",
        "persons.table": f"{SCHEMA_NAME}.persons",
        "activities.table": f"{SCHEMA_NAME}.activities",
        "social_networks.table": f"{SCHEMA_NAME}.social_networks",
        "social_networks.enabled": True,
        "social_networks.remote_messages.enabled": True,
        "social_networks.remote_messages.interval_minutes": 60,
        "communication.enabled": True,
        "behavior.engine": "schedule",
        "roads.enabled": False,
        "partition.table": "",
        "partition.default_rank": 0,
        "partition.require_full_coverage": False,
        "observers.output_dir": "data/output",
        "observers.agent_log.enabled": False,
        "observers.behavior_log.enabled": False,
        "observers.delta_agent_state.enabled": False,
        "observers.social_interaction_log.enabled": profile.runtime.aggregate_diagnostics_enabled,
        "observers.social_interaction_log_file": "social_interactions.parquet",
        "observers.schedule_occupancy_log.enabled": profile.runtime.aggregate_diagnostics_enabled,
        "observers.schedule_occupancy_log_file": "schedule_occupancy.parquet",
        "logging.rank0_only": True,
        "logging.level": "INFO",
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def build_profile_runtime(
    profile_dir: Path,
    destination_dir: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build a resumable routed schedule and runnable local CASMSocial DuckLake."""
    profile_dir, destination_dir, output_dir = (
        path.expanduser().resolve() for path in (profile_dir, destination_dir, output_dir)
    )
    _validate_profile_product(profile_dir, profile)
    _validate_destination_product(destination_dir, profile)
    fingerprint = _fingerprint(profile_dir, destination_dir, profile)
    if resumed := _resumable(output_dir, fingerprint):
        return resumed
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Runtime output exists but is not resumable; use --overwrite: {output_dir}")
    staging = output_dir.with_name(f"{output_dir.name}.building")
    state_path = staging / "build_state.json"
    if staging.exists():
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
        if not isinstance(state, dict) or state.get("inputs") != fingerprint:
            shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _write_json(state_path, {"inputs": fingerprint})

    source = destination_dir / "daily_activities_with_destinations.parquet"
    supply = destination_dir / "destination_supply.parquet"
    source_parts = _source_partitions(source, staging, profile.routing.person_partitions)
    routed_parts, routing = _route_partitions(source_parts, staging, supply, profile)
    routed_path = staging / "daily_activities_routed.parquet"
    acceptance, counts = _combine_and_validate(routed_parts, supply, routed_path, profile)
    runtime_dir = staging / "casmsocial"
    tables = _export_runtime_tables(
        routed_path,
        supply,
        profile_dir,
        runtime_dir,
        profile.runtime.default_day_type,
    )
    partition_ranks = profile.runtime.verification_ranks or profile.runtime.default_ranks or 1
    ducklake = _materialize_ducklake(runtime_dir, staging / "ducklake", partition_ranks)
    config_path = staging / "casmsocial.yaml"
    _write_runtime_config(config_path, profile)
    shutil.rmtree(staging / "stationary_partitions")
    state_path.unlink()
    outputs = {
        "daily_activities_routed.parquet": {
            "sha256": sha256_file(routed_path),
            "rows": counts["rows"],
        },
        **{
            f"casmsocial/{name}.parquet": {"sha256": table["sha256"], "rows": table["rows"]}
            for name, table in tables.items()
        },
        "casmsocial.yaml": {"sha256": sha256_file(config_path)},
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "resumed": False,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "inputs": fingerprint,
        "routing": routing,
        "acceptance": acceptance,
        "counts": counts,
        "runtime_day_type": profile.runtime.default_day_type,
        "destination_resolution": "event_place_id",
        "tables": tables,
        "ducklake": {**ducklake, "path": "ducklake", "tree_sha256": _tree_sha256(staging / "ducklake")},
        "outputs": outputs,
        "governance": "Local identifier-bearing simulation input; do not publish or serve.",
    }
    _write_json(staging / "manifest.json", manifest)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
