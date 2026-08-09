"""Build and assign profile-bounded Colorado destination supply.

Derived from the mydatalakehouse Colorado OSM POI and routing modules. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from importlib.resources import as_file, files
from pathlib import Path

import duckdb
import polars as pl

from casmsocial.datasets.colorado_front_range.profile_population import load_selected_counties
from casmsocial.datasets.colorado_front_range.profile_schedules import _validate_profile_product
from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile, load_osm_attribution
from casmsocial.datasets.colorado_front_range.sources import sha256_file

TAG_PATTERN = re.compile(r'"(?P<key>[^"]+)"=>"(?P<value>[^"]+)"')
MAPPING = {
    "shopping": {
        "shop": {
            "supermarket",
            "convenience",
            "department_store",
            "mall",
            "clothes",
            "bakery",
            "greengrocer",
            "hardware",
            "variety_store",
        }
    },
    "meal": {"amenity": {"restaurant", "cafe", "fast_food", "food_court", "ice_cream", "pub", "bar"}},
    "healthcare": {
        "amenity": {"doctors", "clinic", "hospital", "pharmacy", "dentist", "veterinary"},
        "healthcare": {"doctor", "clinic", "hospital", "pharmacy", "dentist"},
    },
    "recreation": {
        "leisure": {
            "park",
            "playground",
            "sports_centre",
            "fitness_centre",
            "swimming_pool",
            "stadium",
            "pitch",
            "garden",
        },
        "tourism": {"museum", "gallery", "zoo", "attraction"},
    },
    "social": {"amenity": {"community_centre", "library", "place_of_worship", "social_centre", "arts_centre"}},
    "errand": {"amenity": {"bank", "post_office", "townhall", "courthouse", "police", "public_building"}},
}
BASE_CAPACITY = {"shopping": 75, "meal": 50, "healthcare": 40, "recreation": 125, "social": 75, "errand": 30}
DISCRETIONARY_KINDS = (*MAPPING, "other")
OUTPUT_FILES = {
    "osm_pois": "osm_pois.parquet",
    "destination_supply": "destination_supply.parquet",
    "daily_activities": "daily_activities_with_destinations.parquet",
    "osm_attribution": "OPENSTREETMAP_ATTRIBUTION.md",
}


def _sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def _category(tags: dict[str, str]) -> str | None:
    """Map supported OSM tags to one schedule-purpose category."""
    for category, groups in MAPPING.items():
        if any(tags.get(key) in values for key, values in groups.items()):
            return category
    return None


def _place_id(source_id: str, category: str) -> int:
    digest = hashlib.sha256(f"{source_id}\0{category}".encode()).digest()
    return -(int.from_bytes(digest[:8], "big") % 8_000_000_000_000_000_000) - 1


def _load_geospatial_modules():
    try:
        import geopandas as gpd
        import pyogrio
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OSM destination building requires the optional data-builder dependencies; install casmsocial[data-builder]"
        ) from error
    return gpd, pyogrio


def _read_layer(path: Path, layer: str, bbox: tuple[float, float, float, float], config_path: Path):
    gpd, pyogrio = _load_geospatial_modules()
    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(config_path)})
    try:
        try:
            frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox, use_arrow=True)
        except (pyogrio.errors.DataLayerError, pyogrio.errors.DataSourceError):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    finally:
        pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": None})
    if frame.crs is None:
        raise ValueError(f"OSM {layer} layer does not declare a CRS")
    return frame.to_crs("EPSG:4326")


def extract_profile_osm_pois(
    osm_path: Path,
    county_boundaries: Path,
    profile: ColoradoDatasetProfile,
    destination: Path,
    *,
    minimum_places_per_activity_kind: int = 20,
) -> dict[str, object]:
    """Classify OSM points/polygons and spatially assign administrative CBSAs."""
    if minimum_places_per_activity_kind < 1:
        raise ValueError("minimum_places_per_activity_kind must be positive")
    gpd, _ = _load_geospatial_modules()
    counties = load_selected_counties(county_boundaries, profile)
    bbox = tuple(float(value) for value in counties.total_bounds)
    rows: list[dict[str, object]] = []
    examined = unmapped = excluded = 0
    config_resource = files("casmsocial.datasets.colorado_front_range").joinpath("assets", "osmconf.ini")
    with as_file(config_resource) as config_path:
        for layer in ("points", "multipolygons"):
            frame = _read_layer(osm_path, layer, bbox, config_path)
            if frame.empty:
                continue
            if layer == "multipolygons":
                geometry = frame.to_crs("EPSG:26913").geometry.centroid.to_crs("EPSG:4326")
            else:
                geometry = frame.geometry
            for position, (_, feature) in enumerate(frame.iterrows()):
                examined += 1
                point = geometry.iloc[position]
                if point is None or point.is_empty:
                    excluded += 1
                    continue
                other_tags = feature.get("other_tags")
                tags = dict(TAG_PATTERN.findall(other_tags if isinstance(other_tags, str) else ""))
                for key in ("amenity", "shop", "leisure", "tourism", "healthcare"):
                    value = feature.get(key)
                    text = str(value).strip()
                    if text.lower() not in {"", "nan", "none", "<na>"}:
                        tags[key] = text
                if any(tags.get(key) not in (None, "", "no") for key in ("construction", "disused", "abandoned")):
                    excluded += 1
                    continue
                category = _category(tags)
                if category is None:
                    unmapped += 1
                    continue
                osm_id = feature.get("osm_id", feature.get("id", position))
                source_id = f"osm:{layer}:{osm_id}"
                rows.append({
                    "source_id": source_id,
                    "poi_category": category,
                    "longitude": float(point.x),
                    "latitude": float(point.y),
                    "base_capacity": BASE_CAPACITY[category],
                    "geometry": point,
                })
    if not rows:
        raise ValueError("No supported OSM destinations were found inside the profile read extent")
    poi_frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    assigned = gpd.sjoin(poi_frame, counties, how="inner", predicate="within").drop_duplicates([
        "source_id",
        "poi_category",
    ])
    if assigned.empty:
        raise ValueError("No supported OSM destinations fall inside the selected profile counties")
    pois = (
        pl
        .DataFrame({
            "source_id": assigned["source_id"].astype(str).tolist(),
            "poi_category": assigned["poi_category"].astype(str).tolist(),
            "longitude": assigned["longitude"].astype(float).tolist(),
            "latitude": assigned["latitude"].astype(float).tolist(),
            "home_county_geoid": assigned["GEOID"].astype(str).tolist(),
            "cbsa_code": assigned["home_cbsa_code"].astype(str).tolist(),
            "base_capacity": assigned["base_capacity"].astype(int).tolist(),
        })
        .unique(subset=["source_id", "poi_category"])
        .sort("cbsa_code", "poi_category", "source_id")
    )
    required_cbsas = set(profile.geography.home_cbsa_codes)
    coverage = pois.group_by("cbsa_code", "poi_category").len().sort("cbsa_code", "poi_category").to_dicts()
    insufficient = []
    for cbsa in sorted(required_cbsas):
        for category in MAPPING:
            count = pois.filter((pl.col("cbsa_code") == cbsa) & (pl.col("poi_category") == category)).height
            if count < minimum_places_per_activity_kind:
                insufficient.append({"cbsa_code": cbsa, "poi_category": category, "places": count})
    if insufficient:
        raise ValueError(f"OSM destination coverage is below the profile minimum: {insufficient}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pois.write_parquet(destination, compression="zstd")
    return {
        "features_examined": examined,
        "mapped_inside_profile": pois.height,
        "unmapped_features": unmapped,
        "excluded_features": excluded,
        "minimum_places_per_activity_kind": minimum_places_per_activity_kind,
        "coverage": coverage,
    }


def _validate_schedule_product(schedule_dir: Path, profile: ColoradoDatasetProfile) -> dict[str, object]:
    manifest_path = schedule_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("outputs", {}).get("daily_activities", {}) if isinstance(manifest, dict) else {}
    schedule_path = schedule_dir / "daily_activities.parquet"
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "passed"
        or manifest.get("profile_id") != profile.profile_id
        or manifest.get("profile_version") != profile.profile_version
        or not isinstance(output, dict)
        or not schedule_path.is_file()
        or output.get("sha256") != sha256_file(schedule_path)
    ):
        raise ValueError("Profile schedule does not have a matching accepted manifest")
    return manifest


def _build_destination_supply(
    profile_dir: Path,
    poi_path: Path,
    destination: Path,
    profile: ColoradoDatasetProfile,
    capacity_multiplier: float | None,
    full_population_capacity_multiplier: float | None,
) -> dict[str, object]:
    multiplier = (
        capacity_multiplier if capacity_multiplier is not None else profile.routing.destination_capacity_multiplier
    )
    if multiplier is None or multiplier <= 0:
        raise ValueError(
            "Profile must define a positive destination capacity multiplier or receive an explicit override"
        )
    full_multiplier = (
        full_population_capacity_multiplier
        if full_population_capacity_multiplier is not None
        else profile.routing.full_population_capacity_multiplier or 1.0
    )
    if full_multiplier <= 0:
        raise ValueError("Full-population capacity multiplier must be positive")
    connection = duckdb.connect()
    try:
        people = _sql_path(profile_dir / "persons.parquet")
        households = _sql_path(profile_dir / "hh.parquet")
        places = _sql_path(profile_dir / "places.parquet")
        connection.execute(f"CREATE TEMP VIEW people AS SELECT * FROM read_parquet('{people}')")
        connection.execute(f"CREATE TEMP VIEW households AS SELECT * FROM read_parquet('{households}')")
        connection.execute(f"CREATE TEMP VIEW places AS SELECT * FROM read_parquet('{places}')")
        connection.execute(
            "CREATE TEMP VIEW home_supply AS SELECT household.sp_home_id::BIGINT place_id, 'home' activity_kind, "
            "place.latitude::DOUBLE latitude, place.longitude::DOUBLE longitude, min(person.home_cbsa_code) cbsa_code, "
            "count(*)::BIGINT base_capacity, count(*)::BIGINT capacity, 'synthetic_home' supply_source "
            "FROM people person JOIN households household ON person.sp_hh_id = household.sp_id "
            "JOIN places place ON household.sp_home_id = place.sp_id GROUP BY household.sp_home_id, place.latitude, place.longitude"
        )
        connection.execute(
            "CREATE TEMP VIEW anchor_supply AS SELECT place.sp_id::BIGINT place_id, "
            "person.activity_assignment_kind activity_kind, place.latitude::DOUBLE latitude, place.longitude::DOUBLE longitude, "
            "NULL::VARCHAR cbsa_code, greatest(2, count(*))::BIGINT base_capacity, greatest(2, count(*))::BIGINT capacity, "
            "'resolved_synthetic_anchor' supply_source FROM people person JOIN places place ON person.sp_work_id = place.sp_id "
            "WHERE person.activity_assignment_kind IN ('work', 'school', 'daycare') "
            "GROUP BY place.sp_id, person.activity_assignment_kind, place.latitude, place.longitude"
        )
        poi_frame = pl.read_parquet(poi_path)
        scenario_rows = []
        for row in poi_frame.iter_rows(named=True):
            for category in (row["poi_category"], "other"):
                base = int(row["base_capacity"])
                scenario_rows.append({
                    "place_id": _place_id(str(row["source_id"]), category),
                    "activity_kind": category,
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "cbsa_code": row["cbsa_code"],
                    "base_capacity": base,
                    "capacity": max(1, int(base * multiplier * full_multiplier)),
                    "supply_source": "openstreetmap_scenario_capacity",
                })
        scenario_path = destination.with_name("osm_scenario_supply.parquet")
        pl.DataFrame(scenario_rows).write_parquet(scenario_path, compression="zstd")
        connection.execute(
            f"COPY (SELECT * FROM home_supply UNION ALL SELECT * FROM anchor_supply UNION ALL "
            f"SELECT * FROM read_parquet('{_sql_path(scenario_path)}') ORDER BY activity_kind, place_id) "
            f"TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        scenario_path.unlink()
        collisions = connection.execute(
            f"SELECT count(*) FROM (SELECT place_id FROM read_parquet('{_sql_path(destination)}') "
            "GROUP BY place_id HAVING count(*) > 1)"
        ).fetchone()[0]
        if collisions:
            raise ValueError(f"Destination supply generated {collisions} colliding place identifiers")
        invalid_supply = connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(destination)}') WHERE capacity < 1 "
            "OR latitude NOT BETWEEN -90 AND 90 OR longitude NOT BETWEEN -180 AND 180 "
            "OR activity_kind IS NULL"
        ).fetchone()[0]
        if invalid_supply:
            raise ValueError(f"Destination supply contains {invalid_supply} invalid rows")
        counts = [
            {"activity_kind": row[0], "places": row[1]}
            for row in connection.execute(
                f"SELECT activity_kind, count(*) FROM read_parquet('{_sql_path(destination)}') "
                "GROUP BY activity_kind ORDER BY activity_kind"
            ).fetchall()
        ]
    finally:
        connection.close()
    return {
        "capacity_multiplier": multiplier,
        "full_population_capacity_multiplier": full_multiplier,
        "counts": counts,
        "governance": "Capacities are scenario controls, not observed building capacity or occupancy.",
    }


def _assign_destinations(
    profile_dir: Path,
    schedule_dir: Path,
    supply_path: Path,
    destination: Path,
) -> dict[str, object]:
    connection = duckdb.connect()
    try:
        schedules = _sql_path(schedule_dir / "daily_activities.parquet")
        people = _sql_path(profile_dir / "persons.parquet")
        supply = _sql_path(supply_path)
        connection.execute(f"CREATE TEMP VIEW people AS SELECT * FROM read_parquet('{people}')")
        connection.execute(f"CREATE TEMP VIEW supply AS SELECT * FROM read_parquet('{supply}')")
        connection.execute(
            "CREATE TEMP VIEW destination_ranked AS SELECT *, row_number() OVER "
            "(PARTITION BY cbsa_code, activity_kind, grid_cell ORDER BY place_id) destination_rank FROM ("
            "SELECT *, floor(latitude * 20)::BIGINT || ':' || floor(longitude * 20)::BIGINT grid_cell FROM supply "
            "WHERE supply_source = 'openstreetmap_scenario_capacity')"
        )
        connection.execute(
            "CREATE TEMP VIEW destination_counts AS SELECT cbsa_code, activity_kind, grid_cell, count(*) destination_count "
            "FROM destination_ranked GROUP BY ALL"
        )
        connection.execute(
            "CREATE TEMP VIEW fallback_ranked AS SELECT *, row_number() OVER "
            "(PARTITION BY cbsa_code, activity_kind ORDER BY place_id) destination_rank FROM supply "
            "WHERE supply_source = 'openstreetmap_scenario_capacity'"
        )
        connection.execute(
            "CREATE TEMP VIEW fallback_counts AS SELECT cbsa_code, activity_kind, count(*) destination_count "
            "FROM fallback_ranked GROUP BY ALL"
        )
        connection.execute(
            f"CREATE TEMP VIEW stationary_source AS WITH source AS (SELECT *, "
            "lead(CASE WHEN activity_kind <> 'travel' THEN start_minute END IGNORE NULLS) OVER "
            f"(PARTITION BY person_id, day_type ORDER BY activity_sequence) next_stationary_start FROM read_parquet('{schedules}')) "
            "SELECT person_id, day_type, activity_sequence source_sequence, activity_kind, start_minute, "
            "coalesce(next_stationary_start, end_minute)::BIGINT end_minute, place_id, location_source, schedule_source "
            "FROM source WHERE activity_kind <> 'travel'"
        )
        discretionary = ", ".join(f"'{kind}'" for kind in DISCRETIONARY_KINDS)
        connection.execute(
            "CREATE TEMP VIEW selected AS WITH base AS (SELECT activity.*, person.home_cbsa_code, "
            "floor(person.home_latitude * 20)::BIGINT || ':' || floor(person.home_longitude * 20)::BIGINT home_grid_cell, "
            f"CASE WHEN activity.activity_kind IN ({discretionary}, 'home', 'work', 'school', 'daycare') "
            "THEN activity.activity_kind ELSE 'home' END routed_kind "
            "FROM stationary_source activity JOIN people person ON activity.person_id = person.sp_id), choices AS ("
            "SELECT base.*, coalesce(local.place_id, fallback.place_id) destination_place_id FROM base "
            "LEFT JOIN destination_counts counts ON base.routed_kind = counts.activity_kind "
            "AND base.home_cbsa_code = counts.cbsa_code AND base.home_grid_cell = counts.grid_cell "
            "LEFT JOIN destination_ranked local ON base.routed_kind = local.activity_kind "
            "AND base.home_cbsa_code = local.cbsa_code AND base.home_grid_cell = local.grid_cell "
            "AND local.destination_rank = 1 + (hash(base.person_id, base.day_type, base.source_sequence) % counts.destination_count) "
            "LEFT JOIN fallback_counts fallback_count ON base.routed_kind = fallback_count.activity_kind "
            "AND base.home_cbsa_code = fallback_count.cbsa_code LEFT JOIN fallback_ranked fallback "
            "ON base.routed_kind = fallback.activity_kind AND base.home_cbsa_code = fallback.cbsa_code "
            "AND fallback.destination_rank = 1 + (hash(base.person_id, base.day_type, base.source_sequence, 'fallback') "
            " % fallback_count.destination_count)) SELECT person_id, day_type, row_number() OVER "
            "(PARTITION BY person_id, day_type ORDER BY source_sequence) - 1 AS activity_sequence, routed_kind activity_kind, "
            "activity_kind activity_purpose, start_minute, end_minute, CASE WHEN routed_kind IN ("
            f"{discretionary}) THEN destination_place_id ELSE place_id END::BIGINT place_id, CASE WHEN activity_kind IN ("
            f"{discretionary}) THEN 'sampled_destination' WHEN routed_kind = 'home' THEN 'home_anchor' "
            "ELSE location_source END location_source, schedule_source FROM choices"
        )
        missing = connection.execute("SELECT count(*) FROM selected WHERE place_id IS NULL").fetchone()[0]
        if missing:
            raise ValueError(f"No profile destination was available for {missing} stationary activities")
        connection.execute(
            f"COPY (SELECT * FROM selected ORDER BY person_id, day_type, activity_sequence) "
            f"TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        output = f"read_parquet('{_sql_path(destination)}')"
        checks = {
            "unknown_stationary_places": connection.execute(
                f"SELECT count(*) FROM {output} activity ANTI JOIN supply USING (place_id)"
            ).fetchone()[0],
            "unassigned_discretionary_events": connection.execute(
                f"SELECT count(*) FROM {output} WHERE activity_kind IN ({discretionary}) "
                "AND location_source <> 'sampled_destination'"
            ).fetchone()[0],
            "invalid_intervals": connection.execute(
                f"SELECT count(*) FROM {output} WHERE start_minute < 0 OR end_minute > 1440 OR start_minute >= end_minute"
            ).fetchone()[0],
            "noncontiguous_transitions": connection.execute(
                f"WITH ordered AS (SELECT *, lead(start_minute) OVER (PARTITION BY person_id, day_type "
                f"ORDER BY activity_sequence) next_start FROM {output}) SELECT count(*) FROM ordered "
                "WHERE next_start IS NOT NULL AND end_minute <> next_start"
            ).fetchone()[0],
            "incomplete_plan_boundaries": connection.execute(
                f"WITH bounds AS (SELECT person_id, day_type, min(start_minute) first_start, max(end_minute) last_end "
                f"FROM {output} GROUP BY person_id, day_type) SELECT count(*) FROM bounds "
                "WHERE first_start <> 0 OR last_end <> 1440"
            ).fetchone()[0],
        }
        counts = {
            "rows": connection.execute(f"SELECT count(*) FROM {output}").fetchone()[0],
            "persons": connection.execute(f"SELECT count(DISTINCT person_id) FROM {output}").fetchone()[0],
            "assigned_discretionary_events": connection.execute(
                f"SELECT count(*) FROM {output} WHERE activity_kind IN ({discretionary})"
            ).fetchone()[0],
        }
    finally:
        connection.close()
    if any(checks.values()):
        raise ValueError(f"Destination assignment acceptance failed: {checks}")
    return {
        "policy": "event_level_deterministic_home_grid_then_cbsa_fallback",
        "counts": counts,
        "acceptance": {"status": "passed", "required_zero": list(checks), "checks": checks},
    }


def _fingerprint(
    profile_dir: Path,
    schedule_dir: Path,
    osm_path: Path,
    county_boundaries: Path,
    profile: ColoradoDatasetProfile,
    minimum_places_per_activity_kind: int,
    capacity_multiplier: float | None,
    full_population_capacity_multiplier: float | None,
) -> dict[str, object]:
    return {
        "destination_contract_version": 2,
        "profile_manifest_sha256": sha256_file(profile_dir / "manifest.json"),
        "schedule_manifest_sha256": sha256_file(schedule_dir / "manifest.json"),
        "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
        "county_boundaries": {"path": str(county_boundaries), "sha256": sha256_file(county_boundaries)},
        "profile": profile.model_dump(mode="json"),
        "minimum_places_per_activity_kind": minimum_places_per_activity_kind,
        "capacity_multiplier_override": capacity_multiplier,
        "full_population_capacity_multiplier_override": full_population_capacity_multiplier,
    }


def _resumable(output_dir: Path, fingerprint: dict[str, object]) -> dict[str, object] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed" or manifest.get("inputs") != fingerprint:
        return None
    outputs = manifest.get("outputs", {})
    for name, relative_path in OUTPUT_FILES.items():
        artifact = outputs.get(name, {}) if isinstance(outputs, dict) else {}
        path = output_dir / relative_path
        if not isinstance(artifact, dict) or not path.is_file() or artifact.get("sha256") != sha256_file(path):
            return None
    return {**manifest, "resumed": True}


def build_profile_destinations(
    profile_dir: Path,
    schedule_dir: Path,
    osm_path: Path,
    county_boundaries: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
    *,
    minimum_places_per_activity_kind: int = 20,
    capacity_multiplier: float | None = None,
    full_population_capacity_multiplier: float | None = None,
    allow_planned: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build accepted supply and event-level pre-routing destination assignments."""
    profile_dir, schedule_dir, osm_path, county_boundaries, output_dir = (
        path.expanduser().resolve() for path in (profile_dir, schedule_dir, osm_path, county_boundaries, output_dir)
    )
    _validate_profile_product(profile_dir, profile)
    _validate_schedule_product(schedule_dir, profile)
    for path in (osm_path, county_boundaries):
        if not path.is_file():
            raise FileNotFoundError(path)
    if profile.release_status == "planned" and not allow_planned:
        raise ValueError(
            f"Profile {profile.profile_id} is planned; pass --allow-planned to build exploratory destinations"
        )
    fingerprint = _fingerprint(
        profile_dir,
        schedule_dir,
        osm_path,
        county_boundaries,
        profile,
        minimum_places_per_activity_kind,
        capacity_multiplier,
        full_population_capacity_multiplier,
    )
    if resumed := _resumable(output_dir, fingerprint):
        return resumed
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Destination output exists but is not resumable; use --overwrite: {output_dir}")
    staging = output_dir.with_name(f"{output_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        extraction = extract_profile_osm_pois(
            osm_path,
            county_boundaries,
            profile,
            staging / OUTPUT_FILES["osm_pois"],
            minimum_places_per_activity_kind=minimum_places_per_activity_kind,
        )
        supply = _build_destination_supply(
            profile_dir,
            staging / OUTPUT_FILES["osm_pois"],
            staging / OUTPUT_FILES["destination_supply"],
            profile,
            capacity_multiplier,
            full_population_capacity_multiplier,
        )
        assignment = _assign_destinations(
            profile_dir,
            schedule_dir,
            staging / OUTPUT_FILES["destination_supply"],
            staging / OUTPUT_FILES["daily_activities"],
        )
        (staging / OUTPUT_FILES["osm_attribution"]).write_text(
            load_osm_attribution() + "\n",
            encoding="utf-8",
        )
        outputs = {
            name: {"path": relative_path, "sha256": sha256_file(staging / relative_path)}
            for name, relative_path in OUTPUT_FILES.items()
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "resumed": False,
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "release_status": profile.release_status,
            "inputs": fingerprint,
            "extraction": extraction,
            "supply": supply,
            "assignment": assignment,
            "outputs": outputs,
            "routing_status": "ready_for_step_10_routing",
            "governance": {
                "classification": "local_identifier_bearing_osm_derived_simulation_input",
                "distribution_policy": "local_build_only",
                "redistribution_authorized": False,
                "attribution": "© OpenStreetMap contributors",
                "license": "ODbL-1.0",
                "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "attribution_url": "https://www.openstreetmap.org/copyright",
                "notice": OUTPUT_FILES["osm_attribution"],
            },
        }
        temporary = staging / "manifest.json.part"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(staging / "manifest.json")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
