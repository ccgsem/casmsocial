"""Build profile-scoped Colorado populations from the statewide OSF DuckLake.

This module is derived from the Colorado quality, pilot, and fixture builders
in mydatalakehouse at commit ec2336af55fb7221922ad972ec1b5f750746f114.
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

from casmsocial.datasets.colorado_front_range.boundary import cbsa_by_county_geoid
from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile
from casmsocial.datasets.colorado_front_range.sources import sha256_file

TABLE_NAMES = ("places", "hh", "persons", "social_networks")
POPULATION_CONTRACT_VERSION = 2


def _sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def _write_json_atomic(path: Path, content: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_geopandas():
    try:
        import geopandas as gpd
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Profile geography assignment requires the optional data-builder dependencies; "
            "install casmsocial[data-builder]"
        ) from error
    return gpd


def load_selected_counties(county_boundaries: Path, profile: ColoradoDatasetProfile):
    """Load selected Colorado county polygons and attach bundled CBSA codes."""
    gpd = _load_geopandas()
    counties = gpd.read_file(county_boundaries)
    required = {"GEOID", "STATEFP", "geometry"}
    if missing := required - set(counties.columns):
        raise ValueError(f"County boundaries are missing columns: {', '.join(sorted(missing))}")
    if counties.crs is None:
        raise ValueError("County boundaries must declare a CRS")
    counties = counties.copy()
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    mapping = cbsa_by_county_geoid()
    selected_cbsas = set(profile.geography.home_cbsa_codes)
    selected_geoids = {geoid for geoid, cbsa in mapping.items() if cbsa in selected_cbsas}
    selected = counties[(counties["STATEFP"] == "08") & counties["GEOID"].isin(selected_geoids)][
        ["GEOID", "geometry"]
    ].copy()
    if set(selected["GEOID"]) != selected_geoids:
        missing = sorted(selected_geoids - set(selected["GEOID"]))
        raise ValueError(f"County boundary file is missing selected Colorado GEOIDs: {', '.join(missing)}")
    selected["home_cbsa_code"] = selected["GEOID"].map(mapping)
    return selected.to_crs("EPSG:4326")


def write_home_assignments(
    connection: duckdb.DuckDBPyConnection,
    county_boundaries: Path,
    profile: ColoradoDatasetProfile,
    destination: Path,
    *,
    batch_size: int = 250_000,
) -> dict[str, object]:
    """Spatially assign statewide home points to selected profile CBSAs in batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    gpd = _load_geopandas()
    counties = load_selected_counties(county_boundaries, profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    writer = None
    scanned = 0
    selected_rows = 0
    try:
        reader = connection.execute(
            "SELECT sp_id, home_longitude, home_latitude FROM persons WHERE source_state = 'CO' ORDER BY sp_id"
        ).to_arrow_reader(batch_size)
        for batch in reader:
            frame = batch.to_pandas()
            scanned += len(frame)
            valid = frame[
                frame["home_longitude"].notna()
                & frame["home_latitude"].notna()
                & frame["home_longitude"].between(-180, 180)
                & frame["home_latitude"].between(-90, 90)
            ].copy()
            points = gpd.GeoDataFrame(
                valid,
                geometry=gpd.points_from_xy(valid["home_longitude"], valid["home_latitude"]),
                crs="EPSG:4326",
            )
            assigned = gpd.sjoin(points, counties, how="inner", predicate="within")
            assigned = assigned.drop_duplicates("sp_id")
            output = pl.DataFrame(
                {
                    "person_id": assigned["sp_id"].tolist(),
                    "home_county_geoid": assigned["GEOID"].astype(str).tolist(),
                    "home_cbsa_code": assigned["home_cbsa_code"].astype(str).tolist(),
                },
                schema={
                    "person_id": pl.Int64,
                    "home_county_geoid": pl.String,
                    "home_cbsa_code": pl.String,
                },
            )
            table = output.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            selected_rows += output.height
    finally:
        if writer is not None:
            writer.close()
    if writer is None or selected_rows == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"No Colorado residents matched profile boundary {profile.geography.boundary_id}")
    temporary.replace(destination)
    counts = {
        row[0]: row[1]
        for row in pl.read_parquet(destination).group_by("home_cbsa_code").len().sort("home_cbsa_code").iter_rows()
    }
    return {
        "statewide_persons_scanned": scanned,
        "selected_boundary_persons": selected_rows,
        "persons_by_cbsa": counts,
    }


def _create_selected_people(
    connection: duckdb.DuckDBPyConnection, profile: ColoradoDatasetProfile
) -> dict[str, object]:
    population = profile.population
    if population.mode == "full_boundary_population":
        connection.execute("CREATE TEMP TABLE selected_people AS SELECT sp_id FROM candidate_people")
        return {"method": "full_boundary_population", "reserved_per_cbsa": None}

    person_limit = population.person_limit
    minimum = population.minimum_persons_per_cbsa
    if person_limit is None or minimum is None:
        raise ValueError("Sample profile must define person_limit and minimum_persons_per_cbsa")
    candidate_count = connection.execute("SELECT count(*) FROM candidate_people").fetchone()[0]
    if candidate_count < person_limit:
        raise ValueError(f"Profile requests {person_limit} people but boundary contains {candidate_count}")
    small_cbsas = connection.execute(
        "SELECT home_cbsa_code, count(*) records FROM candidate_people GROUP BY home_cbsa_code "
        f"HAVING records < {minimum} ORDER BY home_cbsa_code"
    ).fetchall()
    if small_cbsas:
        raise ValueError(f"CBSAs do not meet minimum sample coverage: {small_cbsas}")
    seed = population.seed
    connection.execute(
        "CREATE TEMP TABLE reserved_people AS SELECT sp_id FROM ("
        "SELECT sp_id, row_number() OVER (PARTITION BY home_cbsa_code ORDER BY hash(sp_id, "
        f"{seed})) cbsa_rank FROM candidate_people) WHERE cbsa_rank <= {minimum}"
    )
    reserved = connection.execute("SELECT count(*) FROM reserved_people").fetchone()[0]
    remaining = person_limit - reserved
    if remaining < 0:
        raise ValueError("minimum_persons_per_cbsa exceeds the requested person limit")
    connection.execute(
        "CREATE TEMP TABLE selected_people AS SELECT sp_id FROM reserved_people UNION ALL SELECT sp_id FROM ("
        "SELECT candidate.sp_id, row_number() OVER (PARTITION BY home_cbsa_code, age_group, "
        "coalesce(activity_assignment_kind, 'none') ORDER BY hash(candidate.sp_id, "
        f"{seed})) stratum_rank FROM candidate_people candidate ANTI JOIN reserved_people USING (sp_id) "
        f"ORDER BY stratum_rank, hash(home_cbsa_code, age_group, coalesce(activity_assignment_kind, 'none'), {seed}), "
        f"hash(candidate.sp_id, {seed}) LIMIT {remaining})"
    )
    connection.execute(
        "CREATE TEMP TABLE network_seed_ties AS WITH eligible AS (SELECT tie.person_id_a, tie.person_id_b, "
        "missing.home_cbsa_code missing_cbsa_code, row_number() OVER (ORDER BY "
        f"CASE WHEN tie.network_kind = 'household' THEN 0 ELSE 1 END, hash(tie.person_id_a, tie.person_id_b, {seed})) "
        "tie_rank FROM social_networks tie JOIN candidate_people first_person ON tie.person_id_a = first_person.sp_id "
        "JOIN candidate_people second_person ON tie.person_id_b = second_person.sp_id "
        "LEFT JOIN selected_people selected_a ON tie.person_id_a = selected_a.sp_id "
        "LEFT JOIN selected_people selected_b ON tie.person_id_b = selected_b.sp_id "
        "JOIN candidate_people missing ON missing.sp_id = CASE WHEN selected_a.sp_id IS NULL "
        "THEN tie.person_id_a ELSE tie.person_id_b END WHERE (selected_a.sp_id IS NULL) <> (selected_b.sp_id IS NULL)) "
        "SELECT person_id_a, person_id_b, missing_cbsa_code FROM eligible WHERE tie_rank = 1"
    )
    connection.execute(
        "CREATE TEMP TABLE network_seed_people AS SELECT person_id_a sp_id FROM network_seed_ties "
        "UNION SELECT person_id_b FROM network_seed_ties"
    )
    connection.execute(
        "CREATE TEMP TABLE missing_network_seed_people AS SELECT seed.sp_id, candidate.home_cbsa_code "
        "FROM network_seed_people seed JOIN candidate_people candidate USING (sp_id) "
        "ANTI JOIN selected_people USING (sp_id)"
    )
    connection.execute(
        "CREATE TEMP TABLE network_seed_removals AS WITH ranked AS (SELECT selected.sp_id, "
        "candidate.home_cbsa_code, row_number() OVER (PARTITION BY candidate.home_cbsa_code ORDER BY "
        f"hash(selected.sp_id, 'network-seed-replacement', {seed})) removal_rank FROM selected_people selected "
        "JOIN candidate_people candidate USING (sp_id) ANTI JOIN network_seed_people USING (sp_id)) "
        "SELECT ranked.sp_id FROM ranked JOIN (SELECT home_cbsa_code, count(*) replacements "
        "FROM missing_network_seed_people GROUP BY home_cbsa_code) needed USING (home_cbsa_code) "
        "WHERE removal_rank <= replacements"
    )
    removed = connection.execute("SELECT count(*) FROM network_seed_removals").fetchone()[0]
    added = connection.execute("SELECT count(*) FROM missing_network_seed_people").fetchone()[0]
    if removed != added:
        raise ValueError(f"Network seed replacement is unbalanced: added={added}, removed={removed}")
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE selected_people AS SELECT selected.sp_id FROM selected_people selected "
        "ANTI JOIN network_seed_removals removal USING (sp_id) UNION ALL SELECT sp_id FROM missing_network_seed_people"
    )
    seeded_ties = connection.execute("SELECT count(*) FROM network_seed_ties").fetchone()[0]
    return {
        "method": "minimum_cbsa_then_round_robin_strata_with_network_seed",
        "reserved_per_cbsa": minimum,
        "strata": population.strata,
        "seed": seed,
        "network_seed_ties": seeded_ties,
        "network_seed_people_added": added,
        "network_seed_people_removed": removed,
    }


def build_profile_tables(
    connection: duckdb.DuckDBPyConnection,
    assignments: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
) -> dict[str, object]:
    """Materialize four endpoint-complete tables for a selected profile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = ", ".join(f"'{code}'" for code in profile.geography.home_cbsa_codes)
    connection.execute(
        "CREATE TEMP TABLE candidate_people AS SELECT person.*, assignment.home_county_geoid, "
        "assignment.home_cbsa_code, CASE WHEN age < 5 THEN '0-4' WHEN age < 15 THEN '5-14' "
        "WHEN age < 25 THEN '15-24' WHEN age < 65 THEN '25-64' ELSE '65+' END age_group "
        f"FROM persons person JOIN read_parquet('{_sql_path(assignments)}') assignment "
        "ON person.sp_id = assignment.person_id "
        f"WHERE assignment.home_cbsa_code IN ({selected})"
    )
    sampling = _create_selected_people(connection, profile)
    paths = {name: output_dir / f"{name}.parquet" for name in TABLE_NAMES}
    connection.execute(
        f"COPY (SELECT candidate.* FROM candidate_people candidate JOIN selected_people USING (sp_id) "
        f"ORDER BY sp_id) TO '{_sql_path(paths['persons'])}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT household.* FROM hh household JOIN (SELECT DISTINCT sp_hh_id sp_id FROM "
        f"read_parquet('{_sql_path(paths['persons'])}')) selected USING (sp_id) ORDER BY household.sp_id) "
        f"TO '{_sql_path(paths['hh'])}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT place.* FROM places place WHERE place.sp_id IN ("
        f"SELECT sp_hh_id FROM read_parquet('{_sql_path(paths['persons'])}') UNION SELECT sp_work_id FROM "
        f"read_parquet('{_sql_path(paths['persons'])}') WHERE sp_work_id IS NOT NULL) ORDER BY place.sp_id) "
        f"TO '{_sql_path(paths['places'])}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT tie.* FROM social_networks tie JOIN read_parquet('{_sql_path(paths['persons'])}') first_person "
        f"ON tie.person_id_a = first_person.sp_id JOIN read_parquet('{_sql_path(paths['persons'])}') second_person "
        f"ON tie.person_id_b = second_person.sp_id ORDER BY person_id_a, person_id_b, network_kind) "
        f"TO '{_sql_path(paths['social_networks'])}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    counts = {
        name: connection.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(path)}')").fetchone()[0]
        for name, path in paths.items()
    }
    persons_by_cbsa = {
        row[0]: row[1]
        for row in connection.execute(
            f"SELECT home_cbsa_code, count(*) FROM read_parquet('{_sql_path(paths['persons'])}') "
            "GROUP BY home_cbsa_code ORDER BY home_cbsa_code"
        ).fetchall()
    }
    checks = {
        "persons_without_household": connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(paths['persons'])}') person ANTI JOIN "
            f"read_parquet('{_sql_path(paths['hh'])}') household ON person.sp_hh_id = household.sp_id"
        ).fetchone()[0],
        "households_without_home_place": connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(paths['hh'])}') household ANTI JOIN "
            f"read_parquet('{_sql_path(paths['places'])}') place ON household.sp_home_id = place.sp_id"
        ).fetchone()[0],
        "activity_assignments_without_place": connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(paths['persons'])}') person ANTI JOIN "
            f"read_parquet('{_sql_path(paths['places'])}') place ON person.sp_work_id = place.sp_id "
            "WHERE person.sp_work_id IS NOT NULL"
        ).fetchone()[0],
        "social_ties_without_endpoints": connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(paths['social_networks'])}') tie "
            f"LEFT JOIN read_parquet('{_sql_path(paths['persons'])}') a ON tie.person_id_a = a.sp_id "
            f"LEFT JOIN read_parquet('{_sql_path(paths['persons'])}') b ON tie.person_id_b = b.sp_id "
            "WHERE a.sp_id IS NULL OR b.sp_id IS NULL"
        ).fetchone()[0],
    }
    required_cbsas = set(profile.geography.home_cbsa_codes)
    if profile.validation.require_all_cbsas_represented and set(persons_by_cbsa) != required_cbsas:
        raise ValueError(f"Profile output does not represent every required CBSA: {persons_by_cbsa}")
    if profile.population.person_limit is not None and counts["persons"] != profile.population.person_limit:
        raise ValueError(f"Profile output has {counts['persons']} persons, expected {profile.population.person_limit}")
    if profile.validation.require_single_rank_smoke and counts["social_networks"] == 0:
        raise ValueError("Profile runtime smoke requires at least one endpoint-complete sampled social tie")
    if any(
        checks[name]
        for name in ("persons_without_household", "households_without_home_place", "social_ties_without_endpoints")
    ):
        raise ValueError(f"Profile population integrity failed: {checks}")
    return {
        "sampling": sampling,
        "counts": counts,
        "persons_by_cbsa": persons_by_cbsa,
        "integrity": {
            "status": "passed",
            "required_zero": [
                "persons_without_household",
                "households_without_home_place",
                "social_ties_without_endpoints",
            ],
            "checks": checks,
        },
        "tables": {
            name: {
                "path": f"{name}.parquet",
                "rows": counts[name],
                "sha256": sha256_file(path),
                "schema": {key: str(value) for key, value in pl.scan_parquet(path).collect_schema().items()},
            }
            for name, path in paths.items()
        },
    }


def _input_fingerprint(
    catalog: Path, catalog_manifest: Path, county_boundaries: Path, profile: ColoradoDatasetProfile
) -> dict[str, object]:
    return {
        "population_contract_version": POPULATION_CONTRACT_VERSION,
        "catalog": str(catalog),
        "catalog_sha256": sha256_file(catalog),
        "catalog_manifest_sha256": sha256_file(catalog_manifest),
        "county_boundaries": str(county_boundaries),
        "county_boundaries_sha256": sha256_file(county_boundaries),
        "profile": profile.model_dump(mode="json"),
    }


def _resumable_output(output_dir: Path, fingerprint: dict[str, object]) -> dict[str, object] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed" or manifest.get("inputs") != fingerprint:
        return None
    tables = manifest.get("tables", {})
    if not isinstance(tables, dict):
        return None
    for name in TABLE_NAMES:
        table = tables.get(name, {})
        path = output_dir / f"{name}.parquet"
        if not path.is_file() or not isinstance(table, dict) or table.get("sha256") != sha256_file(path):
            return None
    return {**manifest, "resumed": True}


def build_profile_population(
    catalog: Path,
    data_path: Path,
    county_boundaries: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
    *,
    batch_size: int = 250_000,
    allow_planned: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build one accepted, resumable profile-scoped population product."""
    catalog = catalog.expanduser().resolve()
    data_path = data_path.expanduser().resolve()
    county_boundaries = county_boundaries.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    catalog_manifest = catalog.with_suffix(f"{catalog.suffix}.manifest.json")
    for path in (catalog, catalog_manifest, county_boundaries):
        if not path.exists():
            raise FileNotFoundError(path)
    lake_manifest = json.loads(catalog_manifest.read_text(encoding="utf-8"))
    if lake_manifest.get("status") != "passed" or lake_manifest.get("catalog_sha256") != sha256_file(catalog):
        raise ValueError("Input OSF DuckLake does not have a matching accepted manifest")
    if Path(str(lake_manifest.get("data_path"))).resolve() != data_path:
        raise ValueError("Input OSF DuckLake manifest does not match --data-path")
    if profile.release_status == "planned" and not allow_planned:
        raise ValueError(
            f"Profile {profile.profile_id} is planned; pass --allow-planned to build exploratory geography"
        )

    fingerprint = _input_fingerprint(catalog, catalog_manifest, county_boundaries, profile)
    if resumed := _resumable_output(output_dir, fingerprint):
        return resumed
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Profile output exists but is not resumable; use --overwrite: {output_dir}")
    staging = output_dir.with_name(f"{output_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    connection = duckdb.connect()
    try:
        connection.execute("LOAD ducklake")
        connection.execute(
            f"ATTACH 'ducklake:{_sql_path(catalog)}' AS source_lake (DATA_PATH '{_sql_path(data_path)}')"
        )
        connection.execute("USE source_lake")
        assignment_path = staging / "home_assignments.parquet"
        geography = write_home_assignments(
            connection,
            county_boundaries,
            profile,
            assignment_path,
            batch_size=batch_size,
        )
        product = build_profile_tables(connection, assignment_path, profile, staging)
        assignment_path.unlink()
    except Exception:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        connection.close()

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "resumed": False,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "release_status": profile.release_status,
        "inputs": fingerprint,
        "geography": geography,
        "schedule_eligibility": {
            "require_exactly_one_weekday_home": profile.population.require_exactly_one_weekday_home,
            "status": "deferred_until_schedule_generation",
        },
        **product,
    }
    _write_json_atomic(staging / "manifest.json", manifest)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
