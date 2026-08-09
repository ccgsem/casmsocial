"""Normalize OSF state archives into the four canonical CASMSocial tables.

This module is derived from ``mydatalakehouse.osf_synthetic_ducklake`` at
commit 4a9687de19ad192b97139f085d3e348dfe187cbd. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import duckdb
import polars as pl
import pyarrow.parquet as pq

from casmsocial.datasets.colorado_front_range.sources import sha256_file

TABLE_NAMES = ("places", "hh", "persons", "social_networks")
PERSON_COLUMNS = ["id", "age", "gender", "assigned", "hhold", "htype", "wp", "urban", "long", "lat"]
WORKPLACE_COLUMNS = ["wp", "long", "lat"]
EDUCATION_ID_COLUMN = "eduID"
NETWORK_KINDS = ("household", "daycare", "school", "work")
PERSON_SOCIAL_NETWORK_KINDS = ("household", "daycare", "school")


def scoped_id(source_state: str, entity_kind: str, source_id: object) -> int:
    """Return a deterministic positive 63-bit identifier scoped by state and kind."""
    if source_id is None or not str(source_id).strip():
        raise ValueError(f"Missing source identifier for {entity_kind}")
    payload = f"{source_state.upper()}:{entity_kind}:{str(source_id).strip()}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)
    return max(1, value)


def assignment_kind(value: object) -> str | None:
    """Classify an OSF person ``wp`` identifier as work, school, or daycare."""
    text = str(value).strip().lower() if value is not None else ""
    for token, kind in (("w", "work"), ("s", "school"), ("d", "daycare")):
        prefix, separator, suffix = text.rpartition(token)
        if separator and prefix and suffix.isdigit():
            return kind
    return None


def find_member(members: list[str], kind: str, state: str) -> str:
    """Find a required state archive member without relying on ZIP ordering."""
    state = state.lower()
    if kind in {"population", "workplace"}:
        candidates = [
            member for member in members if member.lower().endswith(".gpkg") and kind in Path(member).stem.lower()
        ]
    elif kind in NETWORK_KINDS:
        candidates = [
            member
            for member in members
            if member.lower().endswith(".csv")
            and "social_networks" in member.lower()
            and f"_{kind}_network" in Path(member).stem.lower()
        ]
    else:
        raise ValueError(f"Unsupported archive member kind: {kind}")
    if len(candidates) != 1:
        raise ValueError(f"Expected one {kind} member for {state.upper()}, found {len(candidates)}")
    return candidates[0]


def _extract_member(source_zip: ZipFile, member: str, destination: Path) -> Path:
    """Copy a selected ZIP member to a controlled path without trusting its name."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_zip.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    return destination


def _load_pyogrio():
    try:
        import pyogrio
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OSF GeoPackage normalization requires the optional data-builder dependencies; "
            "install casmsocial[data-builder]"
        ) from error
    return pyogrio


def _feature_count(path: Path) -> int:
    return int(_load_pyogrio().read_info(path)["features"])


def _iter_geopackage(path: Path, columns: list[str], batch_size: int, *, read_geometry: bool = False):
    backend = _load_pyogrio()
    for offset in range(0, int(backend.read_info(path)["features"]), batch_size):
        yield backend.read_dataframe(
            path,
            columns=columns,
            read_geometry=read_geometry,
            skip_features=offset,
            max_features=batch_size,
        )


def _to_polars(frame, columns: list[str]) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.select(columns)
    return pl.from_pandas(frame[columns], include_index=False)


def person_rows(source_state: str, frame) -> pl.DataFrame:
    """Map one population batch to state-scoped person records."""
    missing = set(PERSON_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Population source is missing columns: {', '.join(sorted(missing))}")
    rows = _to_polars(frame, PERSON_COLUMNS).with_columns(
        pl.col("id").cast(pl.String),
        pl.col("hhold").cast(pl.String),
        pl.col("wp").cast(pl.String),
        pl.col("wp").map_elements(assignment_kind, return_dtype=pl.String).alias("activity_assignment_kind"),
    )
    rows = rows.filter(pl.col("id").is_not_null() & pl.col("hhold").is_not_null())
    return rows.select(
        pl
        .col("id")
        .map_elements(lambda value: scoped_id(source_state, "person", value), return_dtype=pl.Int64)
        .alias("sp_id"),
        pl
        .col("hhold")
        .map_elements(lambda value: scoped_id(source_state, "household", value), return_dtype=pl.Int64)
        .alias("sp_hh_id"),
        pl
        .when(pl.col("activity_assignment_kind").is_null())
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(
            pl.col("wp").map_elements(lambda value: scoped_id(source_state, "workplace", value), return_dtype=pl.Int64)
        )
        .alias("sp_work_id"),
        pl.col("activity_assignment_kind"),
        pl.col("age").cast(pl.Float64),
        pl.col("gender").cast(pl.String),
        pl.col("assigned").cast(pl.Int64),
        pl.col("urban").cast(pl.Int64),
        pl.col("htype").cast(pl.String).alias("household_type"),
        pl.col("long").cast(pl.Float64).alias("home_longitude"),
        pl.col("lat").cast(pl.Float64).alias("home_latitude"),
        pl.lit(source_state.upper()).alias("source_state"),
    )


def workplace_rows(source_state: str, frame) -> pl.DataFrame:
    """Map one workplace batch to activity-place rows."""
    missing = set(WORKPLACE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Workplace source is missing columns: {', '.join(sorted(missing))}")
    rows = _to_polars(frame, WORKPLACE_COLUMNS).with_columns(pl.col("wp").cast(pl.String))
    return (
        rows
        .filter(pl.col("wp").is_not_null() & (pl.col("wp").str.strip_chars() != ""))
        .select(
            pl
            .col("wp")
            .map_elements(lambda value: scoped_id(source_state, "workplace", value), return_dtype=pl.Int64)
            .alias("sp_id"),
            pl.lit("Workplace").alias("place_type"),
            pl.col("long").cast(pl.Float64).alias("longitude"),
            pl.col("lat").cast(pl.Float64).alias("latitude"),
            pl.lit(source_state.upper()).alias("source_state"),
        )
        .unique("sp_id")
    )


def education_rows(source_state: str, frame, place_type: str) -> pl.DataFrame:
    """Map a school or daycare GeoPackage batch to activity-place rows."""
    if EDUCATION_ID_COLUMN not in frame.columns:
        raise ValueError(f"Education source is missing {EDUCATION_ID_COLUMN}")
    if isinstance(frame, pl.DataFrame):
        required = {"longitude", "latitude"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Education source is missing columns: {', '.join(sorted(missing))}")
        rows = frame.select(EDUCATION_ID_COLUMN, "longitude", "latitude")
    else:
        if frame.geometry.isna().any():
            raise ValueError("Education source contains a site without geometry")
        rows = pl.DataFrame({
            EDUCATION_ID_COLUMN: frame[EDUCATION_ID_COLUMN].astype(str).tolist(),
            "longitude": frame.geometry.x.tolist(),
            "latitude": frame.geometry.y.tolist(),
        })
    return (
        rows
        .with_columns(pl.col(EDUCATION_ID_COLUMN).cast(pl.String))
        .filter(pl.col(EDUCATION_ID_COLUMN).str.strip_chars() != "")
        .select(
            pl
            .col(EDUCATION_ID_COLUMN)
            .map_elements(lambda value: scoped_id(source_state, "workplace", value), return_dtype=pl.Int64)
            .alias("sp_id"),
            pl.lit(place_type).alias("place_type"),
            pl.col("longitude").cast(pl.Float64),
            pl.col("latitude").cast(pl.Float64),
            pl.lit(source_state.upper()).alias("source_state"),
        )
        .unique("sp_id")
    )


def _network_pairs(source, network_kind: str) -> Iterator[tuple[str, str]]:
    if network_kind == "work":
        reader = csv.DictReader(source)
        if reader.fieldnames != ["", "source", "target"]:
            raise ValueError(f"Unexpected work-network columns: {reader.fieldnames}")
        for row in reader:
            if row["source"] and row["target"]:
                yield row["source"].strip(), row["target"].strip()
        return
    for row in csv.reader(source):
        values = [value.strip() for value in row if value.strip()]
        for target in values[1:]:
            yield values[0], target


def _network_row(state: str, network_kind: str, left: str, right: str) -> dict[str, object] | None:
    if not left or not right or left == right:
        return None
    person_id_a, person_id_b = sorted((scoped_id(state, "person", left), scoped_id(state, "person", right)))
    if person_id_a == person_id_b:
        return None
    return {
        "person_id_a": person_id_a,
        "person_id_b": person_id_b,
        "network_kind": network_kind,
        "source_state": state.upper(),
    }


def _network_frames(archive: Path, state: str, batch_size: int) -> Iterator[pl.DataFrame]:
    rows: list[dict[str, object]] = []
    emitted = False
    schema = {
        "person_id_a": pl.Int64,
        "person_id_b": pl.Int64,
        "network_kind": pl.String,
        "source_state": pl.String,
    }
    with ZipFile(archive) as source_zip:
        members = source_zip.namelist()
        for network_kind in PERSON_SOCIAL_NETWORK_KINDS:
            member = find_member(members, network_kind, state)
            with source_zip.open(member) as binary_source:
                with TextIOWrapper(binary_source, encoding="utf-8", newline="") as source:
                    for left, right in _network_pairs(source, network_kind):
                        if row := _network_row(state, network_kind, left, right):
                            rows.append(row)
                        if len(rows) >= batch_size:
                            yield pl.DataFrame(rows, schema=schema)
                            emitted = True
                            rows.clear()
    if rows or not emitted:
        yield pl.DataFrame(rows, schema=schema)


def social_network_rows(archive: Path, state: str) -> pl.DataFrame:
    """Read and canonicalize person-person potential ties for small inputs and tests."""
    return pl.concat(_network_frames(archive, state, 100_000)).unique(["person_id_a", "person_id_b", "network_kind"])


def _write_batches(frames: Iterable[pl.DataFrame], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    writer = None
    rows = 0
    try:
        for frame in frames:
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += frame.height
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"Source produced no batches for {destination.name}")
    temporary.replace(destination)
    return rows


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _write_social_networks(archive: Path, state: str, persons: Path, destination: Path) -> tuple[int, int]:
    raw_path = destination.with_name("social_networks_raw.parquet")
    raw_count = _write_batches(_network_frames(archive, state, 100_000), raw_path)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT DISTINCT r.* FROM read_parquet('{_sql_path(raw_path)}') r "
            f"SEMI JOIN read_parquet('{_sql_path(persons)}') a ON r.person_id_a = a.sp_id "
            f"SEMI JOIN read_parquet('{_sql_path(persons)}') b ON r.person_id_b = b.sp_id) "
            f"TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        output_count = connection.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(destination)}')").fetchone()[
            0
        ]
        return output_count, raw_count - output_count
    finally:
        connection.close()
        raw_path.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_state_tables(
    archive: Path,
    state: str,
    output_dir: Path,
    *,
    education_archive: Path,
    batch_size: int = 250_000,
) -> dict[str, object]:
    """Build places, households, persons, and timeless social ties for one state."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not education_archive.is_file():
        raise FileNotFoundError(education_archive)
    state = state.upper()
    state_dir = output_dir / f"source_state={state}"

    with ZipFile(archive) as source_zip, ZipFile(education_archive) as education_zip, TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        population_path = _extract_member(
            source_zip,
            find_member(source_zip.namelist(), "population", state),
            temporary_root / "population.gpkg",
        )
        workplace_path = _extract_member(
            source_zip,
            find_member(source_zip.namelist(), "workplace", state),
            temporary_root / "workplace.gpkg",
        )
        source_population_rows = _feature_count(population_path)
        person_path = state_dir / "persons.parquet"
        _write_batches(
            (person_rows(state, frame) for frame in _iter_geopackage(population_path, PERSON_COLUMNS, batch_size)),
            person_path,
        )
        workplace_parquet = temporary_root / "workplaces.parquet"
        _write_batches(
            (workplace_rows(state, frame) for frame in _iter_geopackage(workplace_path, WORKPLACE_COLUMNS, batch_size)),
            workplace_parquet,
        )

        education_paths: list[Path] = []
        for kind, place_type in (("school", "School"), ("daycare", "Daycare")):
            member = f"{state.lower()}_{kind}_id.gpkg"
            if member not in education_zip.namelist():
                raise ValueError(f"Education archive is missing {member}")
            education_path = _extract_member(education_zip, member, temporary_root / f"{kind}.gpkg")
            destination = temporary_root / f"{kind}.parquet"
            _write_batches(
                (
                    education_rows(state, frame, place_type)
                    for frame in _iter_geopackage(
                        education_path,
                        [EDUCATION_ID_COLUMN],
                        batch_size,
                        read_geometry=True,
                    )
                ),
                destination,
            )
            education_paths.append(destination)

        connection = duckdb.connect()
        try:
            hh_path = state_dir / "hh.parquet"
            places_path = state_dir / "places.parquet"
            extra_places = " ".join(
                f"UNION ALL SELECT * FROM read_parquet('{_sql_path(path)}')" for path in education_paths
            )
            connection.execute(
                f"COPY (SELECT sp_hh_id AS sp_id, sp_hh_id AS sp_home_id, count(*)::BIGINT AS hh_size, "
                f"any_value(household_type) AS household_type, any_value(source_state) AS source_state "
                f"FROM read_parquet('{_sql_path(person_path)}') GROUP BY sp_hh_id) "
                f"TO '{_sql_path(hh_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.execute(
                f"COPY (SELECT sp_hh_id AS sp_id, 'Household' AS place_type, "
                f"any_value(home_longitude) AS longitude, any_value(home_latitude) AS latitude, "
                f"any_value(source_state) AS source_state FROM read_parquet('{_sql_path(person_path)}') "
                f"GROUP BY sp_hh_id UNION ALL SELECT * FROM read_parquet('{_sql_path(workplace_parquet)}') "
                f"{extra_places}) TO '{_sql_path(places_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()

    social_count, excluded_endpoints = _write_social_networks(
        archive, state, person_path, state_dir / "social_networks.parquet"
    )
    outputs = {name: state_dir / f"{name}.parquet" for name in TABLE_NAMES}
    output_population_rows = pl.scan_parquet(person_path).select(pl.len()).collect().item()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "inputs": {
            "population_archive": {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            "education_archive": {
                "path": str(education_archive),
                "size_bytes": education_archive.stat().st_size,
                "sha256": sha256_file(education_archive),
            },
        },
        "source_quality": {
            "population_rows": source_population_rows,
            "excluded_people_without_person_or_household_id": source_population_rows - output_population_rows,
        },
        "assignments": {
            "cross_state_home_work_or_school_pairs": "not inferable from variable-width source identifiers",
            "education_place_assignments_available": True,
            "activity_assignment_column": "sp_work_id",
            "activity_assignment_type_column": "activity_assignment_kind",
        },
        "social_networks": {
            "cross_state_ties": False,
            "semantics": "state-archive-local person-person potential ties only",
            "excluded_unresolved_endpoint_rows": excluded_endpoints,
            "excluded_source_networks": {"work": "non-person-source work memberships; not person-person social ties"},
        },
        "tables": {
            name: {
                "path": str(path),
                "rows": social_count
                if name == "social_networks"
                else pl.scan_parquet(path).select(pl.len()).collect().item(),
                "sha256": sha256_file(path),
                "schema": {key: str(value) for key, value in pl.scan_parquet(path).collect_schema().items()},
            }
            for name, path in outputs.items()
        },
    }
    _write_manifest(state_dir / "manifest.json", manifest)
    return manifest
