"""Create a deterministic, agent-level DC metro fixture for CASMSocial."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from casmsocial.ducklake_utils import get_ducklake_connection

DEFAULT_FIXTURE_PATH = Path("testdata/dc_metro_synthetic_100_households")
HOUSEHOLD_COUNT = 100
DUCKDB_TYPE_NAMES = {
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_table(path: Path, rows: list[dict], schema: pa.Schema) -> dict:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return {
        "file": str(path.name),
        "format": "parquet",
        "compression": "zstd",
        "rows": table.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "columns": [
            {
                "name": field.name,
                "type": DUCKDB_TYPE_NAMES[field.type],
                "nullable": field.nullable,
            }
            for field in schema
        ],
    }


def build_fixture(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> dict:
    """Write 100 synthetic households, their people, places, and daily plans."""
    fixture_path = fixture_path.expanduser()
    tables_path = fixture_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

    places: list[dict] = []
    households: list[dict] = []
    persons: list[dict] = []
    activities: list[dict] = []
    social_networks: list[dict] = []
    workplaces = [200_001 + index for index in range(12)]
    schools = [300_001 + index for index in range(4)]

    for index, workplace_id in enumerate(workplaces):
        places.append({
            "sp_id": workplace_id,
            "rank": 0,
            "place_type": "Workplace",
            "place_name": f"dc-metro-work-{index + 1:02}",
            "latitude": 38.82 + (index % 4) * 0.025,
            "longitude": -77.10 + (index // 4) * 0.03,
        })
    for index, school_id in enumerate(schools):
        places.append({
            "sp_id": school_id,
            "rank": 0,
            "place_type": "School",
            "place_name": f"dc-metro-school-{index + 1:02}",
            "latitude": 38.85 + (index % 2) * 0.035,
            "longitude": -77.08 + (index // 2) * 0.035,
        })

    person_id = 1
    for household_index in range(HOUSEHOLD_COUNT):
        home_id = 100_001 + household_index
        household_size = 1 + (household_index % 4)
        latitude = 38.79 + (household_index % 10) * 0.012
        longitude = -77.16 + (household_index // 10) * 0.016
        places.append({
            "sp_id": home_id,
            "rank": 0,
            "place_type": "Household",
            "place_name": f"dc-metro-home-{household_index + 1:03}",
            "latitude": latitude,
            "longitude": longitude,
        })
        households.append({
            "sp_id": home_id,
            "sp_home_id": home_id,
            "hh_size": household_size,
            "hh_income": float(45_000 + (household_index % 9) * 12_500),
            "hh_type": "family" if household_size > 1 else "single",
            "hh_race": "synthetic_unspecified",
            "hh_age": 30 + (household_index % 35),
        })
        household_person_ids: list[int] = []
        for member_index in range(household_size):
            age = 8 + ((household_index * 7 + member_index * 13) % 68)
            if member_index == 0 and age < 18:
                age = 28 + (household_index % 30)
            work_id = workplaces[(person_id - 1) % len(workplaces)] if age >= 18 else None
            school_id = schools[(person_id - 1) % len(schools)] if age < 18 else None
            destination = work_id or school_id or home_id
            destination_activity = 1 if work_id else 2 if school_id else 0
            persons.append({
                "sp_id": person_id,
                "sp_hh_id": home_id,
                "sp_work_id": work_id,
                "sp_school_id": school_id,
            })
            activities.extend([
                {
                    "sp_persons_id": person_id,
                    "activity_id": 0,
                    "activity_sequence": 0,
                    "starttime_min": 0,
                    "endtime_min": 480,
                    "sp_act_id": home_id,
                },
                {
                    "sp_persons_id": person_id,
                    "activity_id": destination_activity,
                    "activity_sequence": 1,
                    "starttime_min": 480,
                    "endtime_min": 1020,
                    "sp_act_id": destination,
                },
                {
                    "sp_persons_id": person_id,
                    "activity_id": 0,
                    "activity_sequence": 2,
                    "starttime_min": 1020,
                    "endtime_min": 1439,
                    "sp_act_id": home_id,
                },
            ])
            household_person_ids.append(person_id)
            person_id += 1

        for offset, person_id_a in enumerate(household_person_ids):
            for person_id_b in household_person_ids[offset + 1 :]:
                social_networks.append({
                    "person_id_a": person_id_a,
                    "person_id_b": person_id_b,
                    "network_kind": "household",
                    "tie_strength": 1.0,
                })

    schemas = {
        "persons": pa.schema([
            ("sp_id", pa.int64()),
            ("sp_hh_id", pa.int64()),
            ("sp_work_id", pa.int64()),
            ("sp_school_id", pa.int64()),
        ]),
        "hh": pa.schema([
            ("sp_id", pa.int64()),
            ("sp_home_id", pa.int64()),
            ("hh_size", pa.int32()),
            ("hh_income", pa.float64()),
            ("hh_type", pa.string()),
            ("hh_race", pa.string()),
            ("hh_age", pa.int32()),
        ]),
        "activities": pa.schema([
            ("sp_persons_id", pa.int64()),
            ("activity_id", pa.int32()),
            ("activity_sequence", pa.int32()),
            ("starttime_min", pa.int32()),
            ("endtime_min", pa.int32()),
            ("sp_act_id", pa.int64()),
        ]),
        "places": pa.schema([
            ("sp_id", pa.int64()),
            ("rank", pa.int32()),
            ("place_type", pa.string()),
            ("place_name", pa.string()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
        ]),
        "social_networks": pa.schema([
            ("person_id_a", pa.int64()),
            ("person_id_b", pa.int64()),
            ("network_kind", pa.string()),
            ("tie_strength", pa.float64()),
        ]),
    }
    data = {
        "persons": persons,
        "hh": households,
        "activities": activities,
        "places": places,
        "social_networks": social_networks,
    }
    manifest_tables = []
    for name, rows in data.items():
        output = tables_path / f"{name}.parquet"
        metadata = _write_table(output, rows, schemas[name])
        manifest_tables.append({
            "name": name,
            "source_table": f"dc_metro_synthetic_100.{name}",
            "file": f"tables/{metadata.pop('file')}",
            **metadata,
        })

    manifest = {
        "version": 1,
        "fixture_id": "dc_metro_synthetic_100_households",
        "description": "Deterministic fictional DC metro population for local CASMSocial scenario tests.",
        "generated_by": "scripts/create_dc_metro_synthetic_fixture.py",
        "data_classification": {
            "status": "synthetic_fixture",
            "sharing_boundary": "Local CASMSocial development and simulation only.",
            "notes": [
                "This fixture is generated from code and is not derived from OSF microdata.",
                "Coordinates are fictional points for simulation mechanics, not real homes, workplaces, or schools.",
            ],
        },
        "source": {"schema": "dc_metro_synthetic_100", "reference_area": "fictional DC metro"},
        "tables": manifest_tables,
    }
    (fixture_path / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def validate_fixture(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> dict:
    """Validate generated checksums, row counts, and schema names from the manifest."""
    fixture_path = fixture_path.expanduser()
    manifest = yaml.safe_load((fixture_path / "manifest.yaml").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Fixture manifest must be a mapping")
    for table in manifest["tables"]:
        table_path = fixture_path / table["file"]
        if _sha256(table_path) != table["sha256"]:
            raise ValueError(f"Checksum mismatch: {table['name']}")
        parquet_table = pq.read_table(table_path)
        if parquet_table.num_rows != table["rows"]:
            raise ValueError(f"Row-count mismatch: {table['name']}")
        if [field.name for field in parquet_table.schema] != [column["name"] for column in table["columns"]]:
            raise ValueError(f"Schema mismatch: {table['name']}")
    return manifest


def materialize_fixture(fixture_path: Path, ducklake_path: Path) -> dict[str, int]:
    """Load the validated fixture Parquet files into a local DuckLake catalog."""
    manifest = validate_fixture(fixture_path)
    schema_name = manifest["source"]["schema"]
    connection = get_ducklake_connection(ducklake_path)
    row_counts: dict[str, int] = {}
    try:
        connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
        for table in manifest["tables"]:
            name = table["name"]
            connection.execute(
                f'CREATE OR REPLACE TABLE "{schema_name}"."{name}" AS SELECT * FROM read_parquet(?)',
                [str(fixture_path / table["file"])],
            )
            row_counts[name] = connection.execute(f'SELECT count(*) FROM "{schema_name}"."{name}"').fetchone()[0]
    finally:
        connection.close()
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--ducklake-path", type=Path)
    args = parser.parse_args()
    manifest = build_fixture(args.fixture_path)
    validate_fixture(args.fixture_path)
    print(
        f"Created {manifest['fixture_id']}: "
        + ", ".join(f"{table['name']}={table['rows']}" for table in manifest["tables"])
    )
    if args.ducklake_path:
        row_counts = materialize_fixture(args.fixture_path, args.ducklake_path)
        print("Materialized DuckLake: " + ", ".join(f"{name}={rows}" for name, rows in row_counts.items()))


if __name__ == "__main__":
    main()
