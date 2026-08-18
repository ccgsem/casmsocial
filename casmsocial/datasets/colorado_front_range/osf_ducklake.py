"""Materialize validated OSF state partitions into one managed DuckLake.

This module is derived from ``mydatalakehouse.osf_synthetic_ducklake`` at
commit 4a9687de19ad192b97139f085d3e348dfe187cbd. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from casmsocial.datasets.colorado_front_range.osf_tables import TABLE_NAMES
from casmsocial.datasets.colorado_front_range.sources import sha256_file


def _sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def _manifest_path(catalog_path: Path) -> Path:
    return catalog_path.with_suffix(f"{catalog_path.suffix}.manifest.json")


def _write_json_atomic(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_state_partitions(state_table_dir: Path) -> list[dict[str, object]]:
    """Verify state manifests, table hashes, and cross-state schema consistency."""
    root = state_table_dir.expanduser().resolve()
    state_dirs = sorted(path for path in root.glob("source_state=*") if path.is_dir())
    if not state_dirs:
        raise FileNotFoundError(f"No state partitions found under {root}")

    validated: list[dict[str, object]] = []
    schemas: dict[str, dict[str, str]] = {}
    for state_dir in state_dirs:
        state = state_dir.name.partition("=")[2]
        manifest_path = state_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"State partition has no manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported state manifest: {manifest_path}")
        if manifest.get("state") != state:
            raise ValueError(f"State manifest does not match partition directory: {manifest_path}")
        tables = manifest.get("tables")
        if not isinstance(tables, dict) or set(tables) != set(TABLE_NAMES):
            raise ValueError(f"State manifest must describe exactly {', '.join(TABLE_NAMES)}: {manifest_path}")

        table_paths: dict[str, str] = {}
        for table_name in TABLE_NAMES:
            table = tables[table_name]
            if not isinstance(table, dict):
                raise ValueError(f"Invalid {table_name} manifest entry: {manifest_path}")
            table_path = state_dir / f"{table_name}.parquet"
            if not table_path.is_file():
                raise FileNotFoundError(table_path)
            if table.get("sha256") != sha256_file(table_path):
                raise ValueError(f"Table hash does not match state manifest: {table_path}")
            schema = table.get("schema")
            if not isinstance(schema, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in schema.items()
            ):
                raise ValueError(f"Invalid {table_name} schema in {manifest_path}")
            if table_name in schemas and schema != schemas[table_name]:
                raise ValueError(f"Inconsistent {table_name} schema in {manifest_path}")
            schemas.setdefault(table_name, schema)
            table_paths[table_name] = str(table_path)

        validated.append({
            "state": state,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "tables": table_paths,
        })
    return validated


def _read_rows(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, object]]:
    cursor = connection.execute(query)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _acceptance(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    checks = {
        "persons_without_household": connection.execute(
            "SELECT count(*) FROM persons p ANTI JOIN hh h ON p.sp_hh_id = h.sp_id"
        ).fetchone()[0],
        "households_without_home_place": connection.execute(
            "SELECT count(*) FROM hh h ANTI JOIN places p ON h.sp_home_id = p.sp_id"
        ).fetchone()[0],
        "activity_assignments_without_place": connection.execute(
            "SELECT count(*) FROM persons p ANTI JOIN places place ON p.sp_work_id = place.sp_id "
            "WHERE p.sp_work_id IS NOT NULL"
        ).fetchone()[0],
        "invalid_social_ties": connection.execute(
            "SELECT count(*) FROM social_networks WHERE person_id_a IS NULL OR person_id_b IS NULL "
            "OR person_id_a >= person_id_b OR network_kind IS NULL OR trim(network_kind) = ''"
        ).fetchone()[0],
        "duplicate_social_ties": connection.execute(
            "SELECT count(*) FROM (SELECT person_id_a, person_id_b, network_kind, count(*) records "
            "FROM social_networks GROUP BY ALL HAVING records > 1)"
        ).fetchone()[0],
        "social_ties_without_left_person": connection.execute(
            "SELECT count(*) FROM social_networks tie ANTI JOIN persons p ON tie.person_id_a = p.sp_id"
        ).fetchone()[0],
        "social_ties_without_right_person": connection.execute(
            "SELECT count(*) FROM social_networks tie ANTI JOIN persons p ON tie.person_id_b = p.sp_id"
        ).fetchone()[0],
    }
    required_zero = [
        "persons_without_household",
        "households_without_home_place",
        "invalid_social_ties",
        "duplicate_social_ties",
        "social_ties_without_left_person",
        "social_ties_without_right_person",
    ]
    return {
        "status": "passed" if all(checks[name] == 0 for name in required_zero) else "failed",
        "required_zero": required_zero,
        "checks": checks,
    }


def _table_source_sql(paths: list[str]) -> str:
    values = ", ".join(f"'{_sql_path(Path(path))}'" for path in paths)
    return f"read_parquet([{values}])"


def _cached_manifest(catalog_path: Path, data_path: Path, sources: list[dict[str, object]]) -> dict[str, object] | None:
    manifest_path = _manifest_path(catalog_path)
    if not (catalog_path.is_file() and data_path.is_dir() and manifest_path.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        isinstance(manifest, dict)
        and manifest.get("status") == "passed"
        and manifest.get("source_partitions") == sources
        and manifest.get("catalog_sha256") == sha256_file(catalog_path)
    ):
        return manifest
    return None


def build_ducklake(
    state_table_dir: Path,
    catalog_path: Path,
    data_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build and validate one managed DuckLake from canonical state partitions."""
    sources = validate_state_partitions(state_table_dir)
    catalog_path = catalog_path.expanduser().resolve()
    data_path = data_path.expanduser().resolve()
    if cached := _cached_manifest(catalog_path, data_path, sources):
        return {**cached, "resumed": True}

    manifest_path = _manifest_path(catalog_path)
    existing = [path for path in (catalog_path, data_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "DuckLake outputs exist but do not match current sources; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        catalog_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        if data_path.exists():
            shutil.rmtree(data_path)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=False)
    staging_catalog = catalog_path.with_suffix(f"{catalog_path.suffix}.building")
    staging_catalog.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("LOAD ducklake")
        connection.execute(
            f"ATTACH 'ducklake:{_sql_path(staging_catalog)}' AS osf_lake (DATA_PATH '{_sql_path(data_path)}')"
        )
        connection.execute("USE osf_lake")
        table_counts: dict[str, int] = {}
        for table_name in TABLE_NAMES:
            paths = [source["tables"][table_name] for source in sources]
            connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {_table_source_sql(paths)}")
            table_counts[table_name] = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        state_counts = _read_rows(
            connection,
            "SELECT source_state, count(*) persons FROM persons GROUP BY source_state ORDER BY source_state",
        )
        acceptance = _acceptance(connection)
        if acceptance["status"] != "passed":
            raise ValueError(f"DuckLake acceptance failed: {acceptance['checks']}")
    except Exception:
        connection.close()
        staging_catalog.unlink(missing_ok=True)
        if data_path.exists():
            shutil.rmtree(data_path)
        raise
    else:
        connection.close()

    staging_catalog.replace(catalog_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "resumed": False,
        "catalog": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "data_path": str(data_path),
        "states": [source["state"] for source in sources],
        "source_partitions": sources,
        "tables": table_counts,
        "persons_by_state": state_counts,
        "acceptance": acceptance,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
