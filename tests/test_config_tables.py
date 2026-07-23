from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_TABLE_KEYS = (
    "places.table",
    "households.table",
    "persons.table",
    "activities.table",
)

# These configs have a matching Make target that creates a dedicated DuckLake
# fixture rather than using the developer-local default DuckLake below.
SELF_CONTAINED_FIXTURE_CONFIGS = {"dc_metro_synthetic_100.yaml"}


def _storage_path_for_table(storage_dir: Path, table_name: str) -> Path:
    parts = table_name.split(".")
    assert len(parts) == 2, f"Expected schema-qualified table name, got {table_name!r}"
    return storage_dir / parts[0] / parts[1]


def test_shipped_config_input_tables_exist_in_local_ducklake_storage():
    storage_dir = Path("data/datalakehouse/storage")
    if not storage_dir.exists():
        pytest.skip("Local DuckLake storage is not available")

    missing: list[str] = []
    for config_path in sorted(Path("config").glob("*.yaml")):
        if config_path.name in SELF_CONTAINED_FIXTURE_CONFIGS:
            continue

        params = yaml.safe_load(config_path.read_text()) or {}
        for key in CONFIG_TABLE_KEYS:
            table_name = params.get(key)
            assert table_name, f"{config_path}: missing {key}"

            table_path = _storage_path_for_table(storage_dir, str(table_name))
            if not table_path.exists():
                missing.append(f"{config_path}: {key}={table_name!r} -> {table_path}")

    assert missing == []
