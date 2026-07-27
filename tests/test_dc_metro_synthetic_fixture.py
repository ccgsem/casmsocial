from __future__ import annotations

import duckdb

from scripts.create_dc_metro_synthetic_fixture import build_fixture, validate_fixture


def test_dc_metro_synthetic_fixture_is_valid_and_referentially_complete(tmp_path):
    fixture_path = tmp_path / "dc_metro_synthetic_100_households"
    manifest = build_fixture(fixture_path)

    validate_fixture(fixture_path)
    table_rows = {table["name"]: table["rows"] for table in manifest["tables"]}
    assert table_rows == {
        "persons": 250,
        "hh": 100,
        "activities": 750,
        "places": 116,
        "social_networks": 250,
    }

    connection = duckdb.connect()
    try:
        missing_activity_places = connection.execute(
            """
            SELECT count(*)
            FROM read_parquet(?) AS activities
            LEFT JOIN read_parquet(?) AS places ON activities.sp_act_id = places.sp_id
            WHERE places.sp_id IS NULL
            """,
            [
                str(fixture_path / "tables" / "activities.parquet"),
                str(fixture_path / "tables" / "places.parquet"),
            ],
        ).fetchone()[0]
        assert missing_activity_places == 0
        invalid_network_rows = connection.execute(
            """
            SELECT count(*)
            FROM read_parquet(?)
            WHERE person_id_a >= person_id_b OR network_kind IS NULL
            """,
            [str(fixture_path / "tables" / "social_networks.parquet")],
        ).fetchone()[0]
        assert invalid_network_rows == 0
    finally:
        connection.close()
