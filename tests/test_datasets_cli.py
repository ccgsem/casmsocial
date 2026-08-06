import json

from typer.testing import CliRunner

from casmsocial.datasets.cli import app
from casmsocial.datasets.colorado_front_range import list_profiles, load_profile

runner = CliRunner()


def test_bundled_profiles_load_without_private_repository_dependency():
    assert list_profiles() == [
        "example-10k",
        "example-1k",
        "north-corridor-full",
        "six-metro-full",
    ]
    assert load_profile("example-10k").population.person_limit == 10_000


def test_cli_lists_profile_release_state_as_json():
    result = runner.invoke(app, ["colorado", "profiles", "--format", "json"])
    assert result.exit_code == 0, result.output
    profiles = {row["name"]: row for row in json.loads(result.output)}
    assert profiles["example-10k"]["release_status"] == "supported"
    assert profiles["six-metro-full"]["release_status"] == "planned"


def test_cli_shows_validated_profile():
    profile_result = runner.invoke(app, ["colorado", "show-profile", "example-1k", "--format", "json"])
    assert profile_result.exit_code == 0, profile_result.output
    assert json.loads(profile_result.output)["profile_id"] == "colorado-front-range-example-1k-v1"


def test_cli_rejects_unknown_profile():
    result = runner.invoke(app, ["colorado", "show-profile", "missing"])
    assert result.exit_code == 2
    assert "Unknown Colorado profile" in result.output


def test_cli_prints_osm_attribution_and_local_only_policy():
    result = runner.invoke(app, ["colorado", "osm-attribution"])

    assert result.exit_code == 0, result.output
    assert "© OpenStreetMap contributors" in result.output
    assert "Local build only" in result.output
    assert "ODbL" in result.output


def test_cli_lists_source_acquisition_policies():
    result = runner.invoke(app, ["colorado", "sources", "--format", "json"])
    assert result.exit_code == 0, result.output
    sources = {row["artifact_id"]: row for row in json.loads(result.output)}
    assert sources["osf-colorado-population"]["verification"] == "pinned_sha256"
    assert sources["bls-atus-2024-roster"]["access"] == "manual"


def test_cli_records_and_verifies_manual_source(tmp_path):
    destination = tmp_path / "raw" / "atus" / "2024" / "atusrost_2024.dat"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"staged roster")

    record_result = runner.invoke(
        app,
        ["colorado", "record", "bls-atus-2024-roster", "--data-dir", str(tmp_path), "--format", "json"],
    )
    assert record_result.exit_code == 0, record_result.output
    verify_result = runner.invoke(
        app,
        ["colorado", "verify", "bls-atus-2024-roster", "--data-dir", str(tmp_path), "--format", "json"],
    )
    assert verify_result.exit_code == 0, verify_result.output
    assert json.loads(verify_result.output)["status"] == "verified"


def test_cli_verify_missing_source_fails(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "verify", "osf-colorado-population", "--data-dir", str(tmp_path), "--format", "json"],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "missing"


def test_cli_build_osf_requires_verified_inputs(tmp_path):
    result = runner.invoke(app, ["colorado", "build-osf", "--data-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "osf-colorado-population is missing" in result.output
    assert "verify it before building" in result.output


def test_cli_build_ducklake_requires_state_partitions(tmp_path):
    result = runner.invoke(
        app,
        [
            "colorado",
            "build-ducklake",
            "--input-dir",
            str(tmp_path / "missing"),
            "--catalog",
            str(tmp_path / "lake" / "metadata.ducklake"),
            "--data-path",
            str(tmp_path / "lake" / "files"),
        ],
    )
    assert result.exit_code == 2
    assert "No state partitions found" in result.output


def test_cli_build_population_requires_verified_county_boundaries(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "build-population", "example-1k", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "census-2023-counties is missing" in result.output
    assert "verify it before building" in result.output


def test_cli_build_schedules_requires_verified_atus_extracts(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "build-schedules", "example-1k", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "bls-atus-2024-respondents is missing" in result.output
    assert "record/verify before building" in result.output


def test_cli_build_destinations_requires_verified_osm_extract(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "build-destinations", "example-1k", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "osm-geofabrik-colorado is missing" in result.output
    assert "fetch/verify before building" in result.output


def test_cli_build_runtime_requires_profile_product(tmp_path):
    result = runner.invoke(app, ["colorado", "build-runtime", "example-1k", "--data-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "manifest.json" in result.output


def test_cli_verify_runtime_requires_runtime_product(tmp_path):
    result = runner.invoke(app, ["colorado", "verify-runtime", "example-1k", "--data-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "manifest.json" in result.output


def test_cli_build_all_plan_reports_readiness_without_building(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "build-all", "example-1k", "--data-dir", str(tmp_path), "--plan", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["ready"] is False
    assert len(plan["sources"]) == 7
    assert plan["stages"][-1]["name"] == "runtime_verification"
    assert not (tmp_path / "local").exists()


def test_cli_build_all_lists_all_unverified_sources(tmp_path):
    result = runner.invoke(
        app,
        ["colorado", "build-all", "example-1k", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "osf-colorado-population (missing; fetch and verify)" in result.output
    assert "bls-atus-2024-roster (missing; download manually, record, and verify)" in result.output
