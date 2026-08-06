from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parents[1] / "casmsocial" / "datasets" / "colorado_front_range" / "profiles"
FOUR_CBSAS = {"14540", "19740", "22660", "24540"}
SIX_CBSAS = FOUR_CBSAS | {"17820", "39380"}


def _profiles() -> dict[str, dict]:
    return {path.stem: yaml.safe_load(path.read_text()) for path in sorted(PROFILE_DIR.glob("*.yaml"))}


def test_colorado_profiles_are_standalone_versioned_contracts():
    profiles = _profiles()
    assert set(profiles) == {
        "example-1k",
        "example-10k",
        "north-corridor-full",
        "six-metro-full",
    }
    assert len({profile["profile_id"] for profile in profiles.values()}) == 4
    for profile in profiles.values():
        assert profile["schema_version"] == 1
        assert profile["profile_version"] == "1.0.0"
        assert profile["sources"]["time_use"] == "bls-atus-2024"
        assert profile["population"]["require_endpoint_complete_social_ties"] is True
        assert profile["runtime"]["agent_log_enabled"] is False
        assert profile["runtime"]["aggregate_diagnostics_enabled"] is True
        assert profile["governance"]["publish_inputs"] is False
        assert profile["governance"]["calibrated"] is False


def test_supported_profiles_preserve_current_four_cbsa_scope():
    profiles = _profiles()
    for name in ("example-1k", "example-10k", "north-corridor-full"):
        profile = profiles[name]
        assert profile["release_status"] == "supported"
        assert set(profile["geography"]["home_cbsa_codes"]) == FOUR_CBSAS
    assert profiles["example-1k"]["population"]["person_limit"] == 1_000
    assert profiles["example-10k"]["population"]["person_limit"] == 10_000
    assert profiles["north-corridor-full"]["population"]["person_limit"] is None


def test_six_metro_profile_is_planned_until_acceptance_blockers_pass():
    profile = _profiles()["six-metro-full"]
    assert profile["release_status"] == "planned"
    assert set(profile["geography"]["home_cbsa_codes"]) == SIX_CBSAS
    assert profile["routing"]["destination_capacity_multiplier"] is None
    assert profile["routing"]["full_population_capacity_multiplier"] is None
    assert set(profile["release_blockers"]) == {
        "administrative_osm_cbsa_assignment",
        "colorado_springs_pueblo_destination_coverage",
        "six_metro_capacity_sensitivity",
        "six_metro_full_routing_acceptance",
    }
