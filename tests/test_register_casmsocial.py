from scripts.register_casmsocial import (
    CASMSOCIAL_FACTORY_KEY,
    _build_model_record,
    _build_scenarios,
    _load_scenario_specs,
)


def test_register_casmsocial_loads_canonical_scenario_yaml_files():
    specs = _load_scenario_specs()

    assert set(specs) == {"wake_county_heat"}
    for name, spec in specs.items():
        assert spec["description"]
        params = spec["parameters"]
        assert params["model.name"] == CASMSOCIAL_FACTORY_KEY
        assert params["places.table"]
        assert params["households.table"]
        assert params["persons.table"]
        assert params["activities.table"]


def test_build_scenarios_uses_yaml_parameters():
    scenarios = _build_scenarios()

    assert set(scenarios) == {"wake_county_heat"}
    params = scenarios["wake_county_heat"]
    assert params["places.table"] == "wake_county_heat.places"
    assert params["households.table"] == "wake_county_heat.hh_1000_households"
    assert params["persons.table"] == "wake_county_heat.persons_1000_households"
    assert params["activities.table"] == "wake_county_heat.activities_1000_households"
    assert params["social_networks.table"] == ""
    assert params["social_networks.enabled"] is False


def test_build_model_record_reports_repository_license_and_public_contact():
    record = _build_model_record("2.5.5", "https://example.test/casmsocial")

    assert record["model_license"] == "MIT"
    assert record["model_authors"] == [{"name": "Jon C. Cline", "email": "jcline@mitre.org"}]


def test_casmpop_does_not_publish_legacy_experiment_registry():
    import casmsocial.casmpop as casmpop

    assert not hasattr(casmpop, "experiment_parameters")
