from pathlib import Path

import yaml

from scripts.register_casmsocial import CASMSOCIAL_FACTORY_KEY, _build_scenarios, _load_scenario_specs


def test_register_casmsocial_loads_canonical_scenario_yaml_files():
    specs = _load_scenario_specs()

    assert specs
    for name, spec in specs.items():
        assert spec["description"]
        params = spec["parameters"]
        assert params["model.name"] == CASMSOCIAL_FACTORY_KEY
        assert params["places.table"]
        assert params["households.table"], name
        assert params["persons.table"]
        assert params["activities.table"]


def test_build_scenarios_uses_yaml_parameters():
    specs = _load_scenario_specs()
    scenarios = _build_scenarios()

    assert set(scenarios) == set(specs)
    for name, spec in specs.items():
        assert scenarios[name] == spec["parameters"]


def test_registered_scenarios_use_public_wake_county_heat_fixture():
    specs = _load_scenario_specs()

    assert set(specs) == {"wake_county_heat"}
    params = specs["wake_county_heat"]["parameters"]
    assert params["places.table"] == "wake_county_heat.places"
    assert params["households.table"] == "wake_county_heat.hh_1000_households"
    assert params["persons.table"] == "wake_county_heat.persons_1000_households"
    assert params["activities.table"] == "wake_county_heat.activities_1000_households"
    assert params["contacts.table"] == ""
    assert params["contacts.enabled"] is False
    assert params["communication.enabled"] is False
    assert not any(str(value).startswith(("rti_synth_pop_v2_dmv", "rti_synth_pop_v2_dc")) for value in params.values())


def test_default_direct_run_config_uses_public_wake_county_heat_fixture():
    params = yaml.safe_load(Path("config/casmsocial.yaml").read_text(encoding="utf-8"))

    assert params["places.table"] == "wake_county_heat.places"
    assert params["households.table"] == "wake_county_heat.hh_1000_households"
    assert params["persons.table"] == "wake_county_heat.persons_1000_households"
    assert params["activities.table"] == "wake_county_heat.activities_1000_households"
    assert params["contacts.table"] == ""
    assert params["contacts.enabled"] is False
    assert params["communication.enabled"] is False


def test_casmpop_does_not_publish_legacy_experiment_registry():
    import casmsocial.casmpop as casmpop

    assert not hasattr(casmpop, "experiment_parameters")
