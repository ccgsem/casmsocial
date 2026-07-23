from scripts.register_casmsocial import CASMSOCIAL_FACTORY_KEY, _build_scenarios, _load_scenario_specs


def test_register_casmsocial_loads_canonical_scenario_yaml_files():
    specs = _load_scenario_specs()

    assert set(specs) == {
        "base",
        "dmv",
        "dmv_100",
        "dc",
        "dc_5000",
        "dc_metro_synthetic_100",
    }
    for name, spec in specs.items():
        assert spec["description"]
        params = spec["parameters"]
        assert params["model.name"] == CASMSOCIAL_FACTORY_KEY
        assert params["places.table"]
        assert params["households.table"].endswith(".hh"), name
        assert params["persons.table"]
        assert params["activities.table"]


def test_build_scenarios_uses_yaml_parameters():
    scenarios = _build_scenarios()

    assert scenarios["dmv"]["households.table"] == "rti_synth_pop_v2_dmv.hh"
    assert scenarios["dmv_100"]["households.table"] == "rti_synth_pop_v2_dmv_100.hh"


def test_casmpop_does_not_publish_legacy_experiment_registry():
    import casmsocial.casmpop as casmpop

    assert not hasattr(casmpop, "experiment_parameters")
