"""Keep the shipped GUI presentation manifest aligned with the catalog record.

casmservice computes the manifest's scientific_contract_digest
(scripts/presentation_manifest.py). These checks catch the common drift without
importing casmservice: version, channels, views and field names.
"""

import json
from pathlib import Path

import tomllib

from scripts.register_casmsocial import CASMSOCIAL_MODEL_NAME, _build_model_record

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "presentation" / "casmsocial.runtime.json"
VIEW_FOR_COMPONENT = {"table": "table", "time-series": "time_series", "summary": "summary"}


def _version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_manifest_names_the_registered_model_version():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["model_ref"] == {"model_name": CASMSOCIAL_MODEL_NAME, "model_version": _version()}


def test_manifest_binds_only_declared_channels_views_and_fields():
    manifest = json.loads(MANIFEST.read_text())
    model = _build_model_record(_version(), "https://example.test/casmsocial")
    descriptors = model["model_metadata"]["output_descriptors"]["channels"]
    outputs = model["model_outputs_schema"]["properties"]
    parameters = model["model_parameters_schema"]["properties"]
    for panel in manifest["panels"]:
        if panel["component_id"] == "scalar-form":
            assert {f["field"] for f in panel["fields"]} <= set(parameters)
            continue
        channel = panel["channel"]
        descriptor = descriptors[channel]
        assert VIEW_FOR_COMPONENT[panel["component_id"]] in descriptor["views"]
        row = outputs[channel]["items"]["properties"]
        if panel["component_id"] == "table":
            assert {c["field"] for c in panel["columns"]} <= set(row)
        else:
            assert panel["time_field"] == descriptor["time"]["field"]
            assert {m["field"] for m in panel["metrics"]} <= set(descriptor["metrics"])


def test_descriptors_name_fields_in_the_row_schemas():
    model = _build_model_record(_version(), "https://example.test/casmsocial")
    descriptors = model["model_metadata"]["output_descriptors"]["channels"]
    outputs = model["model_outputs_schema"]["properties"]
    assert set(descriptors) == set(outputs) == set(model["model_metadata"]["observation_channels"])
    for channel, descriptor in descriptors.items():
        row = outputs[channel]["items"]["properties"]
        named = set(descriptor.get("metrics", {}))
        if "time" in descriptor:
            named.add(descriptor["time"]["field"])
        if "entity_id_field" in descriptor:
            named.add(descriptor["entity_id_field"])
        spatial = descriptor.get("spatial", {})
        named |= {spatial[key] for key in ("latitude_field", "longitude_field") if key in spatial}
        assert named <= set(row), channel


def test_agent_log_geography_is_place_latitude_longitude_not_model_xy():
    model = _build_model_record(_version(), "")
    spatial = model["model_metadata"]["output_descriptors"]["channels"]["agent_log"]["spatial"]
    assert spatial == {
        "kind": "geographic",
        "datum": "WGS84",
        "unit": "degree",
        "latitude_field": "latitude",
        "longitude_field": "longitude",
    }
    row = model["model_outputs_schema"]["properties"]["agent_log"]["items"]["properties"]
    assert row["latitude"]["type"] == row["longitude"]["type"] == ["number", "null"]


def test_parameter_defaults_match_casmpop_defaults():
    parameters = _build_model_record(_version(), "")["model_parameters_schema"]["properties"]
    assert {name: spec["default"] for name, spec in parameters.items()} == {
        "random.seed": 42,
        "duration.hours": 168,
        "time.step.minutes": 60,
        "observers.schedule_occupancy_log.enabled": False,
        "observers.social_interaction_log.enabled": False,
    }
