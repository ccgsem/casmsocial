import json

import pytest

from casmsocial.grpc_control import ENDPOINT_FILENAME
from casmsocial.grpc_runner import run_submitted_model, start_runner


def test_submitted_model_loads_configured_plugins(monkeypatch):
    loaded: list[list[str]] = []

    monkeypatch.setattr("casmsocial.grpc_runner.load_builtin_models", lambda: None)
    monkeypatch.setattr("casmsocial.grpc_runner.load_models", lambda plugins: loaded.append(plugins))
    monkeypatch.setattr(
        "casmsocial.grpc_runner.Models.create_model",
        lambda name: (_ for _ in ()).throw(RuntimeError("stop after plugin loading")),
    )

    with pytest.raises(RuntimeError, match="stop after plugin loading"):
        run_submitted_model(
            "run-1",
            json.dumps({"model.name": "wake", "model.plugins": ["wake.plugin"]}).encode(),
            object(),
        )

    assert loaded == [["wake.plugin"]]


def test_submitted_model_propagates_missing_plugin_error(monkeypatch):
    monkeypatch.setattr("casmsocial.grpc_runner.load_builtin_models", lambda: None)
    monkeypatch.setattr(
        "casmsocial.grpc_runner.Models.create_model",
        lambda name: (_ for _ in ()).throw(AssertionError("model must not be created")),
    )

    with pytest.raises(ModuleNotFoundError, match="casmsocial_missing_plugin"):
        run_submitted_model(
            "run-1",
            json.dumps(
                {
                    "model.name": "wake",
                    "model.plugins": ["casmsocial_missing_plugin"],
                }
            ).encode(),
            object(),
        )


def test_runner_writes_combined_loopback_endpoint_manifest(tmp_path):
    control, flights = start_runner(tmp_path)
    try:
        manifest = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())
        assert manifest["control"]["address"].startswith("127.0.0.1:")
        assert manifest["control"]["protocol"] == "casm.runner.v1"
        assert manifest["flight"] == {
            "address": f"127.0.0.1:{flights.port}",
            "protocol": "arrow.flight",
        }
        assert (tmp_path / "arrow_endpoint.txt").read_text().strip() == f"127.0.0.1:{flights.port}"
    finally:
        control.stop(0).wait()
        flights.shutdown()
