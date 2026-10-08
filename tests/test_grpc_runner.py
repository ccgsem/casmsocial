"""CASMSocial launcher integration tests for the standalone CASMSim runtime."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest
from casmsim.grpc_runner import ENDPOINT_FILENAME, SimulatorControlServicer
from casmsim.proto import casm_runner_pb2 as pb2

from casmsocial.adapters.runner import _ObservationBridge
from casmsocial.grpc_runner import run_casmsocial_model, start_runner


def _casmsocial_stubs(load_models_fn=None):
    main_stub = MagicMock()
    main_stub.load_builtin_models = MagicMock()
    factory_stub = MagicMock()
    factory_stub.load_models = MagicMock(side_effect=load_models_fn) if load_models_fn else MagicMock()
    model = MagicMock()
    model.start = MagicMock()
    factory_stub.Models.create_model.return_value = MagicMock(return_value=model)
    return main_stub, factory_stub


def test_submitted_model_loads_configured_plugins(monkeypatch):
    loaded: list[list[str]] = []
    main_stub, factory_stub = _casmsocial_stubs(lambda plugins: loaded.append(list(plugins)))
    monkeypatch.setitem(sys.modules, "casmsocial.__main__", main_stub)
    monkeypatch.setitem(sys.modules, "casmsocial.factory", factory_stub)
    run_casmsocial_model("run-1", json.dumps({"model.name": "wake", "model.plugins": ["wake.plugin"]}).encode(), MagicMock())
    assert loaded == [["wake.plugin"]]
    main_stub.load_builtin_models.assert_called_once()


def test_submitted_model_propagates_missing_plugin_error(monkeypatch):
    plugin_name = "casmsocial_missing_plugin"

    def raise_missing(plugins):
        raise ModuleNotFoundError(f"No module named '{plugins[0]}'")

    main_stub, factory_stub = _casmsocial_stubs(raise_missing)
    monkeypatch.setitem(sys.modules, "casmsocial.__main__", main_stub)
    monkeypatch.setitem(sys.modules, "casmsocial.factory", factory_stub)
    with pytest.raises(ModuleNotFoundError, match=plugin_name):
        run_casmsocial_model("run-1", json.dumps({"model.name": "wake", "model.plugins": [plugin_name]}).encode(), MagicMock())


def test_runner_failure_returns_sanitized_status(caplog):
    broker = MagicMock()
    servicer = SimulatorControlServicer(broker, lambda *_: (_ for _ in ()).throw(ValueError("credential=secret")))
    servicer._run_id = "run-1"
    with caplog.at_level("ERROR"):
        servicer._run("run-1", b"{}")
    response = servicer.GetState(pb2.GetStateRequest(run_id="run-1"), MagicMock())
    assert response.state == pb2.RUN_STATE_FAILED
    assert response.status_message == "Run failed (ValueError); see runner stderr.log."
    assert "credential=secret" not in response.status_message


def test_observation_bridge_contributes_no_model_output_tables():
    assert _ObservationBridge(MagicMock()).get_output_tables(MagicMock()) == {}


def test_runner_writes_combined_loopback_endpoint_manifest(tmp_path):
    control, flights = start_runner(tmp_path)
    try:
        manifest = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())
        assert manifest["control"]["address"].startswith("127.0.0.1:")
        assert manifest["control"]["protocol"] == "casm.runner.v1"
        assert manifest["flight"] == {"address": f"127.0.0.1:{flights.port}", "protocol": "arrow.flight"}
    finally:
        control.stop(0).wait()
        flights.shutdown()
