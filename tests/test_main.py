import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import casmsocial.__main__ as main_mod
from casmsocial.factory import ModelNotFoundError


def test_package_import_is_lightweight():
    sys.modules.pop("casmsocial.citysim.citysocialmodel", None)
    sys.modules.pop("casmsocial", None)

    package = importlib.import_module("casmsocial")

    assert package.__all__ == ["CitySocialModel", "SimTime"]
    assert "CitySocialModel" in dir(package)
    assert "casmsocial.citysim.citysocialmodel" not in sys.modules


def test_run_loads_plugins_and_starts_model(monkeypatch):
    fake_model = MagicMock()
    fake_creator = MagicMock(return_value=fake_model)
    fake_load_models = MagicMock()
    fake_info = MagicMock()
    fake_load_builtin_models = MagicMock()
    fake_comm = object()

    monkeypatch.setattr(main_mod, "load_models", fake_load_models)
    monkeypatch.setattr(main_mod, "load_builtin_models", fake_load_builtin_models)
    monkeypatch.setattr(main_mod.Models, "get_models", lambda: {"demo.model": fake_creator})
    monkeypatch.setattr(main_mod.Models, "create_model", lambda name: fake_creator)
    monkeypatch.setattr(main_mod.logger, "remove", MagicMock())
    monkeypatch.setattr(main_mod.logger, "add", MagicMock())
    monkeypatch.setattr(main_mod.logger, "info", fake_info)
    monkeypatch.setattr(main_mod.MPI, "COMM_WORLD", fake_comm)

    params = {
        "model.name": "demo.model",
        "model.plugins": ["plugin.alpha", "plugin.beta"],
    }

    main_mod.run(params)

    fake_load_builtin_models.assert_called_once_with()
    fake_load_models.assert_called_once_with(["plugin.alpha", "plugin.beta"])
    fake_creator.assert_called_once_with(fake_comm, params)
    fake_model.start.assert_called_once_with()


def test_run_starts_model_without_plugins(monkeypatch):
    fake_model = MagicMock()
    fake_creator = MagicMock(return_value=fake_model)
    fake_load_models = MagicMock()
    fake_load_builtin_models = MagicMock()
    fake_comm = object()

    monkeypatch.setattr(main_mod, "load_models", fake_load_models)
    monkeypatch.setattr(main_mod, "load_builtin_models", fake_load_builtin_models)
    monkeypatch.setattr(main_mod.Models, "get_models", lambda: {"demo.model": fake_creator})
    monkeypatch.setattr(main_mod.Models, "create_model", lambda name: fake_creator)
    monkeypatch.setattr(main_mod.logger, "remove", MagicMock())
    monkeypatch.setattr(main_mod.logger, "add", MagicMock())
    monkeypatch.setattr(main_mod.logger, "info", MagicMock())
    monkeypatch.setattr(main_mod.MPI, "COMM_WORLD", fake_comm)

    params = {"model.name": "demo.model"}

    main_mod.run(params)

    fake_load_builtin_models.assert_called_once_with()
    fake_load_models.assert_not_called()
    fake_creator.assert_called_once_with(fake_comm, params)
    fake_model.start.assert_called_once_with()


def test_run_can_filter_info_logs_to_rank_zero(monkeypatch):
    fake_model = MagicMock()
    fake_creator = MagicMock(return_value=fake_model)
    fake_load_builtin_models = MagicMock()
    fake_comm = MagicMock()
    fake_comm.Get_rank.return_value = 1
    fake_add = MagicMock()

    monkeypatch.setattr(main_mod, "load_builtin_models", fake_load_builtin_models)
    monkeypatch.setattr(main_mod.Models, "get_models", lambda: {"demo.model": fake_creator})
    monkeypatch.setattr(main_mod.Models, "create_model", lambda name: fake_creator)
    monkeypatch.setattr(main_mod.logger, "remove", MagicMock())
    monkeypatch.setattr(main_mod.logger, "add", fake_add)
    monkeypatch.setattr(main_mod.logger, "info", MagicMock())
    monkeypatch.setattr(main_mod.MPI, "COMM_WORLD", fake_comm)

    params = {
        "model.name": "demo.model",
        "logging.rank0_only": True,
    }

    main_mod.run(params)

    log_filter = fake_add.call_args.kwargs["filter"]
    assert log_filter({"level": SimpleNamespace(no=20)}) is False
    assert log_filter({"level": SimpleNamespace(no=30)}) is True


def test_run_returns_error_for_unknown_model(monkeypatch):
    fake_error = MagicMock()
    fake_load_builtin_models = MagicMock()

    def raise_model_not_found(name):
        raise ModelNotFoundError(name)

    monkeypatch.setattr(main_mod, "load_builtin_models", fake_load_builtin_models)
    monkeypatch.setattr(main_mod.Models, "get_models", lambda: {"demo.model": object()})
    monkeypatch.setattr(main_mod.Models, "create_model", raise_model_not_found)
    monkeypatch.setattr(main_mod.logger, "remove", MagicMock())
    monkeypatch.setattr(main_mod.logger, "add", MagicMock())
    monkeypatch.setattr(main_mod.logger, "info", MagicMock())
    monkeypatch.setattr(main_mod.logger, "error", fake_error)

    result = main_mod.run({"model.name": "missing.model"})

    assert result == 1
    fake_load_builtin_models.assert_called_once_with()
    fake_error.assert_called_once_with("Unsupported model type: missing.model")
