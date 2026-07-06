import builtins

import dotenv

import casmsocial.factory as factory_mod


def test_load_models_from_dotenv_ignores_missing_env_var(monkeypatch):
    imported_modules = []
    original_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("plugin."):
            imported_modules.append(name)
            return object()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delenv("CASMSOCIAL_MODELS", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(builtins, "__import__", tracking_import)

    factory_mod.load_models_from_dotenv()

    assert imported_modules == []


def test_load_models_from_dotenv_imports_trimmed_modules(monkeypatch):
    imported_modules = []
    original_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("plugin."):
            imported_modules.append(name)
            return object()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("CASMSOCIAL_MODELS", " plugin.alpha,plugin.beta , , plugin.gamma ")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(builtins, "__import__", tracking_import)

    factory_mod.load_models_from_dotenv()

    assert imported_modules == ["plugin.alpha", "plugin.beta", "plugin.gamma"]
