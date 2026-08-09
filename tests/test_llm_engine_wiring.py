"""Engine-level tests for ``LLMBehaviorEngine`` adapter selection.

These tests exercise ``LLMBehaviorEngine._build_adapter_from_config``, the
classmethod that maps an engine config dict to a constructed adapter
instance. Verifying the dispatch directly rather than through ``__init__``
keeps the tests free of MPI / agent / projection setup overhead — the
classmethod has no dependency on the surrounding ``Person`` runtime.

Coverage:
* default config produces a :class:`LocalBehaviorLLMAdapter`,
* explicit ``adapter="local"`` produces the same,
* ``adapter="anthropic"`` constructs a remote adapter wired to the
  configured model and token budget,
* ``adapter_instance`` short-circuits the named-kind dispatch entirely,
* unknown kinds raise :class:`UnsupportedAdapterKindError` with the
  bad value preserved.

The ``anthropic`` SDK is stubbed via :data:`sys.modules` for the remote-
kind test so the test suite can run without the optional ``[remote]``
extra installed; the stub is restored in a fixture finalizer.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from casmsocial.llm_adapter import (
    BehaviorLLMAdapter,
    LocalBehaviorLLMAdapter,
    UnsupportedAdapterKindError,
)
from casmsocial.person import LLMBehaviorEngine


def _local_config(**overrides: Any) -> dict[str, Any]:
    """Minimal config for the local-adapter branch."""
    base: dict[str, Any] = {
        "adapter": "local",
        "adapter_instance": None,
        "signal_cap": 1.5,
        "memory_decay": 0.65,
        "activity_semantics_overrides": {},
    }
    base.update(overrides)
    return base


# ----------------------------- local-kind dispatch --------------------------


def test_build_adapter_default_kind_returns_local():
    """When ``adapter`` is omitted, the default ``"local"`` kind is used."""
    config = _local_config()
    del config["adapter"]
    adapter = LLMBehaviorEngine._build_adapter_from_config(config)
    assert isinstance(adapter, LocalBehaviorLLMAdapter)


def test_build_adapter_local_kind_returns_local():
    adapter = LLMBehaviorEngine._build_adapter_from_config(_local_config())
    assert isinstance(adapter, LocalBehaviorLLMAdapter)
    # Returned adapter respects config-provided parameters.
    assert adapter.signal_cap == 1.5
    assert adapter.memory_decay == 0.65


def test_build_adapter_local_kind_passes_overrides():
    adapter = LLMBehaviorEngine._build_adapter_from_config(
        _local_config(
            signal_cap=2.0,
            memory_decay=0.4,
            activity_semantics_overrides={"social_ids": [3, 5]},
        )
    )
    assert isinstance(adapter, LocalBehaviorLLMAdapter)
    assert adapter.signal_cap == 2.0
    assert adapter.memory_decay == 0.4
    assert adapter.activity_semantics_overrides == {"social_ids": [3, 5]}


def test_build_adapter_returns_instance_satisfying_protocol():
    """The constructed adapter must satisfy BehaviorLLMAdapter."""
    adapter = LLMBehaviorEngine._build_adapter_from_config(_local_config())
    assert isinstance(adapter, BehaviorLLMAdapter)


# ----------------------------- adapter_instance escape hatch ----------------


def test_adapter_instance_overrides_named_kind():
    """A pre-built adapter under ``adapter_instance`` is returned as-is,
    even when ``adapter`` is set to a different (or unknown) kind."""
    sentinel = LocalBehaviorLLMAdapter(signal_cap=2.5)
    config = _local_config(adapter="anthropic", adapter_instance=sentinel)
    adapter = LLMBehaviorEngine._build_adapter_from_config(config)
    assert adapter is sentinel


def test_adapter_instance_short_circuits_unknown_kind():
    """Even if ``adapter`` is bogus, ``adapter_instance`` overrides the dispatch."""
    sentinel = LocalBehaviorLLMAdapter()
    config = _local_config(adapter="completely-made-up", adapter_instance=sentinel)
    adapter = LLMBehaviorEngine._build_adapter_from_config(config)
    assert adapter is sentinel


# ----------------------------- unknown-kind error ---------------------------


def test_unknown_adapter_kind_raises_typed_error():
    config = _local_config(adapter="telepathy", adapter_instance=None)
    with pytest.raises(UnsupportedAdapterKindError) as excinfo:
        LLMBehaviorEngine._build_adapter_from_config(config)
    assert excinfo.value.kind == "telepathy"
    # Error message lists the valid kinds.
    msg = str(excinfo.value)
    assert "local" in msg
    assert "anthropic" in msg


# ----------------------------- anthropic-kind dispatch ----------------------


@pytest.fixture
def stub_anthropic_module():
    """Inject a no-op ``anthropic`` module into ``sys.modules`` for the
    duration of a single test, so the lazy ``from anthropic import Anthropic``
    inside :class:`AnthropicProvider.__init__` doesn't require the optional
    ``[remote]`` extra to be installed.

    The original module (if any) is restored in the finalizer so other tests
    that genuinely need the real SDK are unaffected.
    """
    original = sys.modules.get("anthropic")
    fake = types.ModuleType("anthropic")

    class _FakeAnthropic:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.messages = types.SimpleNamespace(create=lambda **__kw: None)

    fake.Anthropic = _FakeAnthropic
    sys.modules["anthropic"] = fake
    try:
        yield fake
    finally:
        if original is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = original


def test_anthropic_kind_constructs_anthropic_adapter(stub_anthropic_module):
    """``adapter="anthropic"`` dispatches to AnthropicBehaviorLLMAdapter."""
    from casmsocial.remote_llm_adapter import (
        AnthropicBehaviorLLMAdapter,
        RemoteBehaviorLLMAdapter,
    )

    config: dict[str, Any] = {
        "adapter": "anthropic",
        "adapter_instance": None,
        "anthropic_model": "claude-haiku-4-5-20251001",
        "anthropic_api_key": "test-key",
        "anthropic_max_tokens": 512,
    }
    adapter = LLMBehaviorEngine._build_adapter_from_config(config)
    assert isinstance(adapter, AnthropicBehaviorLLMAdapter)
    assert isinstance(adapter, RemoteBehaviorLLMAdapter)
    # Configured model and max_tokens flow through to the provider.
    assert adapter.provider.model == "claude-haiku-4-5-20251001"
    assert adapter.provider.max_tokens == 512


def test_anthropic_kind_uses_default_model_when_omitted(stub_anthropic_module):
    """If the model isn't specified, the default Haiku release is used."""
    from casmsocial.remote_llm_adapter import AnthropicBehaviorLLMAdapter

    config: dict[str, Any] = {"adapter": "anthropic", "adapter_instance": None}
    adapter = LLMBehaviorEngine._build_adapter_from_config(config)
    assert isinstance(adapter, AnthropicBehaviorLLMAdapter)
    assert adapter.provider.model == "claude-haiku-4-5-20251001"


def test_anthropic_kind_lazy_import_so_module_loads_without_anthropic():
    """Importing ``casmsocial.person`` must not require ``anthropic``.

    This is verified implicitly by the rest of the test file importing
    ``LLMBehaviorEngine`` at module level without a stubbed anthropic;
    if the lazy import in ``_build_adapter_from_config`` were not lazy,
    the import in this file's top-of-module imports would have failed.
    """
    # Sanity: the top-of-module import already happened.
    assert LLMBehaviorEngine is not None


# ----------------------------- configure() lifecycle ------------------------


def test_configure_round_trip_for_adapter_kind():
    """LLMBehaviorEngine.configure(adapter='anthropic') updates the class config."""
    original_config = dict(LLMBehaviorEngine._config)
    try:
        LLMBehaviorEngine.configure(adapter="anthropic")
        assert LLMBehaviorEngine._config["adapter"] == "anthropic"
    finally:
        LLMBehaviorEngine._config = original_config


def test_configure_round_trip_for_adapter_instance():
    """A pre-built adapter set via configure() survives until the next change."""
    original_config = dict(LLMBehaviorEngine._config)
    sentinel = LocalBehaviorLLMAdapter()
    try:
        LLMBehaviorEngine.configure(adapter_instance=sentinel)
        assert LLMBehaviorEngine._config["adapter_instance"] is sentinel
        # Build path resolves to the sentinel.
        adapter = LLMBehaviorEngine._build_adapter_from_config(LLMBehaviorEngine._config)
        assert adapter is sentinel
    finally:
        LLMBehaviorEngine._config = original_config
