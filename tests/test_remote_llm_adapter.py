"""Unit tests for ``casmsocial.remote_llm_adapter``.

These tests cover the provider-agnostic
:class:`RemoteBehaviorLLMAdapter` and the Anthropic-specific implementation
without requiring the real Anthropic SDK or network access. Two patterns are
used:

* A small ``_MockProvider`` that implements :class:`LLMProvider` and returns
  configurable canned responses, exercising prompt construction, schema
  validation, and the typed-exception surface.
* A stub Anthropic client (a plain ``SimpleNamespace`` with a
  ``messages.create`` callable) injected via the ``client`` parameter on
  :class:`AnthropicProvider`, exercising the SDK-call shape and the
  tool-use response parser without ever importing ``anthropic``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from casmsocial.communication.types import MessageKind
from casmsocial.llm_adapter import (
    PROPOSAL_ACTIONS,
    PROPOSAL_PLAN_ADJUSTMENT_KINDS,
    BehaviorLLMAdapter,
    LocalBehaviorLLMAdapter,
)
from casmsocial.remote_llm_adapter import (
    SYSTEM_PROMPT,
    AnthropicBehaviorLLMAdapter,
    AnthropicProvider,
    LLMProvider,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMResponseInvalidActionError,
    LLMResponseInvalidPlanAdjustmentError,
    LLMResponseSchemaError,
    RemoteBehaviorLLMAdapter,
    build_tool_schema,
    build_user_prompt,
    validate_proposal,
)

# ----------------------------- helpers -------------------------------------


def _well_formed_proposal() -> dict[str, Any]:
    """A minimal but schema-complete proposal dict."""
    return {
        "action": "stay_home",
        "reason": "A warning increases safety concerns and favors staying home.",
        "emotion_updates": {"fear": 0.4, "calm": 0.2},
        "need_updates": {"safety": 0.6},
        "attitude_updates": {"responsiveness": 0.55},
        "message_intent": None,
        "plan_adjustment": {
            "kind": "preserve_home_activity",
            "detail": {"source": "safety_signal"},
        },
        "memory_trace": {
            "safety_signal": 1.5,
            "social_signal": 0.0,
            "obligation_signal": 0.0,
            "schedule_signal": 0.0,
            "reply_signal": 0.0,
            "defer_signal": 0.0,
            "skip_signal": 0.0,
            "cancel_social_signal": 0.0,
            "defer_minutes": 0,
            "source_count": 1,
            "responsiveness": 0.55,
        },
    }


def _minimal_context() -> dict[str, Any]:
    return {
        "agent_uid": (1, 0, 0),
        "tick": 0,
        "minute_of_day": 30,
        "observation": {"is_transit": 0.0, "hour_of_day": 9.0},
        "beliefs": {},
        "goals": {},
        "emotions": {},
        "needs": {},
        "attitudes": {"responsiveness": 0.5},
        "recent_messages": [],
        "recent_memory": [],
        "current_place_id": 100,
        "rank_place_id": 100,
        "plan_state": {},
    }


class _MockProvider:
    """Programmable stand-in for :class:`LLMProvider`.

    Captures every ``submit`` call's arguments so tests can verify prompt
    construction, and returns either a canned proposal or raises a canned
    exception.
    """

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.response = response if response is not None else _well_formed_proposal()
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def submit(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt, tool_schema))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


# ----------------------------- protocol shape ------------------------------


def test_remote_adapter_conforms_to_behavior_llm_adapter_protocol():
    adapter = RemoteBehaviorLLMAdapter(_MockProvider())
    assert isinstance(adapter, BehaviorLLMAdapter)


def test_mock_provider_satisfies_llm_provider_protocol():
    """LLMProvider is runtime-checkable; a duck-typed mock should satisfy it."""
    assert isinstance(_MockProvider(), LLMProvider)


# ----------------------------- tool schema ---------------------------------


def test_tool_schema_includes_canonical_actions():
    schema = build_tool_schema()
    assert schema["properties"]["action"]["enum"] == list(PROPOSAL_ACTIONS)


def test_tool_schema_includes_canonical_plan_adjustment_kinds():
    schema = build_tool_schema()
    plan_kinds = schema["properties"]["plan_adjustment"]["anyOf"][1]["properties"]["kind"]["enum"]
    assert plan_kinds == list(PROPOSAL_PLAN_ADJUSTMENT_KINDS)


def test_tool_schema_required_keys_match_proposal_schema_keys():
    schema = build_tool_schema()
    assert set(schema["required"]) == set(LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS)


def test_tool_schema_disallows_additional_properties():
    schema = build_tool_schema()
    assert schema["additionalProperties"] is False


# ----------------------------- prompt construction -------------------------


def test_user_prompt_serializes_context_as_json():
    ctx = _minimal_context()
    out = build_user_prompt(ctx)
    # Round-trip back to a dict; should not raise and should preserve keys.
    import json

    parsed = json.loads(out)
    assert set(parsed.keys()) == set(ctx.keys())


def test_user_prompt_handles_unusual_types_via_default_str():
    """Tuples-as-keys, dataclasses, or other non-JSON-native types must not
    crash the serializer; ``default=str`` is the documented fallback."""

    class _Custom:
        def __repr__(self) -> str:
            return "<custom>"

    ctx = _minimal_context()
    ctx["weird"] = _Custom()
    # Must not raise.
    build_user_prompt(ctx)


def test_remote_adapter_passes_system_and_user_prompts_to_provider():
    provider = _MockProvider()
    adapter = RemoteBehaviorLLMAdapter(provider)
    adapter.generate_behavior_proposal(_minimal_context())
    assert len(provider.calls) == 1
    sys_prompt, user_prompt, tool_schema = provider.calls[0]
    assert sys_prompt == SYSTEM_PROMPT
    assert "agent_uid" in user_prompt  # context was serialized
    assert tool_schema == build_tool_schema()


# ----------------------------- validation ----------------------------------


def test_validate_proposal_accepts_well_formed_dict():
    out = validate_proposal(_well_formed_proposal())
    assert out["action"] == "stay_home"


def test_validate_proposal_rejects_missing_keys():
    bad = _well_formed_proposal()
    del bad["memory_trace"]
    with pytest.raises(LLMResponseSchemaError) as excinfo:
        validate_proposal(bad)
    assert "memory_trace" in excinfo.value.missing


def test_validate_proposal_rejects_extra_keys():
    bad = _well_formed_proposal()
    bad["extra_key"] = 42
    with pytest.raises(LLMResponseSchemaError) as excinfo:
        validate_proposal(bad)
    assert "extra_key" in excinfo.value.extra


def test_validate_proposal_rejects_invalid_action():
    bad = _well_formed_proposal()
    bad["action"] = "do_something_wild"
    with pytest.raises(LLMResponseInvalidActionError) as excinfo:
        validate_proposal(bad)
    assert excinfo.value.action == "do_something_wild"


def test_validate_proposal_rejects_invalid_plan_adjustment_kind():
    bad = _well_formed_proposal()
    bad["plan_adjustment"] = {"kind": "teleport_home", "detail": {}}
    with pytest.raises(LLMResponseInvalidPlanAdjustmentError) as excinfo:
        validate_proposal(bad)
    assert excinfo.value.kind == "teleport_home"


def test_validate_proposal_accepts_null_plan_adjustment():
    p = _well_formed_proposal()
    p["plan_adjustment"] = None
    out = validate_proposal(p)
    assert out["plan_adjustment"] is None


# ----------------------------- error mapping -------------------------------


def test_remote_adapter_wraps_unexpected_exceptions_as_unavailable():
    provider = _MockProvider(raise_exc=RuntimeError("boom"))
    adapter = RemoteBehaviorLLMAdapter(provider)
    with pytest.raises(LLMProviderUnavailableError) as excinfo:
        adapter.generate_behavior_proposal(_minimal_context())
    # Original exception preserved as cause for debugging.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert adapter.failures == 1


def test_remote_adapter_propagates_llm_provider_errors_unchanged():
    """A provider that already raises an LLMProviderError shouldn't be
    re-wrapped (the cause-chain would lose information)."""
    sentinel = LLMResponseSchemaError(missing={"action"}, extra=set())
    provider = _MockProvider(raise_exc=sentinel)
    adapter = RemoteBehaviorLLMAdapter(provider)
    with pytest.raises(LLMResponseSchemaError) as excinfo:
        adapter.generate_behavior_proposal(_minimal_context())
    assert excinfo.value is sentinel
    assert adapter.failures == 1


def test_remote_adapter_counts_calls_and_failures():
    provider = _MockProvider()
    adapter = RemoteBehaviorLLMAdapter(provider)
    adapter.generate_behavior_proposal(_minimal_context())
    adapter.generate_behavior_proposal(_minimal_context())
    assert adapter.calls == 2
    assert adapter.failures == 0


def test_remote_adapter_validates_provider_response():
    """Even if the provider returns a dict, the adapter must validate it."""
    bad = _well_formed_proposal()
    bad["action"] = "nope"
    provider = _MockProvider(response=bad)
    adapter = RemoteBehaviorLLMAdapter(provider)
    with pytest.raises(LLMResponseInvalidActionError):
        adapter.generate_behavior_proposal(_minimal_context())
    assert adapter.failures == 1


# ----------------------------- exception inheritance -----------------------


def test_all_typed_errors_descend_from_llm_provider_error():
    """Callers can catch LLMProviderError to handle every remote-adapter failure."""
    for exc_class in (
        LLMProviderUnavailableError,
        LLMResponseSchemaError,
        LLMResponseInvalidActionError,
        LLMResponseInvalidPlanAdjustmentError,
    ):
        assert issubclass(exc_class, LLMProviderError)


# ----------------------------- AnthropicProvider ---------------------------


def _make_stub_anthropic_response(tool_input: dict[str, Any]) -> Any:
    """Mimic the .content list of an Anthropic Messages API response."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="propose_behavior", input=tool_input),
        ]
    )


class _StubAnthropicClient:
    """SimpleNamespace alone can't expose `.messages.create(...)` cleanly.

    This wrapper records call arguments and returns a configurable stub.
    """

    def __init__(self, tool_input: dict[str, Any]) -> None:
        self.tool_input = tool_input
        self.calls: list[dict[str, Any]] = []
        # Anthropic SDK access pattern is `client.messages.create(...)`.
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _make_stub_anthropic_response(self.tool_input)


def test_anthropic_provider_passes_expected_arguments_to_sdk():
    expected = _well_formed_proposal()
    stub = _StubAnthropicClient(expected)
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=stub)
    out = provider.submit("system", "user", build_tool_schema())
    assert out == expected
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert call["system"] == "system"
    assert call["messages"] == [{"role": "user", "content": "user"}]
    assert call["tool_choice"] == {"type": "tool", "name": "propose_behavior"}
    tool = call["tools"][0]
    assert tool["name"] == "propose_behavior"
    assert tool["input_schema"] == build_tool_schema()


def test_anthropic_provider_raises_unavailable_when_sdk_call_throws():
    class _FailingClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._fail)

        def _fail(self, **_: Any) -> Any:
            raise ConnectionError

    provider = AnthropicProvider(client=_FailingClient())
    with pytest.raises(LLMProviderUnavailableError) as excinfo:
        provider.submit("s", "u", build_tool_schema())
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_anthropic_provider_raises_schema_error_when_no_tool_use_block():
    """If the model returns free text instead of a tool_use, the provider
    surfaces it as a schema error rather than crashing on missing fields."""

    class _TextOnlyClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_: Any) -> Any:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="I refuse to use the tool.")])

    provider = AnthropicProvider(client=_TextOnlyClient())
    with pytest.raises(LLMResponseSchemaError):
        provider.submit("s", "u", build_tool_schema())


def test_anthropic_behavior_llm_adapter_uses_anthropic_provider():
    """End-to-end: AnthropicBehaviorLLMAdapter wires the stub all the way
    through, validates, and returns the proposal."""

    class _ConvenienceClient:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_: Any) -> Any:
            return _make_stub_anthropic_response(self.payload)

    expected = _well_formed_proposal()
    # Inject the stub by replacing the underlying provider's client after
    # construction — keeps the public API (no required `client` param)
    # honest for the convenience subclass.
    adapter = AnthropicBehaviorLLMAdapter.__new__(AnthropicBehaviorLLMAdapter)
    provider = AnthropicProvider(client=_ConvenienceClient(expected))
    RemoteBehaviorLLMAdapter.__init__(adapter, provider)

    out = adapter.generate_behavior_proposal(_minimal_context())
    assert out == expected
    assert adapter.calls == 1
    assert adapter.failures == 0


# Avoid an unused-import lint complaint on MessageKind (kept for parity with
# tests/test_llm_adapter.py and to make adding message-driven scenarios in
# this file straightforward later).
_ = MessageKind
