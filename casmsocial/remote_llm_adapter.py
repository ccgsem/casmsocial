"""Remote LLM-backed behavior adapter conforming to BehaviorLLMAdapter.

Provides a drop-in alternative to :class:`~casmsocial.llm_adapter.LocalBehaviorLLMAdapter`
that delegates deliberation to a real LLM provider. The adapter is split into
two layers so the prompt-and-validation logic stays provider-agnostic:

* :class:`LLMProvider` — minimal :class:`~typing.Protocol` describing the
  transport contract: take a system prompt, a user prompt, and a JSON-Schema
  tool input schema; return the tool input dict.
* :class:`RemoteBehaviorLLMAdapter` — builds the prompt, invokes the
  provider, and validates the response against the canonical proposal
  schema. Conforms to :class:`~casmsocial.llm_adapter.BehaviorLLMAdapter`.

Anthropic Claude is the reference provider implementation; OpenAI or other
providers can be added as additional ``LLMProvider`` subclasses.

The ``anthropic`` package is imported lazily, so this module remains
importable in environments without it. Constructing
:class:`AnthropicProvider` without the SDK installed raises
:class:`LLMProviderUnavailableError` with a helpful message.

Failures (transport, schema, vocabulary) raise typed
:class:`LLMProviderError` subclasses. The surrounding
``LLMBehaviorEngine.decide`` already catches exceptions from the adapter and
falls back to the deterministic baseline engine, so remote adapters do not
need to construct fallback proposals themselves.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from casmsocial.llm_adapter import (
    PROPOSAL_ACTIONS,
    PROPOSAL_PLAN_ADJUSTMENT_KINDS,
    LocalBehaviorLLMAdapter,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Base for any failure during remote-adapter deliberation.

    Caught by ``LLMBehaviorEngine.decide``, which falls back to deterministic
    behavior; remote adapters should not attempt to construct fallback
    proposals themselves.
    """


class LLMProviderUnavailableError(LLMProviderError):
    """Network error, timeout, authentication failure, or other transport-level problem.

    The original provider exception (if any) is attached as ``__cause__`` so
    debugging can still reach it without exposing provider-specific types
    in the public API.
    """


class LLMResponseSchemaError(LLMProviderError):
    """The provider returned a response that did not match the canonical schema."""

    def __init__(self, missing: set[str], extra: set[str]) -> None:
        super().__init__(f"LLM response schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        self.missing = missing
        self.extra = extra


class LLMResponseInvalidActionError(LLMProviderError):
    """The action returned by the provider is not in :data:`PROPOSAL_ACTIONS`."""

    def __init__(self, action: str) -> None:
        valid = ", ".join(PROPOSAL_ACTIONS)
        super().__init__(f"LLM returned action {action!r}, which is not in the bounded vocabulary. Valid: {valid}.")
        self.action = action


class LLMResponseInvalidPlanAdjustmentError(LLMProviderError):
    """The plan-adjustment kind is not in :data:`PROPOSAL_PLAN_ADJUSTMENT_KINDS`."""

    def __init__(self, kind: str) -> None:
        valid = ", ".join(PROPOSAL_PLAN_ADJUSTMENT_KINDS)
        super().__init__(
            f"LLM returned plan_adjustment.kind {kind!r}, which is not in the bounded vocabulary. Valid: {valid}."
        )
        self.kind = kind


# ---------------------------------------------------------------------------
# LLMProvider Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal transport contract for a remote behavior LLM.

    Implementations submit a structured-output query (a system prompt, a
    user prompt, and a JSON-Schema-shaped tool input schema) and return the
    tool's input dict. ``RemoteBehaviorLLMAdapter`` does not depend on which
    provider implements this — Anthropic, OpenAI, a local mock for tests,
    etc.
    """

    def submit(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit a structured-output query and return the tool input.

        Raises:
            LLMProviderError: on transport, schema, or vocabulary failure.
        """
        ...


# ---------------------------------------------------------------------------
# Tool schema and prompt construction
# ---------------------------------------------------------------------------


_TOOL_NAME = "propose_behavior"


def build_tool_schema() -> dict[str, Any]:
    """Construct the JSON Schema for the propose_behavior tool input.

    Mirrors :data:`~casmsocial.llm_adapter.LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS`,
    with ``enum`` constraints on the bounded vocabularies so the provider
    rejects out-of-vocabulary responses server-side rather than relying on
    adapter-side validation alone.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS),
        "properties": {
            "action": {
                "type": "string",
                "enum": list(PROPOSAL_ACTIONS),
                "description": "Bounded action vocabulary; pick exactly one.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation citing the dominant signal channel.",
            },
            "emotion_updates": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Emotion adjustments as {name: number}.",
            },
            "need_updates": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Need adjustments as {name: number}.",
            },
            "attitude_updates": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Attitude adjustments as {name: number}.",
            },
            "message_intent": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "required": [
                            "sender_uid",
                            "receiver_uid",
                            "receiver_place_id",
                            "mode",
                            "payload",
                        ],
                        "properties": {
                            "sender_uid": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "receiver_uid": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "receiver_place_id": {"type": "integer"},
                            "mode": {"type": "string"},
                            "payload": {"type": "object"},
                        },
                    },
                ],
                "description": "Optional outgoing message; null when no message is sent.",
            },
            "plan_adjustment": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "required": ["kind", "detail"],
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": list(PROPOSAL_PLAN_ADJUSTMENT_KINDS),
                            },
                            "detail": {"type": "object"},
                        },
                    },
                ],
                "description": "Optional bounded plan adjustment; null when the schedule is unchanged.",
            },
            "memory_trace": {
                "type": "object",
                "additionalProperties": True,
                "description": (
                    "Per-channel signal magnitudes to write to short-term memory "
                    "(safety_signal, social_signal, obligation_signal, schedule_signal, "
                    "reply_signal, defer_signal, skip_signal, cancel_social_signal, "
                    "defer_minutes, source_count, responsiveness)."
                ),
            },
        },
    }


SYSTEM_PROMPT = """\
You are an agent in a discrete-event social simulation. At each tick, you
receive a compact JSON context describing your situation (observation,
beliefs, goals, emotions, needs, attitudes), recent typed messages from
other agents, recent appraisal memory traces, and your plan-state markers.

Your task is to call the propose_behavior tool with a structured proposal
deciding what the agent does next. Respond ONLY by invoking the tool; do
not produce free-form text.

Bounded action vocabulary:
  - follow_schedule: continue the planned activity sequence.
  - stay_home: remain at the home place; appropriate when safety-relevant
    signals dominate.
  - seek_social_contact: prioritize social interaction; appropriate when
    invitation or social signals dominate.
  - send_message: emit a reply or coordination message; appropriate when
    obligation or coordination signals dominate.

If a plan adjustment is appropriate, set plan_adjustment to one of:
  - preserve_home_activity: keep the agent at home (paired with stay_home).
  - defer_next_activity: push the next activity later in time.
  - skip_flexible_activity: drop the next flexible activity entirely.
  - cancel_social_activity: drop the next social activity entirely.
Otherwise set plan_adjustment to null.

The reason field should cite the dominant signal channel (e.g. "A request
for help creates the strongest obligation to reply.") so the simulation's
log remains interpretable.

emotion_updates, need_updates, attitude_updates, and memory_trace are
key-value maps of numeric adjustments. Use empty objects when nothing
needs updating in a category. memory_trace records the per-channel signal
magnitudes that drove the decision so subsequent ticks can decay and
reuse them.
"""


def build_user_prompt(context: dict[str, Any]) -> str:
    """Serialize the agent context as JSON for the user message.

    ``default=str`` lets unusual types (e.g. tuples-as-keys, dataclasses)
    serialize gracefully rather than raising.
    """
    return json.dumps(context, default=str, indent=2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a tool-call response against the canonical proposal schema.

    Raises one of:
      * :class:`LLMResponseSchemaError` if top-level keys diverge.
      * :class:`LLMResponseInvalidActionError` if ``action`` is out of vocabulary.
      * :class:`LLMResponseInvalidPlanAdjustmentError` if ``plan_adjustment.kind``
        is out of vocabulary.

    Returns the validated dict unchanged.
    """
    expected = set(LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS)
    actual = set(raw.keys())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise LLMResponseSchemaError(missing, extra)

    if raw["action"] not in PROPOSAL_ACTIONS:
        raise LLMResponseInvalidActionError(str(raw["action"]))

    plan_adjustment = raw.get("plan_adjustment")
    if plan_adjustment is not None:
        kind = plan_adjustment.get("kind") if isinstance(plan_adjustment, dict) else None
        if kind not in PROPOSAL_PLAN_ADJUSTMENT_KINDS:
            raise LLMResponseInvalidPlanAdjustmentError(str(kind))

    return raw


# ---------------------------------------------------------------------------
# RemoteBehaviorLLMAdapter
# ---------------------------------------------------------------------------


class RemoteBehaviorLLMAdapter:
    """Provider-agnostic remote adapter conforming to BehaviorLLMAdapter.

    Builds a system+user prompt + tool schema from the agent context,
    delegates the structured-output call to an :class:`LLMProvider`,
    validates the response against the canonical proposal schema, and
    returns it. Failures raise typed :class:`LLMProviderError`
    exceptions; the surrounding behavior engine catches them and falls
    back to deterministic behavior.

    ``calls`` and ``failures`` counters provide minimal observability
    for cost and reliability accounting.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        # Built once; the schema is provider-independent.
        self._tool_schema = build_tool_schema()
        self.calls: int = 0
        self.failures: int = 0

    def generate_behavior_proposal(self, context: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        try:
            user_prompt = build_user_prompt(context)
            raw = self.provider.submit(SYSTEM_PROMPT, user_prompt, self._tool_schema)
        except LLMProviderError:
            self.failures += 1
            raise
        except Exception as exc:
            self.failures += 1
            err = LLMProviderUnavailableError(f"Remote LLM provider call failed: {exc!r}")
            raise err from exc

        try:
            return validate_proposal(raw)
        except LLMProviderError:
            self.failures += 1
            raise


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider:
    """:class:`LLMProvider` that calls Anthropic Claude via the official SDK.

    The ``anthropic`` package is imported lazily; this class can be defined
    in environments without it, but instantiation requires it (or a
    pre-built client passed via the ``client`` parameter, which lets tests
    inject a mock without installing the SDK).

    The model identifier defaults to a current Haiku release. Override
    via the ``model`` parameter when the project pins to a different
    version. The API key is read from ``ANTHROPIC_API_KEY`` by the SDK
    if not provided explicitly.
    """

    def __init__(
        self,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from anthropic import Anthropic  # type: ignore[import-not-found]
            except ImportError as exc:
                err = LLMProviderUnavailableError(
                    "AnthropicProvider requires the `anthropic` package; "
                    "install it with `uv sync --extra remote` or `pip install anthropic`."
                )
                raise err from exc
            client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def submit(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": _TOOL_NAME,
                        "description": "Propose a bounded behavior for the current simulation tick.",
                        "input_schema": tool_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
        except Exception as exc:
            err = LLMProviderUnavailableError(f"Anthropic API call failed: {exc!r}")
            raise err from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
                tool_input = getattr(block, "input", None)
                if not isinstance(tool_input, dict):
                    raise LLMResponseSchemaError(missing=set(), extra=set())
                return tool_input

        # No tool_use block found — the model produced free text instead of
        # invoking the tool. tool_choice forces the call so this is rare,
        # but we surface it as a schema error rather than crashing.
        raise LLMResponseSchemaError(
            missing=set(LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS),
            extra=set(),
        )


class AnthropicBehaviorLLMAdapter(RemoteBehaviorLLMAdapter):
    """Convenience subclass that constructs an :class:`AnthropicProvider` automatically.

    Allows callers to instantiate a working remote adapter in one line::

        adapter = AnthropicBehaviorLLMAdapter(model="claude-haiku-4-5-20251001")
    """

    def __init__(
        self,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(AnthropicProvider(model=model, api_key=api_key, max_tokens=max_tokens))


__all__ = [
    "SYSTEM_PROMPT",
    "AnthropicBehaviorLLMAdapter",
    "AnthropicProvider",
    "LLMProvider",
    "LLMProviderError",
    "LLMProviderUnavailableError",
    "LLMResponseInvalidActionError",
    "LLMResponseInvalidPlanAdjustmentError",
    "LLMResponseSchemaError",
    "RemoteBehaviorLLMAdapter",
    "build_tool_schema",
    "build_user_prompt",
    "validate_proposal",
]
