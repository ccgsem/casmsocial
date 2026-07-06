"""Local no-network adapter for bounded behavior proposals.

This module also defines the canonical contract that any behavior-LLM adapter
(local or remote, deterministic or live-LLM-backed) must satisfy. The contract
has three layers:

* :data:`PROPOSAL_SCHEMA_KEYS` — the canonical eight top-level keys every
  proposal dict must carry (also exposed as a :class:`ClassVar` on
  :class:`LocalBehaviorLLMAdapter`).
* :data:`PROPOSAL_ACTIONS` — the four primitive actions a proposal may
  request. ``Person._parse_llm_response`` enforces this set.
* :data:`PROPOSAL_PLAN_ADJUSTMENT_KINDS` — the four bounded plan-adjustment
  kinds. ``Person._apply_plan_adjustment`` dispatches on these.

The :class:`BehaviorLLMAdapter` :class:`~typing.Protocol` formalizes the
single-method interface ``generate_behavior_proposal(context) -> proposal``
that the surrounding behavior engine consumes. Provider-specific
implementations (e.g. an Anthropic-backed adapter) can be substituted for
:class:`LocalBehaviorLLMAdapter` without changes elsewhere as long as they
return proposals matching this contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from casmsocial.activities import ActivitySemanticsOverrides, activity_semantics_for
from casmsocial.communication.types import MessageKind, build_message_payload

# ---------------------------------------------------------------------------
# Bounded vocabularies — actions and plan-adjustment kinds.
#
# The local adapter, the remote adapters, and ``Person._parse_llm_response``
# all need these sets in agreement. Lifting them to module-level tuples gives
# every consumer a single source of truth and makes them easy to import into
# JSON schemas (Anthropic tool schemas, OpenAI function-calling schemas, etc.)
# when constructing the structured-output contract for a real LLM provider.
# ---------------------------------------------------------------------------

#: The four primitive actions a behavior proposal may request.
PROPOSAL_ACTIONS: tuple[str, ...] = (
    "follow_schedule",
    "stay_home",
    "seek_social_contact",
    "send_message",
)

#: The four bounded plan-adjustment kinds. ``None`` is also valid (no
#: adjustment); these are the values for the ``plan_adjustment.kind`` field
#: when an adjustment is requested.
PROPOSAL_PLAN_ADJUSTMENT_KINDS: tuple[str, ...] = (
    "preserve_home_activity",
    "defer_next_activity",
    "skip_flexible_activity",
    "cancel_social_activity",
)

# ---------------------------------------------------------------------------
# Canonical reason strings.
#
# These are user-visible (the behavior engine writes them to
# ``cognition.last_llm_summary`` and tests assert on prefixes), so changes
# here intentionally require a deliberate edit. Lifting them to constants
# means the strings are reviewed in one place rather than scattered through
# handler bodies.
# ---------------------------------------------------------------------------
REASON_WARNING = "A warning increases safety concerns and favors staying home."
REASON_URGENT_ANNOUNCEMENT = "A high-urgency announcement raises caution and favors staying home."
REASON_SAFETY_RECOMMENDATION = "A trusted urgent recommendation favors staying home."
REASON_DEFER_RECOMMENDATION = "A travel recommendation favors deferring the next activity."
REASON_SKIP_RECOMMENDATION = "A recommendation favors skipping the next flexible activity."
REASON_CANCEL_SOCIAL_RECOMMENDATION = "A recommendation favors canceling the next social activity."
REASON_INVITATION = "An invitation raises social motivation and favors contact seeking."
REASON_REQUEST_HELP = "A request for help creates the strongest obligation to reply."
REASON_COORDINATION = "A coordination request benefits from a prompt acknowledgment."
REASON_REMINDER = "A reminder reinforces the current schedule."
REASON_GENERIC = "Respond to a recent incoming message while preserving the current schedule."
REASON_NO_OVERRIDE = "Recent messages are noted without overriding the current schedule."

REASON_TRANSIT = "Stay on the current travel leg until the next scheduled activity begins."
REASON_LATE_HOUR = "Late hour increases rest needs and favors staying home."
REASON_NO_DISRUPTION = "No salient disruption detected; keep the default schedule."

REASON_SOCIAL_CANCELLATION = "A social cancellation message favors canceling the next social activity."

# Memory-driven dominant reasons (asserted by tests via ``startswith``).
REASON_MEMORY_SAFETY = "Recent caution in memory continues to favor staying home."
REASON_MEMORY_OBLIGATION = "Recent help-related memory continues to favor replying."
REASON_MEMORY_SOCIAL = "Recent social invitations in memory continue to favor contact seeking."
REASON_MEMORY_DEFER = "Recent coordination or travel guidance in memory continues to favor deferring the next activity."
REASON_MEMORY_SKIP = "Recent cancellation guidance in memory continues to favor skipping the next flexible activity."
REASON_MEMORY_CANCEL_SOCIAL = (
    "Recent social cancellation guidance in memory continues to favor canceling the next social activity."
)
REASON_MEMORY_SCHEDULE = "Recent reminders in memory continue to reinforce the current schedule."
REASON_MEMORY_FALLBACK = "Recent message memory influences the current decision."


# Domain exceptions for invalid adapter configuration. Following the project
# convention (see casmsocial.casmpop for sibling patterns), the exception
# carries the offending value and the message lives on the class — keeps the
# call sites short and satisfies ruff TRY003.


class InvalidSignalCapError(ValueError):
    def __init__(self, value: float) -> None:
        super().__init__(f"signal_cap must be positive, got {value!r}")


class InvalidMemoryDecayError(ValueError):
    def __init__(self, value: float) -> None:
        super().__init__(f"memory_decay must lie in [0.0, 1.0], got {value!r}")


class InvalidAppraisalWindowError(ValueError):
    def __init__(self, value: int) -> None:
        super().__init__(f"appraisal_window must be positive, got {value!r}")


class UnsupportedPlanAdjustmentKindError(ValueError):
    """Raised when a plan-adjustment kind outside the bounded vocabulary is requested.

    The adapter and the simulation's plan-application layer must agree on the
    bounded vocabulary in :data:`PROPOSAL_PLAN_ADJUSTMENT_KINDS`. This
    exception flags the disagreement at the boundary.
    """

    def __init__(self, kind: str) -> None:
        valid = ", ".join(PROPOSAL_PLAN_ADJUSTMENT_KINDS)
        super().__init__(f"Unsupported policy adjustment kind: {kind!r}. Valid: {valid}.")


class UnsupportedAdapterKindError(ValueError):
    """Raised when ``LLMBehaviorEngine`` is configured with an unknown ``adapter`` kind.

    The behavior engine's ``_config["adapter"]`` selects between
    :class:`LocalBehaviorLLMAdapter`, the Anthropic-backed remote adapter,
    and any future named adapter. This exception surfaces typo'd config
    values at construction time rather than as silently-wrong simulation
    behavior.
    """

    def __init__(self, kind: str, valid: tuple[str, ...] = ("local", "anthropic")) -> None:
        valid_str = ", ".join(valid)
        super().__init__(f"Unsupported behavior-LLM adapter kind: {kind!r}. Valid: {valid_str}.")
        self.kind = kind


@dataclass(frozen=True, slots=True)
class _MessageContext:
    """Decoded view of a single recent message, computed once per appraisal step.

    Carries the fields every kind-handler needs (sender identity, reply
    routing, kind/topic/urgency/trust, derived weight) so the dispatch loop
    in :meth:`LocalBehaviorLLMAdapter._appraise_recent_messages` can pass a
    single value to whichever handler matched.
    """

    sender_uid: tuple[int, int, int]
    reply_place_id: int | None
    kind: MessageKind | None
    topic: str
    urgency: float
    trust_weight: float
    weight: float


@dataclass(slots=True)
class _ReplyTarget:
    """Identity needed to send a reply to the originator of a message."""

    sender_uid: tuple[int, int, int]
    place_id: int
    text: str


@dataclass(slots=True)
class _ChannelTrace:
    """Per-channel reason and (optional) reply target.

    Each policy-relevant signal channel keeps its own trace so the decision
    layer can cite the rationale of the *winning* channel rather than
    whichever branch updated the appraisal last.
    """

    reason: str = ""
    reply: _ReplyTarget | None = None


@dataclass(slots=True)
class MessageAppraisal:
    """Aggregate signal strengths derived from a recent message window.

    Reasons and reply targets are tracked per signal channel; the dominant
    reason / reply for the proposal is resolved by ``_decide_behavior_policy``
    based on which channel ultimately drives the action.
    """

    safety_signal: float = 0.0
    social_signal: float = 0.0
    obligation_signal: float = 0.0
    schedule_signal: float = 0.0
    reply_signal: float = 0.0
    defer_signal: float = 0.0
    skip_signal: float = 0.0
    cancel_social_signal: float = 0.0
    defer_minutes: int = 0
    # Free-form reason used as a fallback when no channel fired (e.g. messages
    # were noted but not classified) and as the dominant reason for the
    # memory-driven path, which writes it directly.
    dominant_reason: str = ""
    safety_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    obligation_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    social_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    coordination_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    defer_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    skip_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    cancel_social_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    schedule_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    generic_trace: _ChannelTrace = field(default_factory=_ChannelTrace)
    attitude_updates: dict[str, float] = field(default_factory=dict)
    source_count: int = 0
    from_memory: bool = False


@dataclass(slots=True)
class BehaviorPolicyDecision:
    """Policy-layer decision derived from an appraisal."""

    action: str
    reason: str
    emotion_updates: dict[str, float] = field(default_factory=dict)
    need_updates: dict[str, float] = field(default_factory=dict)
    attitude_updates: dict[str, float] = field(default_factory=dict)
    message_intent: dict[str, Any] | None = None
    plan_adjustment: dict[str, Any] | None = None


@runtime_checkable
class BehaviorLLMAdapter(Protocol):
    """Single-method contract every behavior-LLM adapter must satisfy.

    The behavior engine calls :meth:`generate_behavior_proposal` once per
    triggered deliberation, passing a compact agent context (observation,
    beliefs, recent typed messages, recent memory, plan-state markers, …) and
    receiving back a structured proposal whose top-level keys are exactly
    :data:`PROPOSAL_SCHEMA_KEYS`.

    Implementations may produce proposals deterministically
    (:class:`LocalBehaviorLLMAdapter`) or by calling a real LLM provider
    (e.g. an Anthropic- or OpenAI-backed adapter). The engine treats them
    interchangeably; surrounding code does not depend on the implementation
    strategy. ``@runtime_checkable`` lets ``isinstance(obj, BehaviorLLMAdapter)``
    confirm structural conformance, useful for test harnesses and dependency
    injection.

    Implementations are expected to either return a valid proposal or raise.
    The behavior engine catches any exception and falls back to deterministic
    behavior (see ``LLMBehaviorEngine.decide`` in ``person.py``), so adapters
    do not need to construct fallback proposals themselves.
    """

    def generate_behavior_proposal(self, context: dict[str, Any]) -> dict[str, Any]:
        """Produce a structured proposal from compact agent context."""
        ...


class LocalBehaviorLLMAdapter:
    """Deterministic stand-in for an LLM-backed behavior policy.

    Conforms structurally to :class:`BehaviorLLMAdapter`. This adapter
    deliberately uses simple local heuristics so the surrounding behavior-
    engine contract can be developed and tested without any network
    dependency or provider integration.
    """

    # ----- Topic taxonomy used by the appraisal dispatch rules -----
    # Lifted from the original if/elif chain so the matching predicates and
    # the handler bodies share a single source of truth, and so adding a new
    # topic only requires editing one set.
    _URGENT_ANNOUNCEMENT_TOPICS: ClassVar[frozenset[str]] = frozenset({"closure", "hazard", "emergency"})
    _SAFETY_RECOMMENDATION_TOPICS: ClassVar[frozenset[str]] = frozenset({"shelter", "avoid_travel", "stay_home"})
    _DEFER_RECOMMENDATION_TOPICS: ClassVar[frozenset[str]] = frozenset({"avoid_travel", "delay", "traffic"})
    _SKIP_RECOMMENDATION_TOPICS: ClassVar[frozenset[str]] = frozenset(
        {"skip", "cancel", "closed", "cancel_social", "social_cancel"}
    )
    _CANCEL_SOCIAL_TOPICS: ClassVar[frozenset[str]] = frozenset({"cancel_social", "social_cancel"})
    _COORDINATION_DEFER_TOPICS: ClassVar[frozenset[str]] = frozenset({"meetup", "pickup", "arrival"})
    _COORDINATION_SKIP_TOPICS: ClassVar[frozenset[str]] = frozenset({"cancel", "skip"})

    # ----- Policy thresholds -----
    # The decision cascade in ``_decide_behavior_policy`` compares aggregated
    # signals against these constants. Calibration sweeps and reviewers can
    # find every dial here in one place rather than chasing magic numbers
    # through the threshold cascade.
    SAFETY_ACTION_THRESHOLD: ClassVar[float] = 0.9
    OBLIGATION_ACTION_THRESHOLD: ClassVar[float] = 0.8
    SOCIAL_ACTION_THRESHOLD: ClassVar[float] = 0.7
    CANCEL_SOCIAL_ACTION_THRESHOLD: ClassVar[float] = 0.7
    SKIP_ACTION_THRESHOLD: ClassVar[float] = 0.7
    DEFER_ACTION_THRESHOLD: ClassVar[float] = 0.6
    SCHEDULE_REINFORCEMENT_THRESHOLD: ClassVar[float] = 0.5
    REPLY_ACTION_THRESHOLD: ClassVar[float] = 0.55

    # Memory-driven dominant-reason cascade thresholds.
    MEMORY_SAFETY_DOMINANT_THRESHOLD: ClassVar[float] = 0.45
    MEMORY_SOCIAL_DOMINANT_THRESHOLD: ClassVar[float] = 0.45
    MEMORY_OBLIGATION_DOMINANT_THRESHOLD: ClassVar[float] = 0.5
    MEMORY_SCHEDULE_DOMINANT_THRESHOLD: ClassVar[float] = 0.35
    MEMORY_DEFER_DOMINANT_THRESHOLD: ClassVar[float] = 0.35
    MEMORY_SKIP_DOMINANT_THRESHOLD: ClassVar[float] = 0.35
    MEMORY_CANCEL_SOCIAL_DOMINANT_THRESHOLD: ClassVar[float] = 0.35

    # Late-hour and transit gates used by the no-message fallback branches.
    LATE_HOUR_THRESHOLD: ClassVar[float] = 21.0
    TRANSIT_FLAG_THRESHOLD: ClassVar[float] = 0.0

    def __init__(
        self,
        signal_cap: float = 1.5,
        memory_decay: float = 0.65,
        activity_semantics_overrides: ActivitySemanticsOverrides | None = None,
        appraisal_window: int = 5,
    ) -> None:
        # Fail-loud validation so configuration typos surface at construction
        # time rather than as silently wrong appraisals later. The exception
        # classes (subclassing ValueError for backward compatibility with any
        # ``except ValueError`` clauses upstream) carry the offending value.
        if signal_cap <= 0:
            raise InvalidSignalCapError(signal_cap)
        if not 0.0 <= memory_decay <= 1.0:
            raise InvalidMemoryDecayError(memory_decay)
        if appraisal_window <= 0:
            raise InvalidAppraisalWindowError(appraisal_window)

        self.signal_cap = signal_cap
        self.memory_decay = memory_decay
        self.appraisal_window = appraisal_window
        self.activity_semantics_overrides = (
            {} if activity_semantics_overrides is None else dict(activity_semantics_overrides)
        )
        self._appraisal_rules: tuple[
            tuple[
                Callable[[_MessageContext], bool],
                Callable[[MessageAppraisal, _MessageContext], None],
            ],
            ...,
        ] = self._build_appraisal_rules()

    def _build_appraisal_rules(
        self,
    ) -> tuple[
        tuple[
            Callable[[_MessageContext], bool],
            Callable[[MessageAppraisal, _MessageContext], None],
        ],
        ...,
    ]:
        """First-match-wins dispatch table for message appraisal.

        Order matters: rules are tried in sequence and the first matching
        predicate's handler runs. Messages that match no rule are routed to
        :meth:`_handle_generic`.
        """
        return (
            (
                lambda m: m.kind == MessageKind.WARNING,
                self._handle_warning,
            ),
            (
                lambda m: m.kind == MessageKind.ANNOUNCEMENT
                and (m.urgency >= 0.7 or m.topic in self._URGENT_ANNOUNCEMENT_TOPICS),
                self._handle_urgent_announcement,
            ),
            (
                lambda m: m.kind == MessageKind.RECOMMENDATION
                and m.trust_weight >= 0.75
                and m.topic in self._SAFETY_RECOMMENDATION_TOPICS,
                self._handle_safety_recommendation,
            ),
            (
                lambda m: m.kind == MessageKind.RECOMMENDATION
                and m.topic in self._DEFER_RECOMMENDATION_TOPICS
                and m.urgency >= 0.4,
                self._handle_defer_recommendation,
            ),
            (
                lambda m: m.kind == MessageKind.RECOMMENDATION and m.topic in self._SKIP_RECOMMENDATION_TOPICS,
                self._handle_skip_recommendation,
            ),
            (
                lambda m: m.kind == MessageKind.INVITATION,
                self._handle_invitation,
            ),
            (
                lambda m: m.kind == MessageKind.REQUEST_HELP,
                self._handle_request_help,
            ),
            (
                lambda m: m.kind == MessageKind.COORDINATION,
                self._handle_coordination,
            ),
            (
                lambda m: m.kind == MessageKind.REMINDER,
                self._handle_reminder,
            ),
        )

    def _build_message_context(self, raw_message: dict[str, Any]) -> _MessageContext:
        """Decode a raw recent_messages entry into the immutable handler input."""
        sender_uid = tuple(raw_message["sender_uid"])
        payload = dict(raw_message.get("payload", {}))
        metadata = dict(payload.get("metadata", {}))
        kind = self._coerce_kind(payload.get("kind"))
        topic = str(payload.get("topic", "")).strip().lower()
        trust_weight = float(payload.get("trust_weight", 0.5))
        urgency = float(payload.get("urgency", 0.0))
        weight = max(0.1, urgency + trust_weight / 2.0)
        sender_place_id = metadata.get("sender_place_id")
        reply_place_id = int(sender_place_id) if sender_place_id is not None else None
        return _MessageContext(
            sender_uid=sender_uid,
            reply_place_id=reply_place_id,
            kind=kind,
            topic=topic,
            urgency=urgency,
            trust_weight=trust_weight,
            weight=weight,
        )

    def _dispatch_handler(self, msg: _MessageContext) -> Callable[[MessageAppraisal, _MessageContext], None]:
        """Return the handler for a message, falling back to the generic handler."""
        for matches, handler in self._appraisal_rules:
            if matches(msg):
                return handler
        return self._handle_generic

    # Canonical set of top-level keys every proposal dict must contain.
    # Consumers (e.g. Person._parse_llm_response) read by name, but pinning the
    # contract here means any future caller — a real LLM provider, recorder,
    # or logger — can rely on these keys being present in every branch.
    PROPOSAL_SCHEMA_KEYS: tuple[str, ...] = (
        "action",
        "reason",
        "emotion_updates",
        "need_updates",
        "attitude_updates",
        "message_intent",
        "plan_adjustment",
        "memory_trace",
    )

    @staticmethod
    def _build_proposal_dict(
        *,
        action: str,
        reason: str,
        emotion_updates: dict[str, float] | None = None,
        need_updates: dict[str, float] | None = None,
        attitude_updates: dict[str, float] | None = None,
        message_intent: dict[str, Any] | None = None,
        plan_adjustment: dict[str, Any] | None = None,
        memory_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct a behavior proposal with the full canonical schema.

        Single source of truth for the proposal contract: every return path
        in :meth:`generate_behavior_proposal` (message-driven, memory-driven,
        transit fallback, late-hour fallback, default fallback) routes through
        this helper so consumers always see the same keys.
        ``message_intent`` and ``plan_adjustment`` are nullable; the dict-typed
        fields default to empty dicts rather than ``None`` so callers can
        iterate them unconditionally.
        """
        return {
            "action": action,
            "reason": reason,
            "emotion_updates": {} if emotion_updates is None else emotion_updates,
            "need_updates": {} if need_updates is None else need_updates,
            "attitude_updates": {} if attitude_updates is None else attitude_updates,
            "message_intent": message_intent,
            "plan_adjustment": plan_adjustment,
            "memory_trace": {} if memory_trace is None else memory_trace,
        }

    def generate_behavior_proposal(self, context: dict[str, Any]) -> dict[str, Any]:
        """Produce a structured proposal from compact agent context."""
        recent_messages = list(context.get("recent_messages", []))
        observation = dict(context.get("observation", {}))
        agent_uid = tuple(context["agent_uid"])
        attitudes = dict(context.get("attitudes", {}))

        if recent_messages:
            appraisal = self._appraise_recent_messages(
                recent_messages=recent_messages,
                attitudes=attitudes,
            )
            return self._proposal_from_appraisal(appraisal, context, attitudes, agent_uid)

        memory_appraisal = self._appraise_recent_memory(context.get("recent_memory", []), attitudes)
        if memory_appraisal is not None:
            return self._proposal_from_appraisal(memory_appraisal, context, attitudes, agent_uid)

        if float(observation.get("is_transit", 0.0)) > self.TRANSIT_FLAG_THRESHOLD:
            return self._build_proposal_dict(
                action="follow_schedule",
                reason=REASON_TRANSIT,
                emotion_updates={"calm": 0.5},
                need_updates={"safety": 0.4},
            )

        if float(observation.get("hour_of_day", 0.0)) >= self.LATE_HOUR_THRESHOLD:
            return self._build_proposal_dict(
                action="stay_home",
                reason=REASON_LATE_HOUR,
                emotion_updates={"calm": 0.7},
                need_updates={"rest": 0.8},
            )

        return self._build_proposal_dict(
            action="follow_schedule",
            reason=REASON_NO_DISRUPTION,
            emotion_updates={"calm": 0.7},
        )

    def _appraise_recent_messages(
        self,
        *,
        recent_messages: list[dict[str, Any]],
        attitudes: dict[str, Any],
    ) -> MessageAppraisal:
        """Aggregate recent messages into a compact appraisal state.

        Each message is decoded into a :class:`_MessageContext`, dispatched to
        the rule-matched handler, and folded into the appraisal. Per-message
        bookkeeping (responsiveness ratchet, source counter) lives here so
        handlers stay focused on signal-channel updates.
        """
        responsiveness = float(attitudes.get("responsiveness", 0.5))
        appraisal = MessageAppraisal()
        for raw_message in recent_messages[-self.appraisal_window :]:
            msg = self._build_message_context(raw_message)
            appraisal.attitude_updates["responsiveness"] = max(
                appraisal.attitude_updates.get("responsiveness", responsiveness),
                min(1.0, responsiveness + 0.01 + msg.trust_weight / 20.0),
            )
            appraisal.source_count += 1
            handler = self._dispatch_handler(msg)
            handler(appraisal, msg)
        if not appraisal.dominant_reason:
            appraisal.dominant_reason = REASON_NO_OVERRIDE
        return appraisal

    @staticmethod
    def _record_reply_target(
        trace: _ChannelTrace,
        sender_uid: tuple[int, int, int],
        place_id: int | None,
        text: str,
    ) -> None:
        """Attach a reply target to a channel trace, skipping when no destination is known."""
        if place_id is None:
            # No usable destination — skip rather than silently self-routing.
            return
        trace.reply = _ReplyTarget(sender_uid=sender_uid, place_id=place_id, text=text)

    # ------------------------- per-kind handlers --------------------------
    # Each handler updates only its channel(s); first-match-wins dispatch in
    # _build_appraisal_rules picks the right one. New MessageKind support =
    # one new rule + one new handler, no edits to a long if/elif chain.

    def _handle_warning(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.8 + msg.weight
        if signal > appraisal.safety_signal:
            appraisal.safety_trace.reason = REASON_WARNING
        appraisal.safety_signal = self._bounded_add(appraisal.safety_signal, signal)

    def _handle_urgent_announcement(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.6 + msg.weight
        if signal > appraisal.safety_signal:
            appraisal.safety_trace.reason = REASON_URGENT_ANNOUNCEMENT
        appraisal.safety_signal = self._bounded_add(appraisal.safety_signal, signal)

    def _handle_safety_recommendation(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.4 + msg.weight
        if signal > appraisal.safety_signal:
            appraisal.safety_trace.reason = REASON_SAFETY_RECOMMENDATION
        appraisal.safety_signal = self._bounded_add(appraisal.safety_signal, signal)

    def _handle_defer_recommendation(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.35 + msg.weight
        if signal > appraisal.defer_signal:
            appraisal.defer_trace.reason = REASON_DEFER_RECOMMENDATION
            appraisal.defer_minutes = max(appraisal.defer_minutes, 30)
        appraisal.defer_signal = self._bounded_add(appraisal.defer_signal, signal)

    def _handle_skip_recommendation(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.35 + msg.weight
        if msg.topic in self._CANCEL_SOCIAL_TOPICS:
            if signal > appraisal.cancel_social_signal:
                appraisal.cancel_social_trace.reason = REASON_CANCEL_SOCIAL_RECOMMENDATION
            appraisal.cancel_social_signal = self._bounded_add(appraisal.cancel_social_signal, signal)
        else:
            if signal > appraisal.skip_signal:
                appraisal.skip_trace.reason = REASON_SKIP_RECOMMENDATION
            appraisal.skip_signal = self._bounded_add(appraisal.skip_signal, signal)

    def _handle_invitation(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.5 + msg.weight
        if signal > appraisal.social_signal:
            appraisal.social_trace.reason = REASON_INVITATION
            self._record_reply_target(appraisal.social_trace, msg.sender_uid, msg.reply_place_id, "Invitation received")
        appraisal.social_signal = self._bounded_add(appraisal.social_signal, signal)
        appraisal.reply_signal = self._bounded_add(appraisal.reply_signal, 0.4 + msg.trust_weight / 2.0)

    def _handle_request_help(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.7 + msg.weight
        if signal > appraisal.obligation_signal:
            appraisal.obligation_trace.reason = REASON_REQUEST_HELP
            self._record_reply_target(
                appraisal.obligation_trace,
                msg.sender_uid,
                msg.reply_place_id,
                "Help request acknowledged",
            )
        appraisal.obligation_signal = self._bounded_add(appraisal.obligation_signal, signal)
        appraisal.reply_signal = self._bounded_add(appraisal.reply_signal, 0.7 + msg.urgency / 3.0)

    def _handle_coordination(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.4 + msg.weight
        if signal > appraisal.reply_signal:
            appraisal.coordination_trace.reason = REASON_COORDINATION
            self._record_reply_target(
                appraisal.coordination_trace,
                msg.sender_uid,
                msg.reply_place_id,
                "Coordination confirmed",
            )
        appraisal.reply_signal = self._bounded_add(appraisal.reply_signal, signal)
        if msg.topic not in self._CANCEL_SOCIAL_TOPICS and (
            msg.urgency >= 0.6 or msg.topic in self._COORDINATION_DEFER_TOPICS
        ):
            appraisal.defer_signal = self._bounded_add(appraisal.defer_signal, 0.3 + msg.weight)
            appraisal.defer_minutes = max(appraisal.defer_minutes, 30)
        if msg.topic in self._COORDINATION_SKIP_TOPICS:
            appraisal.skip_signal = self._bounded_add(appraisal.skip_signal, 0.35 + msg.weight)
        if msg.topic in self._CANCEL_SOCIAL_TOPICS:
            appraisal.cancel_social_signal = self._bounded_add(appraisal.cancel_social_signal, 0.35 + msg.weight)

    def _handle_reminder(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        signal = 0.5 + msg.weight / 2.0
        if signal > appraisal.schedule_signal:
            appraisal.schedule_trace.reason = REASON_REMINDER
        appraisal.schedule_signal = self._bounded_add(appraisal.schedule_signal, signal)

    def _handle_generic(self, appraisal: MessageAppraisal, msg: _MessageContext) -> None:
        """Fallback for kinds with no specialized handler (CHECK_IN, STATUS_UPDATE, ACKNOWLEDGMENT, unknown)."""
        signal = 0.45 + msg.weight / 2.0
        if signal > appraisal.reply_signal:
            appraisal.generic_trace.reason = REASON_GENERIC
            self._record_reply_target(appraisal.generic_trace, msg.sender_uid, msg.reply_place_id, "Acknowledged")
        appraisal.reply_signal = self._bounded_add(appraisal.reply_signal, signal)

    def _appraise_recent_memory(
        self,
        recent_memory: list[dict[str, Any]],
        attitudes: dict[str, Any],
    ) -> MessageAppraisal | None:
        """Derive a decayed appraisal from recent stored message traces."""
        if not recent_memory:
            return None

        # Bug fix: previously this sliced ``recent_memory[-5:]`` *before*
        # filtering, so a burst of non-message events at the tail would crowd
        # out every relevant message_appraisal entry. Filter first, then take
        # the most recent five matching events.
        appraisal_events = [event for event in recent_memory if event.get("event_type") == "message_appraisal"]
        if not appraisal_events:
            return None

        responsiveness = float(attitudes.get("responsiveness", 0.5))
        appraisal = MessageAppraisal()
        weight = 1.0
        found = False
        for event in reversed(appraisal_events[-self.appraisal_window :]):
            data = dict(event.get("data", {}))
            found = True
            appraisal.safety_signal = self._bounded_add(
                appraisal.safety_signal,
                float(data.get("safety_signal", 0.0)) * weight,
            )
            appraisal.social_signal = self._bounded_add(
                appraisal.social_signal,
                float(data.get("social_signal", 0.0)) * weight,
            )
            appraisal.obligation_signal = self._bounded_add(
                appraisal.obligation_signal,
                float(data.get("obligation_signal", 0.0)) * weight,
            )
            appraisal.schedule_signal = self._bounded_add(
                appraisal.schedule_signal,
                float(data.get("schedule_signal", 0.0)) * weight,
            )
            appraisal.reply_signal = self._bounded_add(
                appraisal.reply_signal,
                float(data.get("reply_signal", 0.0)) * weight,
            )
            appraisal.defer_signal = self._bounded_add(
                appraisal.defer_signal,
                float(data.get("defer_signal", 0.0)) * weight,
            )
            appraisal.skip_signal = self._bounded_add(
                appraisal.skip_signal,
                float(data.get("skip_signal", 0.0)) * weight,
            )
            appraisal.cancel_social_signal = self._bounded_add(
                appraisal.cancel_social_signal,
                float(data.get("cancel_social_signal", 0.0)) * weight,
            )
            appraisal.defer_minutes = max(appraisal.defer_minutes, int(data.get("defer_minutes", 0)))
            appraisal.source_count += int(data.get("source_count", 1))
            appraisal.attitude_updates["responsiveness"] = max(
                appraisal.attitude_updates.get("responsiveness", responsiveness),
                float(data.get("responsiveness", responsiveness)),
            )
            if not appraisal.dominant_reason:
                appraisal.dominant_reason = str(event.get("summary", REASON_MEMORY_FALLBACK))
            weight *= self.memory_decay

        if not found:
            return None

        appraisal.from_memory = True

        if appraisal.safety_signal >= max(
            appraisal.social_signal,
            appraisal.obligation_signal,
            appraisal.schedule_signal,
            self.MEMORY_SAFETY_DOMINANT_THRESHOLD,
        ):
            appraisal.dominant_reason = REASON_MEMORY_SAFETY
        elif appraisal.obligation_signal >= max(
            appraisal.social_signal,
            appraisal.schedule_signal,
            self.MEMORY_OBLIGATION_DOMINANT_THRESHOLD,
        ):
            appraisal.dominant_reason = REASON_MEMORY_OBLIGATION
        elif appraisal.social_signal >= max(appraisal.schedule_signal, self.MEMORY_SOCIAL_DOMINANT_THRESHOLD):
            appraisal.dominant_reason = REASON_MEMORY_SOCIAL
        elif appraisal.defer_signal >= self.MEMORY_DEFER_DOMINANT_THRESHOLD:
            appraisal.dominant_reason = REASON_MEMORY_DEFER
        elif appraisal.skip_signal >= self.MEMORY_SKIP_DOMINANT_THRESHOLD:
            appraisal.dominant_reason = REASON_MEMORY_SKIP
        elif appraisal.cancel_social_signal >= self.MEMORY_CANCEL_SOCIAL_DOMINANT_THRESHOLD:
            appraisal.dominant_reason = REASON_MEMORY_CANCEL_SOCIAL
        elif appraisal.schedule_signal >= self.MEMORY_SCHEDULE_DOMINANT_THRESHOLD:
            appraisal.dominant_reason = REASON_MEMORY_SCHEDULE
        return appraisal

    def _proposal_from_appraisal(
        self,
        appraisal: MessageAppraisal,
        context: dict[str, Any],
        attitudes: dict[str, Any],
        agent_uid: tuple[int, int, int],
    ) -> dict[str, Any]:
        """Convert an aggregate appraisal into a bounded proposal."""
        decision = self._decide_behavior_policy(appraisal, context, attitudes, agent_uid)
        return self._build_proposal_dict(
            action=decision.action,
            reason=decision.reason,
            emotion_updates=decision.emotion_updates,
            need_updates=decision.need_updates,
            attitude_updates=decision.attitude_updates,
            message_intent=decision.message_intent,
            plan_adjustment=decision.plan_adjustment,
            memory_trace=self._memory_trace_from_appraisal(appraisal, decision.attitude_updates),
        )

    def _decide_behavior_policy(
        self,
        appraisal: MessageAppraisal,
        context: dict[str, Any],
        attitudes: dict[str, Any],
        agent_uid: tuple[int, int, int],
    ) -> BehaviorPolicyDecision:
        """Map an appraisal into an action and optional bounded plan adjustment."""
        next_flexible_activity_id = dict(context.get("plan_state", {})).get("next_flexible_activity_id")
        next_social_activity_id = dict(context.get("plan_state", {})).get("next_social_activity_id")
        next_semantics = (
            activity_semantics_for(int(next_flexible_activity_id), self.activity_semantics_overrides)
            if next_flexible_activity_id is not None
            else None
        )
        next_social_semantics = (
            activity_semantics_for(int(next_social_activity_id), self.activity_semantics_overrides)
            if next_social_activity_id is not None
            else None
        )
        # Forward every attitude update from the appraisal — not just
        # responsiveness — so future channels that write attitudes don't get
        # silently dropped here. Each key is ratcheted against its baseline
        # value (or 0.0 if absent) so updates only ever increase. The baseline
        # responsiveness is always carried so callers see at least the
        # unchanged value even when no message wrote it.
        attitude_updates: dict[str, float] = {
            "responsiveness": float(attitudes.get("responsiveness", 0.5)),
        }
        for key, value in appraisal.attitude_updates.items():
            baseline = attitude_updates.get(key, float(attitudes.get(key, 0.0)))
            attitude_updates[key] = max(baseline, float(value))

        # Helper: pick the channel's recorded reason, falling back to whichever
        # global reason the appraisal layer produced (used by the memory path
        # and for the "no channel fired" generic fallback).
        def reason_for(trace: _ChannelTrace) -> str:
            return trace.reason or appraisal.dominant_reason

        if appraisal.safety_signal >= max(
            appraisal.social_signal,
            appraisal.obligation_signal,
            appraisal.schedule_signal,
            self.SAFETY_ACTION_THRESHOLD,
        ):
            return BehaviorPolicyDecision(
                action="stay_home",
                reason=reason_for(appraisal.safety_trace),
                emotion_updates={"fear": min(1.0, appraisal.safety_signal / 2.0), "calm": 0.2},
                need_updates={"safety": min(1.0, appraisal.safety_signal / 1.5)},
                attitude_updates=attitude_updates,
                message_intent=None,
                plan_adjustment=self._plan_adjustment_from_policy(appraisal, "preserve_home_activity"),
            )

        if appraisal.obligation_signal >= max(
            appraisal.social_signal,
            appraisal.schedule_signal,
            self.OBLIGATION_ACTION_THRESHOLD,
        ):
            return BehaviorPolicyDecision(
                action="send_message",
                reason=reason_for(appraisal.obligation_trace),
                emotion_updates={"calm": 0.3},
                need_updates={"social": min(1.0, appraisal.obligation_signal / 2.0)},
                attitude_updates=attitude_updates,
                message_intent=self._reply_intent_from_appraisal(appraisal, agent_uid, prefer="obligation_trace"),
            )

        if appraisal.social_signal >= max(appraisal.schedule_signal, self.SOCIAL_ACTION_THRESHOLD):
            return BehaviorPolicyDecision(
                action="seek_social_contact",
                reason=reason_for(appraisal.social_trace),
                emotion_updates={"joy": min(1.0, appraisal.social_signal / 2.0), "calm": 0.6},
                need_updates={"social": min(1.0, appraisal.social_signal / 1.5)},
                attitude_updates=attitude_updates,
                message_intent=self._reply_intent_from_appraisal(appraisal, agent_uid, prefer="social_trace"),
            )

        if (
            next_social_semantics is not None
            and next_social_semantics.is_social
            and appraisal.cancel_social_signal >= max(appraisal.schedule_signal, self.CANCEL_SOCIAL_ACTION_THRESHOLD)
        ):
            return BehaviorPolicyDecision(
                action="send_message" if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD else "follow_schedule",
                reason=REASON_SOCIAL_CANCELLATION,
                emotion_updates={"calm": 0.5},
                need_updates={},
                attitude_updates=attitude_updates,
                message_intent=(
                    self._reply_intent_from_appraisal(appraisal, agent_uid, prefer="coordination_trace")
                    if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD
                    else None
                ),
                plan_adjustment=self._plan_adjustment_from_policy(appraisal, "cancel_social_activity"),
            )

        if (
            next_semantics is not None
            and next_semantics.is_flexible
            and appraisal.skip_signal >= max(appraisal.schedule_signal, self.SKIP_ACTION_THRESHOLD)
        ):
            return BehaviorPolicyDecision(
                action="send_message" if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD else "follow_schedule",
                reason=reason_for(appraisal.skip_trace),
                emotion_updates={"calm": 0.5},
                need_updates={},
                attitude_updates=attitude_updates,
                message_intent=(
                    self._reply_intent_from_appraisal(appraisal, agent_uid, prefer="coordination_trace")
                    if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD
                    else None
                ),
                plan_adjustment=self._plan_adjustment_from_policy(appraisal, "skip_flexible_activity"),
            )

        if (
            next_semantics is not None
            and next_semantics.is_flexible
            and appraisal.defer_signal >= max(appraisal.schedule_signal, self.DEFER_ACTION_THRESHOLD)
        ):
            return BehaviorPolicyDecision(
                action="send_message" if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD else "follow_schedule",
                reason=reason_for(appraisal.defer_trace),
                emotion_updates={"calm": 0.5},
                need_updates={},
                attitude_updates=attitude_updates,
                message_intent=(
                    self._reply_intent_from_appraisal(appraisal, agent_uid, prefer="coordination_trace")
                    if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD
                    else None
                ),
                plan_adjustment=self._plan_adjustment_from_policy(appraisal, "defer_next_activity"),
            )

        if appraisal.reply_signal >= self.REPLY_ACTION_THRESHOLD:
            # Pure-reply fallthrough: use whichever channel actually captured a
            # reply, with the priority order in _resolve_reply_target.
            target_trace_name = self._dominant_reply_trace_name(appraisal)
            reply_reason = getattr(appraisal, target_trace_name).reason if target_trace_name else ""
            return BehaviorPolicyDecision(
                action="send_message",
                reason=reply_reason or appraisal.dominant_reason,
                emotion_updates={"calm": 0.5},
                need_updates={},
                attitude_updates=attitude_updates,
                message_intent=self._reply_intent_from_appraisal(appraisal, agent_uid),
            )

        if appraisal.schedule_signal >= self.SCHEDULE_REINFORCEMENT_THRESHOLD:
            return BehaviorPolicyDecision(
                action="follow_schedule",
                reason=reason_for(appraisal.schedule_trace),
                emotion_updates={"calm": 0.7},
                need_updates={"rest": 0.1},
                attitude_updates=attitude_updates,
            )

        return BehaviorPolicyDecision(
            action="follow_schedule",
            reason=appraisal.dominant_reason,
            emotion_updates={"calm": 0.6},
            need_updates={},
            attitude_updates=attitude_updates,
        )

    @classmethod
    def _dominant_reply_trace_name(cls, appraisal: MessageAppraisal) -> str | None:
        """Return the trace attribute name whose reply should be used in the pure-reply branch."""
        for name in cls._REPLY_PRIORITY:
            if getattr(appraisal, name).reply is not None:
                return name
        return None

    def _plan_adjustment_from_policy(
        self,
        appraisal: MessageAppraisal,
        adjustment_kind: str,
    ) -> dict[str, Any]:
        """Map a policy choice to a constrained plan adjustment payload."""
        if adjustment_kind == "preserve_home_activity":
            return {
                "kind": "preserve_home_activity",
                "detail": {
                    "source": "safety_signal",
                },
            }

        if adjustment_kind == "defer_next_activity":
            return {
                "kind": "defer_next_activity",
                "detail": {
                    "delay_minutes": max(15, appraisal.defer_minutes or 30),
                    "source": "defer_signal",
                },
            }

        if adjustment_kind == "skip_flexible_activity":
            return {
                "kind": "skip_flexible_activity",
                "detail": {
                    "source": "skip_signal",
                },
            }

        if adjustment_kind == "cancel_social_activity":
            return {
                "kind": "cancel_social_activity",
                "detail": {
                    "source": "cancel_social_signal",
                },
            }

        raise UnsupportedPlanAdjustmentKindError(adjustment_kind)

    def _reply_intent(
        self,
        sender_uid: tuple[int, int, int],
        receiver_uid: tuple[int, int, int],
        receiver_place_id: int,
        text: str,
    ) -> dict[str, Any]:
        """Build a standard acknowledgment reply intent."""
        return {
            "sender_uid": sender_uid,
            "receiver_uid": receiver_uid,
            "receiver_place_id": receiver_place_id,
            "mode": "two_way",
            "payload": build_message_payload(
                MessageKind.ACKNOWLEDGMENT,
                text=text,
                metadata={"auto_reply": True},
            ),
        }

    # Fallback order used when the policy-preferred channel has no reply target
    # but some other channel still can address a sender. Obligation > social >
    # coordination > generic prefers richer/more-meaningful contexts first.
    _REPLY_PRIORITY: tuple[str, ...] = (
        "obligation_trace",
        "social_trace",
        "coordination_trace",
        "generic_trace",
    )

    @classmethod
    def _resolve_reply_target(
        cls,
        appraisal: MessageAppraisal,
        prefer: str | None = None,
    ) -> _ReplyTarget | None:
        """Pick the reply target for the policy-selected channel.

        ``prefer`` is a trace attribute name (e.g. ``"obligation_trace"``).
        If the preferred channel didn't capture a target, fall back to the
        priority order so a reply is still emitted when *some* channel can
        address the sender. Returns ``None`` if no channel has a destination.
        """
        if prefer is not None:
            trace = cast(_ChannelTrace | None, getattr(appraisal, prefer, None))
            if trace is not None and trace.reply is not None:
                return trace.reply
        for name in cls._REPLY_PRIORITY:
            trace = cast(_ChannelTrace, getattr(appraisal, name))
            if trace.reply is not None:
                return trace.reply
        return None

    def _reply_intent_from_appraisal(
        self,
        appraisal: MessageAppraisal,
        agent_uid: tuple[int, int, int],
        prefer: str | None = None,
    ) -> dict[str, Any] | None:
        """Create a reply intent when the aggregate appraisal calls for one."""
        target = self._resolve_reply_target(appraisal, prefer=prefer)
        if target is None:
            return None
        return self._reply_intent(
            sender_uid=agent_uid,
            receiver_uid=target.sender_uid,
            receiver_place_id=target.place_id,
            text=target.text,
        )

    def _memory_trace_from_appraisal(
        self,
        appraisal: MessageAppraisal,
        attitude_updates: dict[str, float],
    ) -> dict[str, Any]:
        """Serialize aggregate appraisal signals for short-term memory.

        Two distinct decay effects coexist:

        * Aggregation-time decay (``weight *= memory_decay`` in
          ``_appraise_recent_memory``) — weights older events *within the
          current memory window* less than newer ones.
        * Storage-time decay (``decay_factor`` here) — applies one hop of
          attenuation each cycle the agent re-appraises and re-stores from
          memory. Without it a stable memory event would saturate at
          ``signal_cap`` forever; with it, traces fade over successive ticks.

        Both are required for the design's "memory fades over time" property
        (see test_llm_behavior_engine_memory_decay_reduces_signal_over_successive_ticks).
        """
        decay_factor = self.memory_decay if appraisal.from_memory else 1.0
        return {
            "safety_signal": appraisal.safety_signal * decay_factor,
            "social_signal": appraisal.social_signal * decay_factor,
            "obligation_signal": appraisal.obligation_signal * decay_factor,
            "schedule_signal": appraisal.schedule_signal * decay_factor,
            "reply_signal": appraisal.reply_signal * decay_factor,
            "defer_signal": appraisal.defer_signal * decay_factor,
            "skip_signal": appraisal.skip_signal * decay_factor,
            "cancel_social_signal": appraisal.cancel_social_signal * decay_factor,
            "defer_minutes": appraisal.defer_minutes,
            "source_count": appraisal.source_count,
            "responsiveness": attitude_updates.get("responsiveness", 0.5),
        }

    def _bounded_add(self, current: float, increment: float) -> float:
        """Accumulate a signal up to the configured saturation cap."""
        return min(self.signal_cap, current + max(0.0, increment))

    def _coerce_kind(self, kind_value: Any) -> MessageKind | None:
        """Convert payload kind values into the shared enum."""
        if isinstance(kind_value, MessageKind):
            return kind_value
        if kind_value is None:
            return None
        try:
            return MessageKind(str(kind_value))
        except ValueError:
            return None
