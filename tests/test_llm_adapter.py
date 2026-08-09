"""Direct unit tests for ``LocalBehaviorLLMAdapter``.

These tests exercise the adapter at its public surface
(:meth:`LocalBehaviorLLMAdapter.generate_behavior_proposal`) with synthetic
contexts, complementing the higher-level scenarios in ``test_person.py`` that
drive the adapter through ``Person`` / ``LLMBehaviorEngine``. They cover:

* canonical schema keys are present in every return path,
* per-channel reply / reason routing under multi-message inputs,
* metadata-missing reply suppression (no self-routing),
* memory-window filter ordering,
* per-hop storage-time memory decay (the design contract pinned by
  ``test_llm_behavior_engine_memory_decay_reduces_signal_over_successive_ticks``).

The adapter has minimal dependencies — only ``casmsocial.activities`` and
``casmsocial.communication.types`` — so these tests run without MPI or any
heavyweight stack components.
"""

from __future__ import annotations

import pytest

from casmsocial.communication.types import MessageKind, build_message_payload
from casmsocial.llm_adapter import (
    PROPOSAL_ACTIONS,
    PROPOSAL_PLAN_ADJUSTMENT_KINDS,
    BehaviorLLMAdapter,
    LocalBehaviorLLMAdapter,
    UnsupportedPlanAdjustmentKindError,
)

# ----------------------------- helpers -------------------------------------


def _message(
    kind: MessageKind,
    *,
    sender_uid: tuple[int, int, int] = (2, 0, 0),
    sender_place_id: int | None = 100,
    text: str = "",
    topic: str = "",
    urgency: float = 0.0,
    trust_weight: float = 0.5,
) -> dict:
    metadata: dict = {}
    if sender_place_id is not None:
        metadata["sender_place_id"] = sender_place_id
    payload = build_message_payload(
        kind,
        topic=topic,
        text=text,
        urgency=urgency,
        trust_weight=trust_weight,
        metadata=metadata,
    )
    return {
        "msg_id": "msg-x",
        "sender_uid": sender_uid,
        "payload": payload,
        "mode": "two_way",
        "tick": 1,
    }


def _context(messages: list[dict] | None = None, **overrides) -> dict:
    ctx: dict = {
        "agent_uid": (1, 0, 0),
        "tick": 0,
        "minute_of_day": 30,
        "observation": {"is_transit": 0.0, "hour_of_day": 9.0},
        "beliefs": {},
        "goals": {},
        "emotions": {},
        "needs": {},
        "attitudes": {"responsiveness": 0.5},
        "recent_messages": messages or [],
        "recent_memory": [],
        "current_place_id": 100,
        "rank_place_id": 100,
        "plan_state": {
            "index": 0,
            "is_activity": True,
            "is_leg": False,
            "activity_id": 0,
            "previous_activity_id": None,
            "next_activity_id": None,
            "next_future_activity_id": None,
            "next_flexible_activity_id": None,
            "next_social_activity_id": None,
        },
    }
    ctx.update(overrides)
    return ctx


# ------------------------- constructor validation --------------------------


def test_constructor_rejects_nonpositive_signal_cap():
    with pytest.raises(ValueError, match="signal_cap"):
        LocalBehaviorLLMAdapter(signal_cap=0)


def test_constructor_rejects_memory_decay_above_one():
    with pytest.raises(ValueError, match="memory_decay"):
        LocalBehaviorLLMAdapter(memory_decay=1.5)


def test_constructor_rejects_memory_decay_below_zero():
    with pytest.raises(ValueError, match="memory_decay"):
        LocalBehaviorLLMAdapter(memory_decay=-0.1)


def test_constructor_rejects_zero_appraisal_window():
    with pytest.raises(ValueError, match="appraisal_window"):
        LocalBehaviorLLMAdapter(appraisal_window=0)


# -------------------------- single-message paths ---------------------------


def test_warning_drives_stay_home_with_no_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(_context([_message(MessageKind.WARNING, text="Storm", urgency=0.8)]))
    assert proposal["action"] == "stay_home"
    assert proposal["reason"].startswith("A warning")
    assert proposal["message_intent"] is None


def test_invitation_drives_social_contact_with_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context([_message(MessageKind.INVITATION, text="Lunch", trust_weight=0.9)])
    )
    assert proposal["action"] == "seek_social_contact"
    assert proposal["reason"].startswith("An invitation")
    assert proposal["message_intent"] is not None
    assert proposal["message_intent"]["payload"]["text"] == "Invitation received"


def test_request_help_drives_obligation_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context([_message(MessageKind.REQUEST_HELP, text="Now", urgency=0.9)])
    )
    assert proposal["action"] == "send_message"
    assert proposal["reason"].startswith("A request for help")
    assert proposal["message_intent"]["payload"]["text"] == "Help request acknowledged"


def test_coordination_drives_acknowledgment():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(
                    MessageKind.COORDINATION,
                    text="Pickup",
                    topic="pickup",
                    urgency=0.7,
                    trust_weight=0.8,
                )
            ]
        )
    )
    assert proposal["action"] == "send_message"
    assert proposal["reason"].startswith("A coordination request")
    assert proposal["message_intent"]["payload"]["text"] == "Coordination confirmed"


def test_reminder_reinforces_schedule_without_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(_context([_message(MessageKind.REMINDER, text="Don't forget")]))
    assert proposal["action"] == "follow_schedule"
    assert proposal["reason"].startswith("A reminder")
    assert proposal["message_intent"] is None


def test_check_in_falls_through_to_generic_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(_context([_message(MessageKind.CHECK_IN, text="hi")]))
    assert proposal["action"] == "send_message"
    assert proposal["message_intent"] is not None


def test_urgent_announcement_drives_stay_home():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(
                    MessageKind.ANNOUNCEMENT,
                    text="Closure",
                    topic="closure",
                    urgency=0.9,
                    trust_weight=0.9,
                )
            ]
        )
    )
    assert proposal["action"] == "stay_home"
    assert proposal["reason"].startswith("A high-urgency announcement")


def test_trusted_safety_recommendation_drives_stay_home():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(
                    MessageKind.RECOMMENDATION,
                    text="Stay home",
                    topic="stay_home",
                    urgency=0.8,
                    trust_weight=0.9,
                )
            ]
        )
    )
    assert proposal["action"] == "stay_home"
    assert proposal["reason"].startswith("A trusted urgent recommendation")


# ------------------ multi-message ordering (Bug #1 region) -----------------


def test_invitation_then_request_help_replies_to_helper():
    """Pre-existing test_person.py coverage — REQUEST_HELP arrives last."""
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(MessageKind.INVITATION, text="lunch", trust_weight=0.8),
                _message(MessageKind.REQUEST_HELP, text="help", urgency=0.9),
            ]
        )
    )
    assert proposal["action"] == "send_message"
    assert proposal["reason"].startswith("A request for help")
    assert proposal["message_intent"]["payload"]["text"] == "Help request acknowledged"


def test_request_help_first_then_invitation_still_replies_to_helper():
    """Bug #1 regression — order swapped from the test above. Pre-fix, the
    invitation overwrote the obligation reply target *and* the dominant
    reason. Per-channel traces preserve both."""
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(
                    MessageKind.REQUEST_HELP,
                    text="help",
                    urgency=0.9,
                    sender_uid=(99, 0, 0),
                    sender_place_id=500,
                ),
                _message(
                    MessageKind.INVITATION,
                    text="lunch",
                    trust_weight=0.8,
                    sender_uid=(7, 0, 0),
                    sender_place_id=700,
                ),
            ]
        )
    )
    intent = proposal["message_intent"]
    assert proposal["action"] == "send_message"
    assert proposal["reason"].startswith("A request for help")
    assert intent is not None
    assert intent["receiver_uid"] == (99, 0, 0)
    assert intent["receiver_place_id"] == 500
    assert intent["payload"]["text"] == "Help request acknowledged"


def test_warning_first_then_invitation_still_cites_warning():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(MessageKind.WARNING, text="storm", urgency=0.9),
                _message(MessageKind.INVITATION, text="lunch", trust_weight=0.9),
            ]
        )
    )
    assert proposal["action"] == "stay_home"
    assert proposal["reason"].startswith("A warning")


# ------------------ Bug #2: no self-routing on missing place_id ------------


def test_missing_sender_place_id_does_not_self_route_reply():
    adapter = LocalBehaviorLLMAdapter()
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(
                    MessageKind.REQUEST_HELP,
                    text="help",
                    urgency=0.9,
                    sender_place_id=None,
                )
            ]
        )
    )
    assert proposal["action"] == "send_message"
    assert proposal["message_intent"] is None


# --------------------- Bug #3: memory-window filter order ------------------


def test_memory_appraisal_event_found_past_five_filler_events():
    adapter = LocalBehaviorLLMAdapter()
    appraisal_event = {
        "event_type": "message_appraisal",
        "summary": "remembered warning",
        "data": {
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
    filler = [{"event_type": "llm_proposal", "summary": f"routine {i}", "data": {}} for i in range(10)]
    proposal = adapter.generate_behavior_proposal(_context([], recent_memory=[appraisal_event, *filler]))
    assert proposal["action"] == "stay_home"
    assert proposal["reason"].startswith("Recent caution in memory")


# ---------- Storage-time memory decay (design contract) --------------------


def test_storage_decay_attenuates_trace_on_memory_hop():
    """Pinned by tests/test_person.py::
    test_llm_behavior_engine_memory_decay_reduces_signal_over_successive_ticks.

    Confirms that aggregating memory and re-storing reduces signal magnitude
    by exactly one ``memory_decay`` factor per cycle. Both decay sites
    (aggregation-time per-hop weighting, storage-time cross-tick
    attenuation) are required for the design's "memory fades over time"
    property; this test fails if either is removed.
    """
    adapter = LocalBehaviorLLMAdapter(memory_decay=0.5)
    appraisal_event = {
        "event_type": "message_appraisal",
        "summary": "remembered warning",
        "data": {
            "safety_signal": 1.0,
            "social_signal": 0.0,
            "obligation_signal": 0.0,
            "schedule_signal": 0.0,
            "reply_signal": 0.0,
            "defer_signal": 0.0,
            "skip_signal": 0.0,
            "cancel_social_signal": 0.0,
            "defer_minutes": 0,
            "source_count": 1,
            "responsiveness": 0.5,
        },
    }
    proposal = adapter.generate_behavior_proposal(_context([], recent_memory=[appraisal_event]))
    trace = proposal["memory_trace"]
    # Aggregation: weight=1.0 for the most recent (and only) event ⇒ 1.0.
    # Storage: from_memory=True ⇒ multiply by memory_decay=0.5 ⇒ 0.5.
    assert trace["safety_signal"] == pytest.approx(0.5)


# ------------------------- canonical schema shape --------------------------


def test_every_return_path_emits_canonical_schema():
    adapter = LocalBehaviorLLMAdapter()
    expected = set(LocalBehaviorLLMAdapter.PROPOSAL_SCHEMA_KEYS)

    paths: dict[str, dict] = {}

    paths["messages"] = _context([_message(MessageKind.WARNING, text="Storm", urgency=0.8)])

    memory_event = {
        "event_type": "message_appraisal",
        "summary": "remembered warning",
        "data": {
            "safety_signal": 1.0,
            "social_signal": 0.0,
            "obligation_signal": 0.0,
            "schedule_signal": 0.0,
            "reply_signal": 0.0,
            "defer_signal": 0.0,
            "skip_signal": 0.0,
            "cancel_social_signal": 0.0,
            "defer_minutes": 0,
            "source_count": 1,
            "responsiveness": 0.5,
        },
    }
    paths["memory"] = _context([], recent_memory=[memory_event])

    transit_ctx = _context([])
    transit_ctx["observation"] = {"is_transit": 1.0, "hour_of_day": 9.0}
    paths["transit"] = transit_ctx

    late_ctx = _context([])
    late_ctx["observation"] = {"is_transit": 0.0, "hour_of_day": 22.0}
    paths["late_hour"] = late_ctx

    default_ctx = _context([])
    default_ctx["observation"] = {"is_transit": 0.0, "hour_of_day": 9.0}
    paths["default"] = default_ctx

    for label, ctx in paths.items():
        proposal = adapter.generate_behavior_proposal(ctx)
        assert (
            set(proposal.keys()) == expected
        ), f"{label} path missing/extra keys: {sorted(set(proposal.keys()) ^ expected)}"
        for key in ("emotion_updates", "need_updates", "attitude_updates", "memory_trace"):
            assert isinstance(proposal[key], dict), f"{label}: {key} is {type(proposal[key]).__name__}"
        for key in ("message_intent", "plan_adjustment"):
            assert proposal[key] is None or isinstance(
                proposal[key], dict
            ), f"{label}: {key} is {type(proposal[key]).__name__}"


# ------------------- configurable appraisal window -------------------------


def test_appraisal_window_one_only_considers_most_recent_message():
    """With window=1, only the last message contributes to the appraisal.

    Sending an INVITATION (social) followed by a low-urgency CHECK_IN should
    pick the CHECK_IN (generic) under window=1, because the invitation
    falls outside the window.
    """
    adapter = LocalBehaviorLLMAdapter(appraisal_window=1)
    proposal = adapter.generate_behavior_proposal(
        _context(
            [
                _message(MessageKind.INVITATION, text="lunch", trust_weight=0.9),
                _message(MessageKind.CHECK_IN, text="hi"),
            ]
        )
    )
    # With only the CHECK_IN seen, the social signal never accumulates and
    # no INVITATION reason is set.
    assert "invitation" not in proposal["reason"].lower()


# ----------------- BehaviorLLMAdapter contract --------------------------


def test_local_adapter_conforms_to_behavior_llm_adapter_protocol():
    """LocalBehaviorLLMAdapter must structurally satisfy BehaviorLLMAdapter."""
    adapter = LocalBehaviorLLMAdapter()
    assert isinstance(adapter, BehaviorLLMAdapter)


def test_proposal_actions_match_local_adapter_actions():
    """Every action emitted by the local adapter must lie in PROPOSAL_ACTIONS.

    Walks each return path (messages, memory, transit, late hour, default)
    and verifies the produced action is in the canonical vocabulary.
    """
    adapter = LocalBehaviorLLMAdapter()
    contexts: list[dict] = [
        _context([_message(MessageKind.WARNING, text="storm", urgency=0.9)]),
        _context([_message(MessageKind.INVITATION, text="lunch", trust_weight=0.9)]),
        _context([_message(MessageKind.REQUEST_HELP, text="now", urgency=0.9)]),
        _context([_message(MessageKind.REMINDER, text="don't forget")]),
        _context([_message(MessageKind.CHECK_IN, text="hi")]),
    ]
    transit_ctx = _context([])
    transit_ctx["observation"] = {"is_transit": 1.0, "hour_of_day": 9.0}
    contexts.append(transit_ctx)

    late_ctx = _context([])
    late_ctx["observation"] = {"is_transit": 0.0, "hour_of_day": 22.0}
    contexts.append(late_ctx)

    default_ctx = _context([])
    default_ctx["observation"] = {"is_transit": 0.0, "hour_of_day": 9.0}
    contexts.append(default_ctx)

    for ctx in contexts:
        proposal = adapter.generate_behavior_proposal(ctx)
        assert (
            proposal["action"] in PROPOSAL_ACTIONS
        ), f"action {proposal['action']!r} not in canonical PROPOSAL_ACTIONS"


def test_proposal_plan_adjustment_kinds_match_local_adapter_kinds():
    """Every plan_adjustment.kind emitted by the local adapter must lie in
    PROPOSAL_PLAN_ADJUSTMENT_KINDS — and the local adapter must be able to
    emit each one given the right context."""
    adapter = LocalBehaviorLLMAdapter(
        activity_semantics_overrides={"social_ids": [99]},
    )

    # WARNING -> preserve_home_activity
    p_safety = adapter.generate_behavior_proposal(_context([_message(MessageKind.WARNING, text="storm", urgency=0.9)]))
    assert p_safety["plan_adjustment"] is not None
    assert p_safety["plan_adjustment"]["kind"] == "preserve_home_activity"

    # COORDINATION pickup with a flexible next activity -> defer_next_activity
    flex_ctx = _context(
        [
            _message(
                MessageKind.COORDINATION,
                text="Pickup",
                topic="pickup",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]
    )
    flex_ctx["plan_state"]["next_flexible_activity_id"] = 1
    p_defer = adapter.generate_behavior_proposal(flex_ctx)
    assert p_defer["plan_adjustment"] is not None
    assert p_defer["plan_adjustment"]["kind"] == "defer_next_activity"

    # COORDINATION cancel with a flexible next activity -> skip_flexible_activity
    skip_ctx = _context(
        [
            _message(
                MessageKind.COORDINATION,
                text="Cancel pickup",
                topic="cancel",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]
    )
    skip_ctx["plan_state"]["next_flexible_activity_id"] = 1
    p_skip = adapter.generate_behavior_proposal(skip_ctx)
    assert p_skip["plan_adjustment"] is not None
    assert p_skip["plan_adjustment"]["kind"] == "skip_flexible_activity"

    # COORDINATION cancel_social with a social next activity -> cancel_social_activity
    cancel_ctx = _context(
        [
            _message(
                MessageKind.COORDINATION,
                text="Cancel lunch",
                topic="cancel_social",
                urgency=0.8,
                trust_weight=0.8,
            )
        ]
    )
    cancel_ctx["plan_state"]["next_social_activity_id"] = 99  # matches override
    p_cancel = adapter.generate_behavior_proposal(cancel_ctx)
    assert p_cancel["plan_adjustment"] is not None
    assert p_cancel["plan_adjustment"]["kind"] == "cancel_social_activity"

    # All emitted kinds must lie in the canonical vocabulary.
    for proposal in (p_safety, p_defer, p_skip, p_cancel):
        assert proposal["plan_adjustment"]["kind"] in PROPOSAL_PLAN_ADJUSTMENT_KINDS


def test_unsupported_plan_adjustment_kind_raises_typed_error():
    """Internal callers use string literals; the typed error guards against
    future drift between the bounded vocabulary and the dispatch."""
    adapter = LocalBehaviorLLMAdapter()
    # Direct call into the private dispatch to exercise the guard.
    from casmsocial.llm_adapter import MessageAppraisal

    appraisal = MessageAppraisal()
    with pytest.raises(UnsupportedPlanAdjustmentKindError) as excinfo:
        adapter._plan_adjustment_from_policy(appraisal, "not_a_real_kind")
    # Error message includes the invalid kind and the canonical vocabulary.
    msg = str(excinfo.value)
    assert "not_a_real_kind" in msg
    for kind in PROPOSAL_PLAN_ADJUSTMENT_KINDS:
        assert kind in msg
