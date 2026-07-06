"""Person Agent Base Class"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import astuple, dataclass, field
from typing import Any, ClassVar, NamedTuple

import repast4py.context as ctx
from loguru import logger
from mpi4py import MPI
from repast4py import core
from repast4py.space import ContinuousPoint as cpt

from casmsocial.activities import (
    Act,
    Leg,
    PlanElement,
    Plans,
    PlanState,
    activity_at,
    activity_semantics_for,
    resolve_plan_state,
    restore_plans,
    serialize_plans,
)
from casmsocial.communication.types import CommMessage, MessageIntent
from casmsocial.data_utilities import create_dataclass_record_from_dict
from casmsocial.llm_adapter import (
    BehaviorLLMAdapter,
    LocalBehaviorLLMAdapter,
    UnsupportedAdapterKindError,
)
from casmsocial.sim_time import SimTime

rank = MPI.COMM_WORLD.Get_rank()
DEFAULT_LOCATION = cpt(x=0, y=0, z=0)


class BehaviorEngine:
    """
    A behavioral engine that governs the decision-making of a Person agent.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: Person):
        self.agent = agent
        # self.social_network = social_network
        # self.environment = environment

    def decide(self, context: ctx.SharedContext, cal: SimTime):
        """
        Simulate decision-making based on the agent's attributes, social network, and environment.
        """
        pass
        # temperature = self.environment.get("temperature", 20)
        # precipitation = self.environment.get("precipitation", 0)
        # social_influence = (
        #     sum(friend.income for friend in self.social_network) / len(self.social_network)
        #     if self.social_network
        #     else 0
        # )

        # if self.agent.age < 18:
        #     self.agent.current_action = "Studying"
        # elif self.agent.income + social_influence > 50000:
        #     self.agent.current_action = "Investing"
        # elif temperature < 0 and precipitation > 10:
        #     self.agent.current_action = "Staying Indoors"
        # else:
        #     self.agent.current_action = "Working"


class ScheduleBehaviorEngine(BehaviorEngine):
    """Minimal behavior engine that follows the existing schedule only."""

    __slots__ = ()

    def decide(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Move the agent according to its current plan without cognition state."""
        places_proj = context.get_projection("places_projection")
        self.agent.move(cal, places_proj)


@dataclass(slots=True)
class CognitiveState:
    """Mutable cognitive state used by BehaviorEngineV2."""

    beliefs: dict[str, float] = field(default_factory=dict)
    goals: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    needs: dict[str, float] = field(default_factory=dict)
    attitudes: dict[str, float] = field(default_factory=dict)
    episodic_memory: list[dict[str, Any]] = field(default_factory=list)
    stress: float = 0.0
    social_influence: float = 0.0
    last_decision: str = "follow_schedule"
    decision_confidence: float = 0.5
    last_llm_tick: int | None = None
    llm_failures: int = 0
    last_llm_summary: str = ""


@dataclass(slots=True)
class PlanAdjustment:
    """Bounded request for a future plan change."""

    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEvent:
    """Compact behavior event retained for short-term reasoning."""

    tick: int
    event_type: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMBehaviorProposal:
    """Structured result returned by the local LLM-style adapter."""

    action: str
    reason: str
    emotion_updates: dict[str, float] = field(default_factory=dict)
    need_updates: dict[str, float] = field(default_factory=dict)
    attitude_updates: dict[str, float] = field(default_factory=dict)
    message_intent: MessageIntent | None = None
    plan_adjustment: PlanAdjustment | None = None
    memory_trace: dict[str, Any] = field(default_factory=dict)


class BehaviorEngineV2(BehaviorEngine):
    """A staged cognitive decision pipeline for Person agents.

    This is a concrete skeleton designed for extension. It separates behavior
    into four phases:
      1) perception
      2) belief update
      3) goal scoring
      4) action selection

    The default implementation keeps behavior conservative and compatible with
    the existing schedule-driven movement.
    """

    __slots__ = ("cognition",)

    _DEFAULT_BELIEFS: ClassVar[dict[str, float]] = {
        "risk": 0.1,
        "routine_preference": 0.8,
        "social_affinity": 0.5,
    }
    _DEFAULT_GOALS: ClassVar[dict[str, float]] = {
        "follow_schedule": 1.0,
        "stay_home": 0.5,
        "seek_social_contact": 0.3,
    }
    _DEFAULT_EMOTIONS: ClassVar[dict[str, float]] = {
        "calm": 0.7,
        "fear": 0.1,
        "joy": 0.3,
    }
    _DEFAULT_NEEDS: ClassVar[dict[str, float]] = {
        "safety": 0.2,
        "social": 0.4,
        "rest": 0.3,
    }
    _DEFAULT_ATTITUDES: ClassVar[dict[str, float]] = {
        "responsiveness": 0.5,
    }

    def __init__(self, agent: Person):
        super().__init__(agent)
        self.cognition = CognitiveState(
            beliefs=self._DEFAULT_BELIEFS.copy(),
            goals=self._DEFAULT_GOALS.copy(),
            emotions=self._DEFAULT_EMOTIONS.copy(),
            needs=self._DEFAULT_NEEDS.copy(),
            attitudes=self._DEFAULT_ATTITUDES.copy(),
        )

    def decide(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Run one cognitive decision cycle and execute the chosen action."""
        observation = self._perceive(context, cal)
        self._update_beliefs(observation)
        goal_scores = self._evaluate_goals(observation)
        action = self._select_action(goal_scores)
        self._execute_action(action, context, cal)
        self._update_memory(observation, action)

    def _perceive(self, context: ctx.SharedContext, cal: SimTime) -> dict[str, float]:
        """Collect a compact feature vector from local context and clock."""
        incoming_messages = float(len(self.agent.recent_messages))
        hour = float(cal.hour_of_day)
        is_weekday = 1.0 if cal.is_weekday else 0.0
        current_leg = self.agent.current_leg(cal)
        return {
            "incoming_messages": incoming_messages,
            "hour_of_day": hour,
            "is_weekday": is_weekday,
            "is_transit": 1.0 if current_leg is not None else 0.0,
            "pending_acks": float(len(self.agent.pending_acks)),
        }

    def _update_beliefs(self, observation: dict[str, float]) -> None:
        """Apply lightweight belief dynamics.

        The formulas here are intentionally simple placeholders for model-
        specific calibration.
        """
        novelty = min(1.0, observation["incoming_messages"] / 10.0)
        current_risk = self.cognition.beliefs.get("risk", 0.1)
        self.cognition.beliefs["risk"] = max(0.0, min(1.0, 0.9 * current_risk + 0.1 * novelty))

        current_affinity = self.cognition.beliefs.get("social_affinity", 0.5)
        self.cognition.social_influence = max(0.0, min(1.0, 0.8 * current_affinity + 0.2 * novelty))

    def _evaluate_goals(self, observation: dict[str, float]) -> dict[str, float]:
        """Score candidate goals based on beliefs and temporal cues."""
        risk = self.cognition.beliefs.get("risk", 0.1)
        routine_preference = self.cognition.beliefs.get("routine_preference", 0.8)
        social_affinity = self.cognition.beliefs.get("social_affinity", 0.5)

        follow_schedule = self.cognition.goals.get("follow_schedule", 1.0) * routine_preference
        stay_home = self.cognition.goals.get("stay_home", 0.5) * risk
        seek_social_contact = (
            self.cognition.goals.get("seek_social_contact", 0.3)
            * social_affinity
            * (1.0 - risk)
            * observation["is_weekday"]
        )
        return {
            "follow_schedule": follow_schedule,
            "stay_home": stay_home,
            "seek_social_contact": seek_social_contact,
        }

    def _select_action(self, goal_scores: dict[str, float]) -> str:
        """Select the highest-scoring action.

        Tie-break policy: deterministic lexical order for reproducibility.
        """
        max_score = max(goal_scores.values())
        top_actions = sorted([action for action, score in goal_scores.items() if score == max_score])
        selected = top_actions[0]
        self.cognition.last_decision = selected
        self.cognition.decision_confidence = max_score
        return selected

    def _execute_action(self, action: str, context: ctx.SharedContext, cal: SimTime) -> None:
        """Execute selected action against simulation state."""
        places_proj = context.get_projection("places_projection")
        if action == "stay_home":
            self._force_home_if_possible(places_proj, cal)
            return

        if action == "seek_social_contact":
            # Skeleton behavior: currently falls back to schedule movement.
            # A richer implementation can query high-contact locations.
            self.agent.move(cal, places_proj)
            return

        self.agent.move(cal, places_proj)

    def _force_home_if_possible(self, places_proj, cal: SimTime) -> None:
        """Move to home place (index 0) if available."""
        home_place_id = self.agent.places[0] if len(self.agent.places) > 0 else None
        if home_place_id is None:
            return
        home_place = places_proj.lookup_place(home_place_id)
        if home_place is None:
            return
        self.agent.state.place_id = home_place_id
        self.agent.state.rank_place_id = home_place_id
        places_proj.move_agent_to_place(self.agent, home_place)
        self.agent.state.minute_last_moved = cal.minute_of_day

    def _update_memory(self, observation: dict[str, float], action: str) -> None:
        """Persist minimal traces for future decision logic."""
        self.cognition.memory["last_hour_of_day"] = observation["hour_of_day"]
        self.cognition.memory["last_incoming_messages"] = observation["incoming_messages"]
        self.cognition.memory["last_action"] = 1.0 if action == "follow_schedule" else 0.0


class LLMBehaviorEngine(BehaviorEngineV2):
    """Behavior engine that uses a configurable LLM adapter and typed proposals.

    The adapter is selected via ``_config["adapter"]`` and is constructed once
    per engine instance. Two named kinds are built in:

    * ``"local"`` (default) — :class:`LocalBehaviorLLMAdapter`. Deterministic,
      no network, no third-party dependencies.
    * ``"anthropic"`` — :class:`~casmsocial.remote_llm_adapter.AnthropicBehaviorLLMAdapter`.
      Calls Anthropic Claude via the optional ``anthropic`` SDK
      (``uv sync --extra remote``).

    For dependency injection (testing, custom providers, in-process Bedrock /
    OpenAI / vLLM clients), set ``_config["adapter_instance"]`` to a pre-built
    object satisfying :class:`~casmsocial.llm_adapter.BehaviorLLMAdapter`. When
    set, this short-circuits the named-kind dispatch and is used directly.
    """

    _config: ClassVar[dict[str, Any]] = {
        # Adapter selection.
        "adapter": "local",
        "adapter_instance": None,
        # Local-adapter parameters.
        "deliberation_interval": 60,
        "max_memory_events": 20,
        "signal_cap": 1.5,
        "memory_decay": 0.65,
        "activity_semantics_overrides": {},
        # Anthropic-adapter parameters (consumed only when adapter == "anthropic").
        "anthropic_model": "claude-haiku-4-5-20251001",
        "anthropic_api_key": None,
        "anthropic_max_tokens": 1024,
    }

    __slots__ = ("adapter", "deliberation_interval", "max_memory_events")

    @classmethod
    def configure(cls, **kwargs) -> None:
        """Set runtime configuration shared by new engine instances."""
        cls._config = {
            **cls._config,
            **{key: value for key, value in kwargs.items() if value is not None},
        }

    @classmethod
    def _build_adapter_from_config(cls, config: dict[str, Any]) -> BehaviorLLMAdapter:
        """Construct the configured adapter from a config dict.

        Exposed as a classmethod so tests can exercise adapter selection
        without standing up a full ``Person`` / MPI environment.
        """
        # Escape hatch: a pre-built adapter overrides everything else. Used
        # for dependency injection (custom providers, in-process mocks).
        instance = config.get("adapter_instance")
        if instance is not None:
            return instance

        kind = str(config.get("adapter", "local")).lower()
        if kind == "local":
            return LocalBehaviorLLMAdapter(
                signal_cap=float(config["signal_cap"]),
                memory_decay=float(config["memory_decay"]),
                activity_semantics_overrides=dict(config.get("activity_semantics_overrides", {})),
            )
        if kind == "anthropic":
            # Lazy import so installs without the [remote] extra don't pay
            # the import cost when only the local adapter is used.
            from casmsocial.remote_llm_adapter import AnthropicBehaviorLLMAdapter

            return AnthropicBehaviorLLMAdapter(
                model=str(config.get("anthropic_model", "claude-haiku-4-5-20251001")),
                api_key=config.get("anthropic_api_key") or None,
                max_tokens=int(config.get("anthropic_max_tokens", 1024)),
            )

        raise UnsupportedAdapterKindError(kind)

    def __init__(self, agent: Person):
        super().__init__(agent)
        self.adapter: BehaviorLLMAdapter = self._build_adapter_from_config(self._config)
        self.deliberation_interval = int(self._config["deliberation_interval"])
        self.max_memory_events = int(self._config["max_memory_events"])

    def decide(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Run a bounded deliberation cycle and fall back to deterministic behavior."""
        observation = self._perceive(context, cal)
        self._update_beliefs(observation)

        if not self._needs_deliberation(observation, cal):
            goal_scores = self._evaluate_goals(observation)
            action = self._select_action(goal_scores)
            self._execute_action(action, context, cal)
            self._update_memory(observation, action)
            return

        try:
            proposal = self._deliberate_with_llm(observation, cal)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Agent {} local deliberation failed: {}", self.agent.id, exc)
            self.cognition.llm_failures += 1
            super().decide(context, cal)
            return

        if proposal is None:
            super().decide(context, cal)
            return

        self._apply_proposal(proposal, context, cal)
        self._update_memory(observation, proposal.action)

    def _needs_deliberation(self, observation: dict[str, float], cal: SimTime) -> bool:
        """Limit deliberation to salient events and coarse intervals."""
        if self.agent.recent_messages:
            return True
        if observation["is_transit"] > 0.0:
            return True

        last_tick = self.cognition.last_llm_tick
        if last_tick is None:
            return True
        return int(cal.tick) - last_tick >= self.deliberation_interval

    def _build_llm_context(self, observation: dict[str, float], cal: SimTime) -> dict[str, Any]:
        """Build a compact context payload for the local adapter."""
        plan_state = self.agent.get_plan_state(cal)
        recent_memories = self.cognition.episodic_memory[-self.max_memory_events :]
        recent_messages = [
            {
                "msg_id": msg.msg_id,
                "sender_uid": msg.sender_uid,
                "payload": dict(msg.payload),
                "mode": msg.mode,
                "tick": msg.tick,
            }
            for msg in self.agent.recent_messages[-5:]
        ]
        return {
            "agent_uid": self.agent.uid,
            "tick": int(cal.tick),
            "minute_of_day": cal.minute_of_day,
            "observation": dict(observation),
            "beliefs": dict(self.cognition.beliefs),
            "goals": dict(self.cognition.goals),
            "emotions": dict(self.cognition.emotions),
            "needs": dict(self.cognition.needs),
            "attitudes": dict(self.cognition.attitudes),
            "recent_messages": recent_messages,
            "recent_memory": list(recent_memories),
            "current_place_id": self.agent.place_id,
            "rank_place_id": self.agent.rank_place_id,
            "plan_state": {
                "index": plan_state.index,
                "is_activity": plan_state.is_activity,
                "is_leg": plan_state.is_leg,
                "activity_id": getattr(plan_state.element, "activity_id", None) if plan_state.is_activity else None,
                "previous_activity_id": getattr(plan_state.previous_activity, "activity_id", None),
                "next_activity_id": getattr(plan_state.next_activity, "activity_id", None),
                "next_future_activity_id": self._next_future_activity_id(cal),
                "next_flexible_activity_id": self._next_flexible_activity_id(cal),
                "next_social_activity_id": self._next_social_activity_id(cal),
            },
        }

    def _deliberate_with_llm(self, observation: dict[str, float], cal: SimTime) -> LLMBehaviorProposal | None:
        """Run the local adapter and parse the proposal."""
        context_payload = self._build_llm_context(observation, cal)
        proposal_data = self.adapter.generate_behavior_proposal(context_payload)
        proposal = self._parse_llm_response(proposal_data)
        self.cognition.last_llm_tick = int(cal.tick)
        self.cognition.last_llm_summary = proposal.reason
        return proposal

    def _next_future_activity_id(self, cal: SimTime) -> int | None:
        """Return the next future activity id in the selected plan, if any."""
        if not self.agent.plans:
            return None
        activities_idx = self.agent.selectActivities(cal)
        if activities_idx >= len(self.agent.plans):
            return None

        for element in self.agent.plans[activities_idx]:
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                return int(element.activity_id)
        return None

    def _next_flexible_activity_id(self, cal: SimTime) -> int | None:
        """Return the next future flexible activity id in the selected plan, if any."""
        if not self.agent.plans:
            return None
        activities_idx = self.agent.selectActivities(cal)
        if activities_idx >= len(self.agent.plans):
            return None

        for element in self.agent.plans[activities_idx]:
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                semantics = activity_semantics_for(
                    element.activity_id,
                    dict(self._config.get("activity_semantics_overrides", {})),
                )
                if semantics.is_flexible:
                    return int(element.activity_id)
        return None

    def _next_social_activity_id(self, cal: SimTime) -> int | None:
        """Return the next future social activity id in the selected plan, if any."""
        if not self.agent.plans:
            return None
        activities_idx = self.agent.selectActivities(cal)
        if activities_idx >= len(self.agent.plans):
            return None

        for element in self.agent.plans[activities_idx]:
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                semantics = activity_semantics_for(
                    element.activity_id,
                    dict(self._config.get("activity_semantics_overrides", {})),
                )
                if semantics.is_social:
                    return int(element.activity_id)
        return None

    def _parse_llm_response(self, data: dict[str, Any]) -> LLMBehaviorProposal:
        """Validate the local adapter response."""
        action = str(data.get("action", "follow_schedule"))
        if action not in {"follow_schedule", "stay_home", "seek_social_contact", "send_message"}:
            raise ValueError(f"Unsupported action: {action}")

        message_intent = None
        raw_intent = data.get("message_intent")
        if raw_intent is not None:
            message_intent = MessageIntent(
                sender_uid=tuple(raw_intent["sender_uid"]),
                receiver_uid=tuple(raw_intent["receiver_uid"]),
                receiver_place_id=int(raw_intent["receiver_place_id"]),
                mode=str(raw_intent["mode"]),
                payload=dict(raw_intent.get("payload", {})),
            )

        plan_adjustment = None
        raw_adjustment = data.get("plan_adjustment")
        if raw_adjustment is not None:
            plan_adjustment = PlanAdjustment(
                kind=str(raw_adjustment["kind"]),
                detail=dict(raw_adjustment.get("detail", {})),
            )

        return LLMBehaviorProposal(
            action=action,
            reason=str(data.get("reason", "")),
            emotion_updates={k: float(v) for k, v in dict(data.get("emotion_updates", {})).items()},
            need_updates={k: float(v) for k, v in dict(data.get("need_updates", {})).items()},
            attitude_updates={k: float(v) for k, v in dict(data.get("attitude_updates", {})).items()},
            message_intent=message_intent,
            plan_adjustment=plan_adjustment,
            memory_trace=dict(data.get("memory_trace", {})),
        )

    def _apply_proposal(self, proposal: LLMBehaviorProposal, context: ctx.SharedContext, cal: SimTime) -> None:
        """Apply bounded cognitive updates and execute an existing action."""
        self.cognition.emotions.update(proposal.emotion_updates)
        self.cognition.needs.update(proposal.need_updates)
        self.cognition.attitudes.update(proposal.attitude_updates)
        requested_kind = proposal.plan_adjustment.kind if proposal.plan_adjustment is not None else ""
        adjustment_result: dict[str, Any] = {
            "plan_adjustment_requested_kind": requested_kind,
            "plan_adjustment_applied": False,
            "plan_adjustment_skip_reason": "",
            "plan_adjustment_kind": "",
            "plan_adjustment_delay_minutes": 0,
            "plan_adjustment_target_activity_id": -1,
            "plan_adjustment_target_place_id": 0,
        }

        if proposal.plan_adjustment is not None:
            applied_adjustment = self._apply_plan_adjustment(proposal.plan_adjustment, cal)
            if applied_adjustment is not None:
                adjustment_result.update(applied_adjustment)

        self._record_memory_event(
            cal=cal,
            event_type="message_appraisal" if proposal.memory_trace else "llm_proposal",
            summary=proposal.reason or proposal.action,
            data={
                "action": proposal.action,
                **adjustment_result,
                **dict(proposal.memory_trace),
            },
        )

        if proposal.message_intent is not None:
            self.agent.queue_message_intent(proposal.message_intent)

        action = proposal.action
        if action == "send_message":
            action = "follow_schedule"

        self._execute_action(action, context, cal)

    def _apply_plan_adjustment(self, adjustment: PlanAdjustment, cal: SimTime) -> dict[str, Any]:
        """Apply a constrained deterministic plan adjustment."""
        if not self.agent.plans:
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "no_plans",
            }
        activities_idx = self.agent.selectActivities(cal)
        if activities_idx >= len(self.agent.plans):
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "invalid_plan_index",
            }
        if adjustment.kind == "preserve_home_activity":
            current_plan = list(self.agent.plans[activities_idx])
            home_place_id = self.agent.places[0] if self.agent.places else 0
            home_activity = Act(
                person_id=self.agent.id,
                activity_id=0,
                activity_sequence=int(adjustment.detail.get("activity_sequence", 0)),
                starttime_min=cal.minute_of_day,
                endtime_min=1439,
                place_id=home_place_id,
            )

            preserved_prefix: list[PlanElement] = []
            for element in current_plan:
                if isinstance(element, Act) and element.endtime_min < cal.minute_of_day:
                    preserved_prefix.append(element)
                    continue
                break

            self.agent.plans[activities_idx] = preserved_prefix + [home_activity]
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": adjustment.kind,
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": int(home_activity.activity_id),
                "plan_adjustment_target_place_id": int(home_activity.place_id),
            }

        if adjustment.kind == "defer_next_activity":
            return self._defer_next_activity(activities_idx, cal, adjustment)

        if adjustment.kind == "skip_flexible_activity":
            return self._skip_flexible_activity(activities_idx, cal, adjustment)

        if adjustment.kind == "cancel_social_activity":
            return self._cancel_social_activity(activities_idx, cal, adjustment)

        return {
            "plan_adjustment_requested_kind": adjustment.kind,
            "plan_adjustment_applied": False,
            "plan_adjustment_skip_reason": "unsupported_adjustment",
        }

    def _defer_next_activity(self, activities_idx: int, cal: SimTime, adjustment: PlanAdjustment) -> dict[str, Any]:
        """Delay the next future activity by a bounded amount if slack is available."""
        plan = list(self.agent.plans[activities_idx])
        delay_minutes = int(adjustment.detail.get("delay_minutes", 0))
        if delay_minutes <= 0:
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "non_positive_delay",
            }

        next_index = -1
        next_activity: Act | None = None
        saw_future_activity = False
        for index, element in enumerate(plan):
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                saw_future_activity = True
                semantics = activity_semantics_for(
                    element.activity_id,
                    dict(self._config.get("activity_semantics_overrides", {})),
                )
                if semantics.is_home:
                    continue
                if not semantics.is_travel_sensitive:
                    continue
                next_index = index
                next_activity = element
                break
        if next_index == -1:
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": (
                    "no_eligible_future_activity" if saw_future_activity else "no_future_activity"
                ),
            }

        if not isinstance(next_activity, Act):
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "invalid_target_activity",
            }

        following_activity = next(
            (element for element in plan[next_index + 1 :] if isinstance(element, Act)),
            None,
        )
        available_delay = 1439 - next_activity.endtime_min
        if isinstance(following_activity, Act):
            available_delay = min(available_delay, following_activity.starttime_min - next_activity.endtime_min)
        if available_delay <= 0:
            return {
                "plan_adjustment_requested_kind": adjustment.kind,
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "no_slack",
                "plan_adjustment_target_activity_id": int(next_activity.activity_id),
                "plan_adjustment_target_place_id": int(next_activity.place_id),
            }

        applied_delay = min(delay_minutes, available_delay)
        plan[next_index] = Act(
            person_id=next_activity.person_id,
            activity_id=next_activity.activity_id,
            activity_sequence=next_activity.activity_sequence,
            starttime_min=next_activity.starttime_min + applied_delay,
            endtime_min=next_activity.endtime_min + applied_delay,
            place_id=next_activity.place_id,
        )
        self.agent.plans[activities_idx] = plan
        return {
            "plan_adjustment_requested_kind": adjustment.kind,
            "plan_adjustment_applied": True,
            "plan_adjustment_skip_reason": "",
            "plan_adjustment_kind": adjustment.kind,
            "plan_adjustment_delay_minutes": applied_delay,
            "plan_adjustment_target_activity_id": int(next_activity.activity_id),
            "plan_adjustment_target_place_id": int(next_activity.place_id),
        }

    def _skip_flexible_activity(self, activities_idx: int, cal: SimTime, adjustment: PlanAdjustment) -> dict[str, Any]:
        """Remove the next future flexible activity from the active plan."""
        del adjustment
        plan = list(self.agent.plans[activities_idx])

        target_index = -1
        target_activity: Act | None = None
        saw_future_activity = False
        for index, element in enumerate(plan):
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                saw_future_activity = True
                semantics = activity_semantics_for(
                    element.activity_id,
                    dict(self._config.get("activity_semantics_overrides", {})),
                )
                if not semantics.is_flexible:
                    continue
                target_index = index
                target_activity = element
                break

        if target_index == -1:
            return {
                "plan_adjustment_requested_kind": "skip_flexible_activity",
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": (
                    "no_eligible_future_activity" if saw_future_activity else "no_future_activity"
                ),
            }

        if not isinstance(target_activity, Act):
            return {
                "plan_adjustment_requested_kind": "skip_flexible_activity",
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "invalid_target_activity",
            }

        has_prev_activity = target_index > 0 and isinstance(plan[target_index - 2], Act) if target_index >= 2 else False
        has_next_activity = target_index + 2 < len(plan) and isinstance(plan[target_index + 2], Act)
        remove_start = target_index
        remove_end = target_index + 1

        if target_index > 0 and isinstance(plan[target_index - 1], Leg):
            remove_start = target_index - 1
        if target_index + 1 < len(plan) and isinstance(plan[target_index + 1], Leg):
            remove_end = target_index + 2

        updated_plan = plan[:remove_start] + plan[remove_end:]
        if has_prev_activity and has_next_activity:
            updated_plan.insert(remove_start, Leg(mode="travel"))

        self.agent.plans[activities_idx] = updated_plan
        return {
            "plan_adjustment_requested_kind": "skip_flexible_activity",
            "plan_adjustment_applied": True,
            "plan_adjustment_skip_reason": "",
            "plan_adjustment_kind": "skip_flexible_activity",
            "plan_adjustment_delay_minutes": 0,
            "plan_adjustment_target_activity_id": int(target_activity.activity_id),
            "plan_adjustment_target_place_id": int(target_activity.place_id),
        }

    def _cancel_social_activity(self, activities_idx: int, cal: SimTime, adjustment: PlanAdjustment) -> dict[str, Any]:
        """Remove the next future social activity from the active plan."""
        del adjustment
        plan = list(self.agent.plans[activities_idx])

        target_index = -1
        target_activity: Act | None = None
        saw_future_activity = False
        for index, element in enumerate(plan):
            if isinstance(element, Act) and element.starttime_min > cal.minute_of_day:
                saw_future_activity = True
                semantics = activity_semantics_for(
                    element.activity_id,
                    dict(self._config.get("activity_semantics_overrides", {})),
                )
                if not semantics.is_social:
                    continue
                target_index = index
                target_activity = element
                break

        if target_index == -1:
            return {
                "plan_adjustment_requested_kind": "cancel_social_activity",
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": (
                    "no_eligible_future_activity" if saw_future_activity else "no_future_activity"
                ),
            }

        if not isinstance(target_activity, Act):
            return {
                "plan_adjustment_requested_kind": "cancel_social_activity",
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "invalid_target_activity",
            }

        has_prev_activity = target_index > 0 and isinstance(plan[target_index - 2], Act) if target_index >= 2 else False
        has_next_activity = target_index + 2 < len(plan) and isinstance(plan[target_index + 2], Act)
        remove_start = target_index
        remove_end = target_index + 1

        if target_index > 0 and isinstance(plan[target_index - 1], Leg):
            remove_start = target_index - 1
        if target_index + 1 < len(plan) and isinstance(plan[target_index + 1], Leg):
            remove_end = target_index + 2

        updated_plan = plan[:remove_start] + plan[remove_end:]
        if has_prev_activity and has_next_activity:
            updated_plan.insert(remove_start, Leg(mode="travel"))

        self.agent.plans[activities_idx] = updated_plan
        return {
            "plan_adjustment_requested_kind": "cancel_social_activity",
            "plan_adjustment_applied": True,
            "plan_adjustment_skip_reason": "",
            "plan_adjustment_kind": "cancel_social_activity",
            "plan_adjustment_delay_minutes": 0,
            "plan_adjustment_target_activity_id": int(target_activity.activity_id),
            "plan_adjustment_target_place_id": int(target_activity.place_id),
        }

    def _record_memory_event(
        self,
        cal: SimTime,
        event_type: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append a bounded short-term memory trace."""
        event = MemoryEvent(
            tick=int(cal.tick),
            event_type=event_type,
            summary=summary,
            data={} if data is None else dict(data),
        )
        self.cognition.episodic_memory.append(
            {
                "tick": event.tick,
                "event_type": event.event_type,
                "summary": event.summary,
                "data": event.data,
            }
        )
        if len(self.cognition.episodic_memory) > self.max_memory_events:
            self.cognition.episodic_memory = self.cognition.episodic_memory[-self.max_memory_events :]


# 1. Define a PersonData Class
@dataclass(slots=True)
class PersonData:
    """Data for a Person."""

    person_id: int
    place_id: int
    rank_place_id: int
    activity_id: int
    places: namedtuple
    activities_idx: int
    minute_last_moved: int = 0
    network: object | None = None


@dataclass(slots=True)
class PersonLastKnownPlace:
    """Data for a Person's last known place.
    This is used to store the last known place of a person.
    """

    person_id: int
    place_id: int
    minute_last_updated: int = 0


@dataclass
class ChiSimPersonData:
    person_id: int
    act_type: int
    place_id: int
    places: namedtuple


# 2. Define a Person Class
# @dataclass(slots=True)
class Person(core.Agent):
    """Person class."""

    # class variables
    TYPE = 0  # class variable
    __person_data_class: type[dataclass] = PersonData
    __behavior_engine: BehaviorEngine = BehaviorEngine

    @classmethod
    def setPersonDataClass(cls, person_data_class: type[dataclass]) -> None:
        """Register a person data class for the person."""
        cls.__person_data_class = person_data_class

    @classmethod
    def getPersonDataClass(cls) -> type[dataclass]:
        """Get the person data class for the person."""
        return cls.__person_data_class

    @classmethod
    def registerBehaviorEngine(cls, behavior_engine: BehaviorEngine) -> None:
        """Register a behavior engine for the person."""
        cls.__behavior_engine = behavior_engine

    @classmethod
    def getBehaviorEngine(cls) -> BehaviorEngine:
        """Get the behavior engine for the person."""
        return cls.__behavior_engine

    @staticmethod
    def _initialize_behavior_engine(
        person: Person,
        behavior_engine_type: type[BehaviorEngine],
    ) -> None:
        """Attach a behavior engine, deferring the stateless schedule engine."""
        if behavior_engine_type is ScheduleBehaviorEngine:
            person._behavior_engine = None
            person._behavior_engine_type = behavior_engine_type
            return
        person.behavior_engine = behavior_engine_type(person)

    @staticmethod
    def _initialize_communication_state(person: Person) -> None:
        """Attach empty communication buffers to a person."""
        person.inbox = []
        person.recent_messages = []
        person.outbound_message_intents = []
        person.pending_acks = {}

    @classmethod
    def from_default_fields(
        cls,
        local_id: int,
        rank: int,
        plans: Plans,
        places: namedtuple,
        *,
        x: int | float | None = 0,
        y: int | float | None = 0,
        minute_last_moved: int = 0,
        network: object | None = None,
        behavior_engine: type[BehaviorEngine] | None = None,
        initialize_communication_state: bool = True,
    ) -> Person:
        """Construct a default Person without per-row dictionary materialization."""
        person = cls.__new__(cls)
        core.Agent.__init__(person, local_id, cls.TYPE, rank)
        person.plans = plans

        if x is None or (isinstance(x, float) and math.isinf(x)):
            x = 0
        if y is None or (isinstance(y, float) and math.isinf(y)):
            y = 0

        person.location = DEFAULT_LOCATION if x == 0 and y == 0 else cpt(x=int(x), y=int(y), z=0)
        person.state = PersonData(
            person_id=local_id,
            place_id=places[0],
            rank_place_id=places[0],
            activity_id=0,
            places=places,
            activities_idx=0,
            minute_last_moved=minute_last_moved,
            network=network,
        )
        person.network = network

        if initialize_communication_state:
            cls._initialize_communication_state(person)

        behavior_engine_type = behavior_engine or cls.getBehaviorEngine()
        cls._initialize_behavior_engine(person, behavior_engine_type)
        return person

    @classmethod
    def from_default_zero_location(
        cls,
        local_id: int,
        rank: int,
        plans: Plans,
        places: namedtuple,
        behavior_engine: type[BehaviorEngine],
        initialize_communication_state: bool = True,
    ) -> Person:
        """Construct a default Person for inputs without custom location columns."""
        person = cls.__new__(cls)
        core.Agent.__init__(person, local_id, cls.TYPE, rank)
        person.plans = plans
        person.location = DEFAULT_LOCATION
        person.state = PersonData(
            person_id=local_id,
            place_id=places[0],
            rank_place_id=places[0],
            activity_id=0,
            places=places,
            activities_idx=0,
        )
        person.network = None

        if initialize_communication_state:
            cls._initialize_communication_state(person)

        if behavior_engine is ScheduleBehaviorEngine:
            person._behavior_engine = None
            person._behavior_engine_type = behavior_engine
        else:
            person.behavior_engine = behavior_engine(person)
        return person

    @classmethod
    def from_default_schedule_zero_location(
        cls,
        local_id: int,
        rank: int,
        weekday_plan: list,
        places: namedtuple,
    ) -> Person:
        """Construct a schedule-only default Person for the hot startup path."""
        person = cls.__new__(cls)
        core.Agent.__init__(person, local_id, cls.TYPE, rank)
        person.plans = [weekday_plan, []]
        person.location = DEFAULT_LOCATION
        home_place_id = places[0]
        person.state = PersonData(local_id, home_place_id, home_place_id, 0, places, 0)
        person.network = None
        person._behavior_engine = None
        person._behavior_engine_type = ScheduleBehaviorEngine
        return person

    # plans: Optional[list[Plan]] = field(default_factory=list)
    # places: Optional[list[int]] = field(default_factory=list)
    # currentPlaceID: Optional[str] = field(default=None)

    def __init__(self, local_id: int, rank: int, plans: Plans, places: namedtuple, initDict: dict):
        """Constructor for the Person class.

        Arguments:
            local_id: The ID for this person on this process, combines with the
                rank to form a simulation-wide unique ID.
            rank: The rank of this process.
            plans: A list of one or more activity plans. Each plan is a list
                of activities and travel legs. The activity entries provide the
                timed schedule for this Person. Each activity has a place type
                (int). The place types are implementation-agnostic so "0"
                could mean "home" in one simulation or "grocery store" in
                another. If there are multiple plans, one plan could be for
                weekdays and another for weekends, for example.
            places: A list of place_id's that correspond to values coming out of
                the schedule. e.g. if the schedule returns "0", this Person will
                try to go to the place with the ID of places[0]
            initDict: A dictionary of initial values for the person
        """
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)

        self.plans: Plans = plans

        person_data_class = Person.getPersonDataClass()
        if person_data_class is PersonData:
            x = initDict.get("x", 0)
            y = initDict.get("y", 0)
            if isinstance(x, float) and math.isinf(x):
                x = 0
            if isinstance(y, float) and math.isinf(y):
                y = 0
            self.location = cpt(x=int(x), y=int(y), z=0)
            self.state = PersonData(
                person_id=local_id,
                place_id=places[0],
                rank_place_id=places[0],
                activity_id=0,
                places=places,
                activities_idx=0,
                minute_last_moved=initDict.get("minute_last_moved", 0),
                network=initDict.get("network"),
            )
        else:
            # `location` is currently referenced required but not used
            if "x" not in initDict:
                initDict["x"] = 0
            if "y" not in initDict:
                initDict["y"] = 0
            if math.isinf(initDict["x"]) or math.isinf(initDict["y"]):
                initDict["x"] = 0
                initDict["y"] = 0

            self.location = cpt(x=int(initDict["x"]), y=int(initDict["y"]), z=0)

            # map input parameters to dict
            initDict["person_id"] = local_id
            initDict["place_id"] = places[0]
            initDict["rank_place_id"] = places[0]
            initDict["activity_id"] = 0
            initDict["activities_idx"] = 0
            # initDict['location'] = starting_location
            initDict["places"] = places
            initDict["network"] = initDict.get("network")

            self.state = create_dataclass_record_from_dict(person_data_class, initDict)
        self.network = self.state.network

        self._initialize_communication_state(self)

        # create the default behavior engine
        self._initialize_behavior_engine(self, Person.getBehaviorEngine())

    @property
    def pt(self) -> cpt:
        """"""
        return self.location

    @property
    def currentPlaceID(self) -> int:
        return self.state.place_id

    @property
    def place_id(self) -> int:
        """Current place identifier, aligned with the serialized person state."""
        return self.state.place_id

    @place_id.setter
    def place_id(self, value: int) -> None:
        self.state.place_id = value

    @property
    def rank_place_id(self) -> int:
        """Place id that determines rank ownership for this person."""
        return self.state.rank_place_id

    @rank_place_id.setter
    def rank_place_id(self, value: int) -> None:
        self.state.rank_place_id = value

    @property
    def places(self) -> list[int]:
        return list(self.state.places)

    @property
    def last_known_place(self) -> PersonLastKnownPlace:
        """Returns the last known location of this person."""
        return PersonLastKnownPlace(
            person_id=self.state.person_id,
            place_id=self.state.place_id,
            minute_last_updated=self.state.minute_last_moved,
        )

    @property
    def behavior_engine(self) -> BehaviorEngine:
        """Return this person's behavior engine, creating lazy schedule engines on demand."""
        behavior_engine = self.__dict__.get("_behavior_engine")
        if behavior_engine is None:
            behavior_engine_type = self.__dict__.get("_behavior_engine_type")
            if behavior_engine_type is None:
                behavior_engine_type = Person.getBehaviorEngine()
            behavior_engine = behavior_engine_type(self)
            self.__dict__["_behavior_engine"] = behavior_engine
        return behavior_engine

    @behavior_engine.setter
    def behavior_engine(self, value: BehaviorEngine | None) -> None:
        self.__dict__["_behavior_engine"] = value
        self.__dict__["_behavior_engine_type"] = type(value) if value is not None else None

    def setBehaviorEngine(self, behaviorEngine: type[BehaviorEngine]):
        """Sets the behavior engine for this person."""
        if behaviorEngine is not ScheduleBehaviorEngine and "inbox" not in self.__dict__:
            self._initialize_communication_state(self)
        self._initialize_behavior_engine(self, behaviorEngine)

    def save(self) -> tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person.
        """
        return (
            self.uid,  # 0: uid is a tuple
            serialize_plans(self.plans),  # 1: plans is a list[Plan]
            tuple(e for e in self.location.coordinates),  # 2: location
            astuple(self.state),  # 3: state is a PersonData object
            dict(getattr(self, "pending_acks", {})),  # 4: lightweight communication state
        )

    def move(self, cal: SimTime, places_proj) -> bool:
        """Update place membership based on the current plan element.

        Args:
            cal: The calendar for the current time.
            places_proj: The PlacesProjection (or EnhancedPlacesProjection) to use for moving the agent.
        Returns:
            True if place membership changed, False otherwise.
        """
        return self.move_at(cal.minute_of_day, cal.is_weekday, places_proj)

    def move_at(self, minute_of_day: int, is_weekday: bool, places_proj) -> bool:
        """Update place membership for a primitive simulation minute.

        This is the hot-path counterpart to ``get_plan_state``. It preserves
        the same plan semantics but avoids allocating a ``PlanState`` for every
        person on every tick.
        """
        plans = self.plans
        if not plans:
            return False

        activities_idx = self._selected_plan_index(is_weekday)
        if activities_idx >= len(plans):
            return False

        previous_activity: Act | None = None
        plan = plans[activities_idx]
        plan_len = len(plan)
        for index, element in enumerate(plan):
            if isinstance(element, Act):
                if element.starttime_min <= minute_of_day <= element.endtime_min:
                    next_place_id = self._place_id_for_activity(element.activity_id)
                    if next_place_id == self.state.place_id:
                        return False
                    return self._transition_to_activity_place_at_minute(
                        element,
                        next_place_id,
                        places_proj,
                        minute_of_day,
                    )
                if minute_of_day > element.endtime_min:
                    previous_activity = element
                continue

            if previous_activity is None:
                continue

            next_activity = None
            next_index = index + 1
            if next_index < plan_len:
                next_element = plan[next_index]
                if isinstance(next_element, Act):
                    next_activity = next_element
                else:
                    for remaining_index in range(next_index + 1, plan_len):
                        remaining_element = plan[remaining_index]
                        if isinstance(remaining_element, Act):
                            next_activity = remaining_element
                            break

            if (
                next_activity is not None
                and previous_activity.endtime_min < minute_of_day < next_activity.starttime_min
            ):
                return self._transition_to_leg_at_minute(
                    element,
                    previous_activity,
                    places_proj,
                    minute_of_day,
                )
        return False

    def _selected_plan_index(self, is_weekday: bool) -> int:
        """Return the active plan index for a weekday/weekend flag."""
        activities_idx = self.state.activities_idx
        if activities_idx < 2:
            if is_weekday:
                return 0
            if len(self.plans) > 1:
                return 1
            return 0
        return activities_idx

    def selectActivities(self, cal: SimTime) -> int:
        """Select the activities for the time of day and day of week.
        Args:
            cal: The calendar for the current time.
        Returns:
            The index of the activities to use for the current time.
        """
        return self._selected_plan_index(cal.is_weekday)

    def selectNextPlace(self, cal: SimTime) -> int:
        """Select the next place to go to based on the schedule for time of
        day and day of week.
        Args:
            cal: The calendar for the current time.
        Returns:
            The ID of the next place to go to.  If the activity is not in the
            list of places, it defaults to home (place_id 0).
        """
        if not self.plans:
            return 0

        time = cal.minute_of_day

        activities_idx = self.selectActivities(cal)
        act = activity_at(self.plans[activities_idx], time)

        next_activity_id = 0  # home is the default
        if act is not None and act.activity_id < len(self.places):
            next_activity_id = int(act.activity_id)
            # else:  if the activity is not in the list of places, go home

        return next_activity_id

    def get_plan_state(self, cal: SimTime) -> PlanState:
        """Resolve the current position within the selected plan."""
        if not self.plans:
            return PlanState(element=None, index=-1)

        activities_idx = self.selectActivities(cal)
        if activities_idx >= len(self.plans):
            return PlanState(element=None, index=-1)

        return resolve_plan_state(self.plans[activities_idx], cal.minute_of_day)

    def current_leg(self, cal: SimTime) -> Leg | None:
        """Return the active leg for the current time, if the person is in transit."""
        plan_state = self.get_plan_state(cal)
        if isinstance(plan_state.element, Leg):
            return plan_state.element
        return None

    def _transition_to_leg(self, plan_state: PlanState, places_proj, cal: SimTime) -> bool:
        """Remove the person from the origin place while in transit."""
        return self._transition_to_leg_at_minute(
            plan_state.element,
            plan_state.previous_activity,
            places_proj,
            cal.minute_of_day,
        )

    def _transition_to_leg_at_minute(
        self,
        leg: PlanElement | None,
        previous_activity: Act | None,
        places_proj,
        minute_of_day: int,
    ) -> bool:
        """Remove the person from the origin place while in transit."""
        if not isinstance(leg, Leg):
            return False
        if not hasattr(places_proj, "remove_agent_from_place"):
            logger.debug("Places projection does not support leg transitions for agent {}.", self.id)
            return False

        current_place = places_proj.get_place_for_agent(self) if hasattr(places_proj, "get_place_for_agent") else None
        origin_place_id = (
            self._place_id_for_activity(previous_activity.activity_id) if previous_activity is not None else None
        )

        if current_place is None:
            return False
        if origin_place_id is not None and current_place.id != origin_place_id:
            return False

        places_proj.remove_agent_from_place(self)
        self.place_id = 0
        if origin_place_id is not None:
            self.rank_place_id = origin_place_id
        self.state.minute_last_moved = minute_of_day
        logger.debug("Rank {}: Agent {} entered leg transit via {}.", rank, self.id, leg)
        return True

    def _transition_to_activity(self, plan_state: PlanState, places_proj, cal: SimTime) -> bool:
        """Move the person into the destination place for the active activity."""
        return self._transition_to_activity_at_minute(plan_state.element, places_proj, cal.minute_of_day)

    def _transition_to_activity_at_minute(
        self,
        activity: PlanElement | None,
        places_proj,
        minute_of_day: int,
    ) -> bool:
        """Move the person into the destination place for the active activity."""
        if not isinstance(activity, Act):
            return False

        next_place_id = self._place_id_for_activity(activity.activity_id)
        return self._transition_to_activity_place_at_minute(activity, next_place_id, places_proj, minute_of_day)

    def _transition_to_activity_place_at_minute(
        self,
        activity: Act,
        next_place_id: int,
        places_proj,
        minute_of_day: int,
    ) -> bool:
        """Move the person into an already-resolved activity place."""
        if not next_place_id:
            logger.debug("Agent {} has no place to go - going remote.", self.id)
            logger.debug("places = {}", self.places)
            logger.debug("plans = {}", self.plans)
            next_place_id = 0

        if next_place_id == self.state.place_id:
            return False

        place = places_proj.lookup_place(next_place_id)
        if place is None:
            rank_for_place = getattr(places_proj, "rank_for_place", None)
            target_rank = rank_for_place(next_place_id) if rank_for_place is not None else None
            projection_rank = getattr(places_proj, "rank", None)
            if target_rank is None or target_rank == projection_rank:
                logger.debug("Place {} not found.", next_place_id)
                logger.debug("places = {}", self.places)
                return False

            current_place = (
                places_proj.get_place_for_agent(self) if hasattr(places_proj, "get_place_for_agent") else None
            )
            if current_place is not None and hasattr(places_proj, "remove_agent_from_place"):
                places_proj.remove_agent_from_place(self)
            self.place_id = next_place_id
            self.rank_place_id = next_place_id
            self.state.minute_last_moved = minute_of_day
            logger.debug("Rank {}: Agent {} is moving to remote place {}", rank, self.id, self.state.place_id)
            return True

        current_place = places_proj.get_place_for_agent(self) if hasattr(places_proj, "get_place_for_agent") else None
        self.place_id = next_place_id
        self.rank_place_id = next_place_id
        logger.debug("Rank {}: Agent {} is moving to place {}", rank, self.id, self.state.place_id)
        if current_place is None and hasattr(places_proj, "assign_agent_to_place"):
            places_proj.assign_agent_to_place(self, place)
        else:
            places_proj.move_agent_to_place(self, place)
        self.state.minute_last_moved = minute_of_day
        return True

    def _place_id_for_activity(self, activity_id: int | float) -> int:
        """Resolve the projection place id for an activity index."""
        places = self.state.places
        if isinstance(activity_id, int):
            activity_idx = activity_id
        else:
            try:
                activity_idx = int(activity_id)
            except (TypeError, ValueError):
                return 0
        if 0 <= activity_idx < len(places):
            return places[activity_idx]
        return 0

    def count_colocations(self, cspace):
        # subtract self
        num_here = cspace.get_num_agents(self.state.location) - 1
        logger.debug("Agent {} sees {} other agents.", self.id, num_here)
        # meet_log.total_meets += num_here
        # if num_here < meet_log.min_meets:
        #     meet_log.min_meets = num_here
        # if num_here > meet_log.max_meets:
        #     meet_log.max_meets = num_here
        # self.meet_count += num_here

    def make_contacts(self, contacts):
        pass

    def decide_messages(self, model) -> list[MessageIntent]:
        """Return communication intents for the current tick.

        TODO: override in model-specific person subclasses once communication
        behavior is calibrated.
        """
        intents = list(getattr(self, "outbound_message_intents", ()))
        self.outbound_message_intents = []
        return intents

    def queue_message_intent(self, intent: MessageIntent) -> None:
        """Buffer a communication intent for the next communication phase."""
        if "outbound_message_intents" not in self.__dict__:
            self.outbound_message_intents = []
        self.outbound_message_intents.append(intent)

    def receive(self, msg: CommMessage) -> None:
        """Accept a communication-manager message into the inbox."""
        if "inbox" not in self.__dict__:
            self.inbox = []
        self.inbox.append(msg)

    def process_inbox(self, model) -> None:
        """Process and clear the current inbox.

        TODO: replace the placeholder logging with model-specific message
        semantics once higher-level communication behaviors are defined.
        """
        inbox = getattr(self, "inbox", [])
        for msg in inbox:
            logger.debug("Agent {} received routed message {} from {}", self.id, msg.msg_id, msg.sender_uid)

        self.recent_messages = list(inbox)
        self.inbox = []

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        self.behavior_engine.decide(context, cal)
        # self.move(cal, context.get_projection("places"))

    @classmethod
    def restore(cls, person_data: tuple) -> Person:
        """Creates or updates a local person from person_data.

        Args:
            person_data: tuple containing the data returned by Person.save().
        """
        # person_data: Tuple = (
        #     self.uid,
        #     serialize_plans(self.plans),
        #     tuple(e for e in self.location.coordinates),
        #     astuple(self.state)

        # 0: uid is a tuple
        uid = person_data[0]

        plans = restore_plans(person_data[1])

        # pt_array = list(person_data[2])
        # pt = cpt(pt_array[0], pt_array[1], 0)

        # person_data[3] is a PersonData object as tuple
        # the third element of the tuple is the places list
        places = person_data[3][4]

        person = Person(uid[0], uid[2], plans, places, {})

        # restore the state
        person.state = Person.getPersonDataClass()(*person_data[3])
        person.network = person.state.network
        if len(person_data) > 4:
            person.pending_acks = dict(person_data[4])

        return person

    def __str__(self):
        return (
            "Person: "
            f"id={self.id}, "
            f"pt={self.pt}, "
            f"currentPlaceID={self.currentPlaceID}, "
            f"plans={self.plans}, "
            f"state={self.state}"
        )


# 3. Define a PersonConfig NamedTuple
class PersonConfig(NamedTuple):
    name: str
    person_type: type[Person]
    dataType: type[PersonData]
    behaviorEngine: type[BehaviorEngine]


# 4. test code
def test_person():
    """Test the Person class."""
    person_data = {"person_id": 1, "place_id": 0, "activity_id": 0, "places": [0, 1, 2]}

    person_data_class = dataclass(PersonData, frozen=True)

    Person.registerPersonDataClass(person_data_class)

    person = Person(
        local_id=1,
        rank=0,
        plans=[],
        places=[0, 1, 2],
        initDict=person_data,
    )

    logger.debug(person)

    person_data = person.save()
    logger.debug(person_data)

    restored_person = Person.restore(person_data)
    logger.debug(restored_person)

    logger.debug("Person test passed.")


def test_person_serialization(person: Person):
    logger.debug("Testing person serialization.")
    person_data = person.save()
    logger.debug(person_data)

    restored_person = Person.restore(person_data)
    logger.debug(restored_person)


def test_activities(person: Person):
    logger.debug("Testing activities.")
    plans = person.plans
    logger.debug(f"Number of plans: {len(plans)}")

    for plan_idx, plan in enumerate(plans):
        logger.debug(f"Plan {plan_idx}:")
        for element in plan:
            logger.debug(element)


if __name__ == "__main__":
    test_person()


person_cache = {}
person_id_map = {}
