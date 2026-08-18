from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any, TypeAlias


@dataclass(slots=True)
class Act:
    """Names a place that a person will go to at a particular time."""

    person_id: int
    activity_id: int
    activity_sequence: int
    starttime_min: int
    endtime_min: int
    place_id: int

    def contains(self, time: float) -> bool:
        """Return True if the time is within the start and end times of the activity."""
        return self.starttime_min <= time <= self.endtime_min


Activity: TypeAlias = Act


@dataclass(slots=True)
class Leg:
    """Travel between two activities."""

    mode: str
    origin_place_id: int | None = None
    destination_place_id: int | None = None
    origin_node_id: int | None = None
    destination_node_id: int | None = None
    distance_m: float | None = None
    travel_time_min: int | None = None


DEFAULT_TRAVEL_LEG = Leg(mode="travel")


PlanElement: TypeAlias = Activity | Leg
Plan: TypeAlias = list[PlanElement]
Plans: TypeAlias = list[Plan]


@dataclass(frozen=True)
class ActivitySemantics:
    """Lightweight semantic labels for an activity slot."""

    is_home: bool = False
    is_social: bool = False
    is_flexible: bool = False
    is_mandatory: bool = False
    is_travel_sensitive: bool = False


ActivitySemanticsOverrides: TypeAlias = dict[str, list[int] | tuple[int, ...] | set[int]]


def default_activity_semantics(activity_index: int) -> ActivitySemantics:
    """Return coarse default semantics for a plan activity index.

    The current default is intentionally conservative:
    - `0` is treated as home
    - all non-home activities are treated as travel-sensitive and flexible
    - no default activity is treated as social or mandatory without richer
      scenario-specific metadata
    - `is_social` is reserved for explicit scenario mappings rather than
      inferred from the raw activity index
    Future models can replace or extend this mapping with richer domain-
    specific semantics.
    """
    if int(activity_index) == 0:
        return ActivitySemantics(
            is_home=True,
            is_social=False,
            is_flexible=False,
            is_mandatory=False,
            is_travel_sensitive=False,
        )
    return ActivitySemantics(
        is_home=False,
        is_social=False,
        is_flexible=True,
        is_mandatory=False,
        is_travel_sensitive=True,
    )


def activity_semantics_for(
    activity_index: int,
    overrides: ActivitySemanticsOverrides | None = None,
) -> ActivitySemantics:
    """Resolve activity semantics using conservative defaults plus overrides."""
    base = default_activity_semantics(activity_index)
    if not overrides:
        return base

    activity_id = int(activity_index)

    def includes(name: str) -> bool:
        values = overrides.get(name, ())
        return activity_id in {int(value) for value in values}

    return ActivitySemantics(
        is_home=base.is_home or includes("home_ids"),
        is_social=base.is_social or includes("social_ids"),
        is_flexible=base.is_flexible or includes("flexible_ids"),
        is_mandatory=base.is_mandatory or includes("mandatory_ids"),
        is_travel_sensitive=base.is_travel_sensitive or includes("travel_sensitive_ids"),
    )


@dataclass(frozen=True)
class PlanState:
    """Resolved position of a person within a plan at a specific time."""

    element: PlanElement | None
    index: int
    previous_activity: Activity | None = None
    next_activity: Activity | None = None

    @property
    def is_activity(self) -> bool:
        return isinstance(self.element, Act)

    @property
    def is_leg(self) -> bool:
        return isinstance(self.element, Leg)


def make_plan(activities: list[Activity], leg_mode: str = "travel") -> Plan:
    """Create a plan by inserting a leg between each adjacent activity."""
    if not activities:
        return []

    plan: Plan = [activities[0]]
    leg = DEFAULT_TRAVEL_LEG if leg_mode == "travel" else None
    for activity in activities[1:]:
        plan.append(leg if leg is not None else Leg(mode=leg_mode))
        plan.append(activity)
    return plan


def make_routed_plan(
    activities: list[Activity],
    places: list[int],
    road_network: Any,
    leg_mode: str = "drive",
) -> Plan:
    """Create a plan with legs populated from road-network routing."""
    if not activities:
        return []

    plan: Plan = [activities[0]]
    previous_activity = activities[0]

    for next_activity in activities[1:]:
        origin_place_id = resolve_activity_place(previous_activity, places)
        destination_place_id = resolve_activity_place(next_activity, places)

        route = None
        if origin_place_id is not None and destination_place_id is not None:
            route = road_network.route_between_places(origin_place_id, destination_place_id, leg_mode)

        if route is None:
            leg = Leg(
                mode=leg_mode,
                origin_place_id=origin_place_id,
                destination_place_id=destination_place_id,
            )
        else:
            leg = Leg(
                mode=leg_mode,
                origin_place_id=route.origin_place_id,
                destination_place_id=route.destination_place_id,
                origin_node_id=route.origin_node_id,
                destination_node_id=route.destination_node_id,
                distance_m=route.distance_m,
                travel_time_min=route.travel_time_min,
            )

        plan.append(leg)
        plan.append(next_activity)
        previous_activity = next_activity

    return plan


def validate_leg_against_schedule(
    previous_activity: Activity,
    leg: Leg,
    next_activity: Activity,
) -> bool:
    """Return True when routed leg duration fits within the scheduled gap."""
    if leg.travel_time_min is None:
        return True

    gap_min = next_activity.starttime_min - previous_activity.endtime_min
    return leg.travel_time_min <= gap_min


def resolve_activity_place(activity: Activity, places: list[int]) -> int | None:
    """Resolve an event destination, falling back to the legacy anchor vector.

    Schedule imports may provide a distinct ``place_id`` for every activity
    event. Older models omit that value and continue to resolve the place from
    the person's activity-type anchor vector.
    """
    if activity.place_id:
        return int(activity.place_id)
    return _place_id_for_activity(activity.activity_id, places)


def _place_id_for_activity(activity_id: int | float, places: list[int]) -> int | None:
    """Resolve the place id backing an activity index."""
    try:
        activity_idx = int(activity_id)
    except (TypeError, ValueError):
        return None
    if 0 <= activity_idx < len(places):
        return places[activity_idx]
    return None


def activity_at(plan: Plan, time: float) -> Activity | None:
    """Find the activity active at a particular time within a plan."""
    state = resolve_plan_state(plan, time)
    return state.element if isinstance(state.element, Act) else None


def resolve_plan_state(plan: Plan, time: float, start_index: int = 0) -> PlanState:
    """Resolve whether time falls in an activity or a leg between activities."""
    previous_activity: Activity | None = None
    if start_index > 0:
        start_index = min(start_index, len(plan))
        for index in range(start_index - 1, -1, -1):
            element = plan[index]
            if isinstance(element, Act):
                previous_activity = element
                break

    for index in range(start_index, len(plan)):
        element = plan[index]
        if isinstance(element, Act):
            if element.contains(time):
                return PlanState(
                    element=element,
                    index=index,
                    previous_activity=element,
                    next_activity=element,
                )
            if time > element.endtime_min:
                previous_activity = element
            continue

        next_activity = _next_activity(plan, index)
        if previous_activity is None or next_activity is None:
            continue
        if previous_activity.endtime_min < time < next_activity.starttime_min:
            return PlanState(
                element=element,
                index=index,
                previous_activity=previous_activity,
                next_activity=next_activity,
            )

    return PlanState(element=None, index=-1, previous_activity=previous_activity, next_activity=None)


def _next_activity(plan: Plan, start_index: int) -> Activity | None:
    for index in range(start_index + 1, len(plan)):
        element = plan[index]
        if isinstance(element, Act):
            return element
    return None


def serialize_plans(plans: Plans) -> tuple:
    """Serialize plans into tuples for repast agent synchronization."""
    return tuple(serialize_plan(plan) for plan in plans)


def serialize_plan(plan: Plan) -> tuple:
    """Serialize a single plan into tuples."""
    return tuple(serialize_plan_element(element) for element in plan)


def serialize_plan_element(element: PlanElement) -> tuple:
    """Serialize a single plan element."""
    if isinstance(element, Act):
        return ("activity", astuple(element))
    return (
        "leg",
        (
            element.mode,
            element.origin_place_id,
            element.destination_place_id,
            element.origin_node_id,
            element.destination_node_id,
            element.distance_m,
            element.travel_time_min,
        ),
    )


def restore_plans(data: tuple) -> Plans:
    """Restore plans serialized by serialize_plans."""
    return [restore_plan(plan) for plan in data]


def restore_plan(data: tuple) -> Plan:
    """Restore a single plan serialized by serialize_plan."""
    return [restore_plan_element(element) for element in data]


def restore_plan_element(data: tuple) -> PlanElement:
    """Restore a single plan element."""
    kind, payload = data
    if kind == "activity":
        return Act(*payload)
    if kind == "leg":
        return Leg(*payload)
    raise ValueError(f"Unknown plan element kind: {kind}")
