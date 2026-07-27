"""Generate time-resolved interactions from potential social ties.

The social-network input states who may interact. This module determines when
an interaction is possible; it deliberately does not assume that a tie is a
scheduled encounter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

InteractionChannel = Literal["in_person", "remote"]


@dataclass(frozen=True, order=True)
class SocialTie:
    """A canonical undirected potential relationship."""

    person_id_a: int
    person_id_b: int
    network_kind: str
    tie_strength: float | None = None

    def __post_init__(self) -> None:
        if self.person_id_a >= self.person_id_b:
            raise ValueError("SocialTie endpoints must be canonical: person_id_a < person_id_b")
        if not self.network_kind:
            raise ValueError("SocialTie network_kind must not be empty")


@dataclass(frozen=True)
class PresenceInterval:
    """A planned or observed interval during which a person is at a place."""

    person_id: int
    place_id: int
    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        if self.end_minute <= self.start_minute:
            raise ValueError("PresenceInterval end_minute must be after start_minute")


@dataclass(frozen=True)
class InteractionEvent:
    """A possible interaction, derived for a bounded time interval or tick."""

    person_id_a: int
    person_id_b: int
    channel: InteractionChannel
    network_kind: str
    start_minute: int
    end_minute: int
    place_id: int | None = None
    tie_strength: float | None = None


def generate_in_person_events(
    ties: Iterable[SocialTie], presences: Iterable[PresenceInterval]
) -> list[InteractionEvent]:
    """Return events only where tied people overlap at the same place.

    Event end times are exclusive. A relationship never produces an event on
    its own: a non-empty overlap in both time and place is required.
    """
    presences_by_person: dict[int, list[PresenceInterval]] = {}
    for presence in presences:
        presences_by_person.setdefault(presence.person_id, []).append(presence)

    events: list[InteractionEvent] = []
    for tie in ties:
        for left in presences_by_person.get(tie.person_id_a, []):
            for right in presences_by_person.get(tie.person_id_b, []):
                if left.place_id != right.place_id:
                    continue
                start_minute = max(left.start_minute, right.start_minute)
                end_minute = min(left.end_minute, right.end_minute)
                if start_minute >= end_minute:
                    continue
                events.append(
                    InteractionEvent(
                        person_id_a=tie.person_id_a,
                        person_id_b=tie.person_id_b,
                        channel="in_person",
                        network_kind=tie.network_kind,
                        start_minute=start_minute,
                        end_minute=end_minute,
                        place_id=left.place_id,
                        tie_strength=tie.tie_strength,
                    )
                )
    return sorted(events, key=lambda event: (event.start_minute, event.end_minute, event.person_id_a, event.person_id_b))


def generate_remote_message_opportunities(
    ties: Iterable[SocialTie], available_person_ids: Iterable[int], minute: int, duration_minutes: int = 1
) -> list[InteractionEvent]:
    """Return remote opportunities when both endpoints are available.

    Remote opportunities do not require shared place or schedule overlap. A
    behavior or communication layer chooses whether to send a message and its
    content; this function only provides eligible, tie-weighted channels.
    """
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    available = set(available_person_ids)
    return [
        InteractionEvent(
            person_id_a=tie.person_id_a,
            person_id_b=tie.person_id_b,
            channel="remote",
            network_kind=tie.network_kind,
            start_minute=minute,
            end_minute=minute + duration_minutes,
            tie_strength=tie.tie_strength,
        )
        for tie in ties
        if tie.person_id_a in available and tie.person_id_b in available
    ]
