"""Typed message payloads for distributed agent communication."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

AgentUID = tuple[int, int, int]
MessageMode = str


class MessageKind(StrEnum):
    """Base catalog of communication intent categories."""

    STATUS_UPDATE = "status_update"
    INVITATION = "invitation"
    WARNING = "warning"
    REQUEST_HELP = "request_help"
    COORDINATION = "coordination"
    ACKNOWLEDGMENT = "acknowledgment"
    CHECK_IN = "check_in"
    RECOMMENDATION = "recommendation"
    REMINDER = "reminder"
    ANNOUNCEMENT = "announcement"


@dataclass(slots=True)
class MessagePayload:
    """Structured communication payload shared by intents and transport messages."""

    kind: MessageKind
    topic: str = ""
    text: str = ""
    urgency: float = 0.0
    trust_weight: float = 0.5
    requested_place_id: int | None = None
    requested_start_min: int | None = None
    expires_tick: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_message_payload(
    kind: MessageKind,
    *,
    topic: str = "",
    text: str = "",
    urgency: float = 0.0,
    trust_weight: float = 0.5,
    requested_place_id: int | None = None,
    requested_start_min: int | None = None,
    expires_tick: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized message payload dictionary."""
    payload = MessagePayload(
        kind=kind,
        topic=topic,
        text=text,
        urgency=urgency,
        trust_weight=trust_weight,
        requested_place_id=requested_place_id,
        requested_start_min=requested_start_min,
        expires_tick=expires_tick,
        metadata={} if metadata is None else dict(metadata),
    )
    return asdict(payload)


@dataclass(slots=True)
class MessageIntent:
    """Agent-authored intent that describes a desired communication action."""

    sender_uid: AgentUID
    receiver_uid: AgentUID
    receiver_place_id: int
    mode: MessageMode
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommMessage:
    """Transport packet exchanged by the communication manager."""

    msg_id: str
    sender_uid: AgentUID
    sender_place_id: int
    receiver_uid: AgentUID
    receiver_place_id: int
    mode: MessageMode
    payload: dict[str, Any] = field(default_factory=dict)
    tick: int = 0


@dataclass(slots=True)
class AckMessage:
    """Receipt for a remote two-way message."""

    msg_id: str
    original_sender_uid: AgentUID
    receiver_uid: AgentUID
    receiver_place_id: int
    tick: int
    status: str = "received"
