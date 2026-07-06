"""Communication architecture primitives for place-based agent messaging."""

from casmsocial.communication.manager import CommunicationManager
from casmsocial.communication.types import AckMessage, CommMessage, MessageIntent

__all__ = [
    "AckMessage",
    "CommMessage",
    "CommunicationManager",
    "MessageIntent",
]
