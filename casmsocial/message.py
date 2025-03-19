"""
Author: Jon Cline
Created: 03 Dec 2024

Defining the message interface
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """
    Message class
    """

    sender: int
    recipients: list[int]
    message: str
    timestamp: str
    metadata: dict = field(default_factory=dict)
    attachments: dict = field(default_factory=dict)

    def __init__(
        self,
        sender: tuple[int, int, int],
        recipient: int,
        message: str,
        timestamp: str,
        metadata: Optional[dict] = None,
        attachments: Optional[dict] = None,
    ):
        if attachments is None:
            attachments = {}
        if metadata is None:
            metadata = {}
        self.sender = sender
        self.recipient = recipient
        self.message = message
        self.timestamp = timestamp
        self.metadata = metadata
        self.attachments = attachments
