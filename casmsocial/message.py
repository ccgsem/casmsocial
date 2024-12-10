# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 03 Dec 2024

Defining the message interface
"""

from typing import (
    List,
    Dict,
    Tuple
)

from dataclasses import dataclass, field

@dataclass
class Message:
    """
    Message class
    """
    sender: int
    recipients: List[int]
    message: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)
    attachments: Dict = field(default_factory=dict)

    def __init__(
            self,
            sender: Tuple[int, int, int],
            recipient: int,
            message: str,
            timestamp: str,
            metadata: Dict = {},
            attachments: Dict = {}
        ):
        self.sender = sender
        self.recipient = recipient
        self.message = message
        self.timestamp = timestamp
        self.metadata = metadata
        self.attachments = attachments


