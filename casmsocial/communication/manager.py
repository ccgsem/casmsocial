"""MPI-backed communication routing for place-based agent messaging."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from uuid import uuid4

from loguru import logger

from casmsocial.communication.types import AckMessage, CommMessage, MessageIntent


class CommunicationManager:
    """Routes local and remote messages while keeping MPI concerns centralized."""

    def __init__(self, comm) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.clear_buffers()

    def clear_buffers(self) -> None:
        """Reset rank-addressed outgoing buffers for the next communication phase."""
        self.remote_message_buffers: dict[int, list[CommMessage]] = {rank: [] for rank in range(self.size)}
        self.remote_ack_buffers: dict[int, list[AckMessage]] = {rank: [] for rank in range(self.size)}

    def route(self, intents: Iterable[MessageIntent], model, tick: int) -> list[CommMessage]:
        """Convert intents into local deliveries or remote buffers."""
        delivered_local: list[CommMessage] = []

        for intent in intents:
            sender = model.get_person_by_uid(intent.sender_uid)
            if sender is None:
                logger.debug(f"Skipping message intent for missing sender {intent.sender_uid}")
                continue

            sender_place_id = getattr(sender, "rank_place_id", sender.place_id)
            message = self._create_message(intent=intent, sender_place_id=sender_place_id, tick=tick)
            destination_rank = self._rank_for_receiver(message)

            if self._is_local_delivery(message, destination_rank, sender_place_id, model):
                if self.deliver_local(message, model):
                    delivered_local.append(message)
                continue

            self.queue_remote(message, destination_rank, sender)

        return delivered_local

    def deliver_local(self, message: CommMessage, model) -> bool:
        """Deliver a same-place message to a local receiver."""
        receiver = model.get_person_by_uid(message.receiver_uid)
        if receiver is None:
            logger.debug(f"Local receiver {message.receiver_uid} not present on rank {self.rank}")
            return False

        receiver.receive(message)
        return True

    def queue_remote(self, message: CommMessage, destination_rank: int, sender) -> None:
        """Buffer a remote message for MPI exchange."""
        self.remote_message_buffers[destination_rank].append(message)

        if message.mode == "two_way":
            sender.pending_acks[message.msg_id] = {
                "receiver_uid": message.receiver_uid,
                "tick_sent": message.tick,
                "payload_summary": dict(message.payload),
                "status": "pending",
            }

    def exchange_remote(self, model) -> list[CommMessage]:
        """Exchange remote message buffers across ranks and deliver received messages."""
        received = self._alltoall(self.remote_message_buffers)
        flattened = [message for rank_messages in received for message in rank_messages]
        self.consume_remote_messages(flattened, model)
        return flattened

    def consume_remote_messages(self, messages: Iterable[CommMessage], model) -> list[AckMessage]:
        """Deliver remote messages received from other ranks and queue acknowledgments."""
        generated_acks: list[AckMessage] = []

        for message in messages:
            receiver = model.get_person_by_uid(message.receiver_uid)
            if receiver is None:
                logger.debug(f"Remote receiver {message.receiver_uid} not present on rank {self.rank}")
                continue

            receiver.receive(message)

            if message.mode == "two_way":
                ack = self.generate_ack(message)
                generated_acks.append(ack)
                self.queue_ack(ack)

        return generated_acks

    def generate_ack(self, message: CommMessage) -> AckMessage:
        """Create a lightweight acknowledgment for a two-way remote message."""
        return AckMessage(
            msg_id=message.msg_id,
            original_sender_uid=message.sender_uid,
            receiver_uid=message.receiver_uid,
            receiver_place_id=message.receiver_place_id,
            tick=message.tick,
        )

    def queue_ack(self, ack: AckMessage) -> None:
        """Buffer an acknowledgment for delivery back to the sender's owning rank."""
        destination_rank = ack.original_sender_uid[2]
        self.remote_ack_buffers[destination_rank].append(ack)

    def exchange_acks(self, model) -> list[AckMessage]:
        """Exchange remote acknowledgment buffers and mark sender state."""
        received = self._alltoall(self.remote_ack_buffers)
        flattened = [ack for rank_acks in received for ack in rank_acks]
        self.consume_acks(flattened, model)
        return flattened

    def consume_acks(self, acknowledgments: Iterable[AckMessage], model) -> None:
        """Apply received acknowledgments to sender-side pending ack state."""
        for ack in acknowledgments:
            sender = model.get_person_by_uid(ack.original_sender_uid)
            if sender is None:
                logger.debug(f"Ack sender {ack.original_sender_uid} not present on rank {self.rank}")
                continue

            pending = sender.pending_acks.get(ack.msg_id)
            if pending is None:
                logger.debug(f"Ack for unknown message {ack.msg_id} on rank {self.rank}")
                continue

            pending["status"] = ack.status
            pending["ack_tick"] = ack.tick
            pending["receiver_uid"] = ack.receiver_uid

    def _alltoall(self, rank_buffers: dict[int, list]) -> list[list]:
        """Run an MPI all-to-all exchange using per-rank list buffers."""
        send_buffers = [list(rank_buffers.get(rank, [])) for rank in range(self.size)]
        return self.comm.alltoall(send_buffers)

    def _create_message(self, intent: MessageIntent, sender_place_id: int, tick: int) -> CommMessage:
        """Promote an intent into a transport message."""
        return CommMessage(
            msg_id=str(uuid4()),
            sender_uid=intent.sender_uid,
            sender_place_id=sender_place_id,
            receiver_uid=intent.receiver_uid,
            receiver_place_id=intent.receiver_place_id,
            mode=intent.mode,
            payload=dict(intent.payload),
            tick=tick,
        )

    def _is_local_delivery(self, message: CommMessage, destination_rank: int, sender_place_id: int, model) -> bool:
        """Local delivery is limited to same-rank, same-place communication."""
        if destination_rank != self.rank or message.mode != "local":
            return False

        receiver = model.get_person_by_uid(message.receiver_uid)
        if receiver is None:
            return False

        receiver_place_id = getattr(receiver, "rank_place_id", receiver.place_id)
        return receiver_place_id == sender_place_id

    def _rank_for_receiver(self, message: CommMessage) -> int:
        """Resolve the owner rank for a receiver from its current uid."""
        return message.receiver_uid[2]


def message_to_dict(message: CommMessage) -> dict:
    """Serialize a communication message for logging or debugging."""
    return asdict(message)
