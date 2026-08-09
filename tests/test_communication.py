from __future__ import annotations

from collections import namedtuple

from casmsocial.communication import CommunicationManager, MessageIntent
from casmsocial.communication.types import MessageKind, build_message_payload
from casmsocial.person import Person


class FakeComm:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def alltoall(self, send_buffers):
        return [[] for _ in send_buffers]


class FakeModel:
    def __init__(self, rank: int, place_to_rank: dict[int, int], people: list[Person]):
        self.rank = rank
        self.place_to_rank = place_to_rank
        self.person_uid_map = {person.uid: person for person in people}
        self.place_members = {}
        for person in people:
            self.place_members.setdefault(person.place_id, []).append(person.uid)

    def get_person_by_uid(self, uid):
        return self.person_uid_map.get(uid)


def build_person(local_id: int, rank: int, place_id: int) -> Person:
    places_type = namedtuple("TestPlaces", ["home"])
    places = places_type(place_id)
    return Person(local_id, rank, [], places, {"sp_id": local_id})


def test_local_same_place_delivery():
    sender = build_person(local_id=1, rank=0, place_id=10)
    receiver = build_person(local_id=2, rank=0, place_id=10)
    model = FakeModel(rank=0, place_to_rank={10: 0}, people=[sender, receiver])
    manager = CommunicationManager(FakeComm(rank=0, size=1))

    intent = MessageIntent(
        sender_uid=sender.uid,
        receiver_uid=receiver.uid,
        receiver_place_id=receiver.place_id,
        mode="local",
        payload=build_message_payload(MessageKind.CHECK_IN, topic="local"),
    )

    manager.route([intent], model, tick=1)

    assert len(receiver.inbox) == 1
    assert receiver.inbox[0].payload == build_message_payload(MessageKind.CHECK_IN, topic="local")


def test_remote_one_way_delivery():
    sender = build_person(local_id=1, rank=0, place_id=10)
    receiver = build_person(local_id=2, rank=1, place_id=20)
    model0 = FakeModel(rank=0, place_to_rank={10: 0, 20: 1}, people=[sender])
    model1 = FakeModel(rank=1, place_to_rank={10: 0, 20: 1}, people=[receiver])
    manager0 = CommunicationManager(FakeComm(rank=0, size=2))
    manager1 = CommunicationManager(FakeComm(rank=1, size=2))

    intent = MessageIntent(
        sender_uid=sender.uid,
        receiver_uid=receiver.uid,
        receiver_place_id=receiver.place_id,
        mode="one_way",
        payload=build_message_payload(MessageKind.STATUS_UPDATE, topic="remote"),
    )

    manager0.route([intent], model0, tick=1)
    forwarded = list(manager0.remote_message_buffers[1])
    manager1.consume_remote_messages(forwarded, model1)

    assert len(receiver.inbox) == 1
    assert receiver.inbox[0].payload == build_message_payload(MessageKind.STATUS_UPDATE, topic="remote")
    assert manager1.remote_ack_buffers[0] == []


def test_remote_two_way_delivery_with_ack_receipt():
    sender = build_person(local_id=1, rank=0, place_id=10)
    receiver = build_person(local_id=2, rank=1, place_id=20)
    model0 = FakeModel(rank=0, place_to_rank={10: 0, 20: 1}, people=[sender])
    model1 = FakeModel(rank=1, place_to_rank={10: 0, 20: 1}, people=[receiver])
    manager0 = CommunicationManager(FakeComm(rank=0, size=2))
    manager1 = CommunicationManager(FakeComm(rank=1, size=2))

    intent = MessageIntent(
        sender_uid=sender.uid,
        receiver_uid=receiver.uid,
        receiver_place_id=receiver.place_id,
        mode="two_way",
        payload=build_message_payload(MessageKind.ACKNOWLEDGMENT, topic="remote_ack"),
    )

    manager0.route([intent], model0, tick=7)
    forwarded = list(manager0.remote_message_buffers[1])
    delivered = manager1.consume_remote_messages(forwarded, model1)
    manager0.consume_acks(delivered, model0)

    ack_state = next(iter(sender.pending_acks.values()))

    assert len(receiver.inbox) == 1
    assert ack_state["status"] == "received"
    assert ack_state["ack_tick"] == 7


def test_local_delivery_to_receiver_on_leg_uses_rank_place():
    sender = build_person(local_id=1, rank=0, place_id=10)
    receiver = build_person(local_id=2, rank=0, place_id=10)
    receiver.place_id = 0
    receiver.rank_place_id = 10

    model = FakeModel(rank=0, place_to_rank={10: 0}, people=[sender, receiver])
    manager = CommunicationManager(FakeComm(rank=0, size=1))

    intent = MessageIntent(
        sender_uid=sender.uid,
        receiver_uid=receiver.uid,
        receiver_place_id=receiver.place_id,
        mode="local",
        payload=build_message_payload(MessageKind.CHECK_IN, topic="leg_local"),
    )

    manager.route([intent], model, tick=3)

    assert len(receiver.inbox) == 1
    assert receiver.inbox[0].payload == build_message_payload(MessageKind.CHECK_IN, topic="leg_local")
    assert receiver.inbox[0].sender_place_id == 10


def test_remote_delivery_to_receiver_on_leg_uses_receiver_uid_rank():
    sender = build_person(local_id=1, rank=0, place_id=10)
    receiver = build_person(local_id=2, rank=1, place_id=20)
    receiver.place_id = 0
    receiver.rank_place_id = 20

    model0 = FakeModel(rank=0, place_to_rank={10: 0, 20: 1}, people=[sender])
    model1 = FakeModel(rank=1, place_to_rank={10: 0, 20: 1}, people=[receiver])
    manager0 = CommunicationManager(FakeComm(rank=0, size=2))
    manager1 = CommunicationManager(FakeComm(rank=1, size=2))

    intent = MessageIntent(
        sender_uid=sender.uid,
        receiver_uid=receiver.uid,
        receiver_place_id=receiver.place_id,
        mode="one_way",
        payload=build_message_payload(MessageKind.STATUS_UPDATE, topic="leg_remote"),
    )

    manager0.route([intent], model0, tick=5)
    forwarded = list(manager0.remote_message_buffers[1])
    manager1.consume_remote_messages(forwarded, model1)

    assert len(receiver.inbox) == 1
    assert receiver.inbox[0].payload == build_message_payload(MessageKind.STATUS_UPDATE, topic="leg_remote")
