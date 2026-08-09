"""Minimal multi-rank communication demo.

Run with:
    mpiexec -n 2 python examples/communication_mpi_demo.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import namedtuple
from dataclasses import dataclass

from mpi4py import MPI

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casmsocial.communication import CommunicationManager, MessageIntent  # noqa: E402
from casmsocial.person import Person  # noqa: E402


class DemoPerson(Person):
    """Simple person that emits one deterministic cross-rank message."""

    def decide_messages(self, model) -> list[MessageIntent]:
        if model.rank == 0:
            return [
                MessageIntent(
                    sender_uid=self.uid,
                    receiver_uid=model.remote_person_uid,
                    receiver_place_id=model.remote_place_id,
                    mode="one_way",
                    payload={"kind": "greeting", "text": "hello from rank 0"},
                )
            ]

        return [
            MessageIntent(
                sender_uid=self.uid,
                receiver_uid=model.remote_person_uid,
                receiver_place_id=model.remote_place_id,
                mode="two_way",
                payload={"kind": "ping", "text": "ack me from rank 1"},
            )
        ]


@dataclass
class DemoModel:
    comm: MPI.Comm
    rank: int
    place_to_rank: dict[int, int]
    local_person: DemoPerson
    remote_person_uid: tuple[int, int, int]
    remote_place_id: int

    def __post_init__(self) -> None:
        self.communication_manager = CommunicationManager(self.comm)
        self.person_uid_map = {self.local_person.uid: self.local_person}
        self.place_members = {self.local_person.place_id: [self.local_person.uid]}

    def get_person_by_uid(self, uid):
        return self.person_uid_map.get(uid)

    def collect_message_intents(self) -> list[MessageIntent]:
        return self.local_person.decide_messages(self)


def build_person(person_id: int, rank: int, place_id: int) -> DemoPerson:
    places_type = namedtuple("DemoPlaces", ["home"])
    places = places_type(place_id)
    return DemoPerson(person_id, rank, [], places, {"sp_id": person_id})


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_place_id = 100 + rank
    remote_place_id = 100 + (1 - rank)

    local_person = build_person(person_id=rank + 1, rank=rank, place_id=local_place_id)
    remote_person_uid = (2 if rank == 0 else 1, DemoPerson.TYPE, 1 - rank)

    model = DemoModel(
        comm=comm,
        rank=rank,
        place_to_rank={100: 0, 101: 1},
        local_person=local_person,
        remote_person_uid=remote_person_uid,
        remote_place_id=remote_place_id,
    )

    manager = model.communication_manager
    manager.route(model.collect_message_intents(), model, tick=1)
    manager.exchange_remote(model)
    manager.exchange_acks(model)

    print(
        {
            "rank": rank,
            "inbox": [message.payload for message in local_person.inbox],
            "pending_acks": local_person.pending_acks,
        }
    )


if __name__ == "__main__":
    main()
