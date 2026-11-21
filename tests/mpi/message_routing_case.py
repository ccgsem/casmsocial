"""Two ranks exchange messages with people who live away from the rank they were created on.

Each rank holds a sender and one person created on the other rank (as after migration: a uid keeps its creation rank).
Every sender messages both people, one on its own rank and one across. All four messages must arrive.
"""

import json

from mpi4py import MPI

from casmsocial.communication import CommunicationManager, MessageIntent
from casmsocial.communication.types import MessageKind, build_message_payload
from casmsocial.person import Person

PLACE_TO_RANK = {100: 0, 200: 1}
HOME = {0: 100, 1: 200}


class Model:
    def __init__(self, rank, people):
        self.rank = rank
        self.place_to_rank = PLACE_TO_RANK
        self.person_uid_map = {person.uid: person for person in people}

    def get_person_by_uid(self, uid):
        return self.person_uid_map.get(uid)


def person(local_id, created_on, place_id):
    return Person.from_default_fields(local_id, created_on, [], (place_id,))


def main():
    comm = MPI.COMM_WORLD
    rank, other = comm.Get_rank(), 1 - comm.Get_rank()
    sender = person(10 + rank, rank, HOME[rank])
    moved_here = person(20 + rank, other, HOME[rank])  # created on the other rank, lives here
    model = Model(rank, [sender, moved_here])
    manager = CommunicationManager(comm)

    moved_there_uid = (20 + other, Person.TYPE, rank)  # created here, lives on the other rank
    intents = [
        MessageIntent(
            sender.uid, moved_here.uid, HOME[rank], "one_way", build_message_payload(MessageKind.WARNING, topic="local")
        ),
        MessageIntent(
            sender.uid,
            moved_there_uid,
            HOME[other],
            "two_way",
            build_message_payload(MessageKind.WARNING, topic="across"),
        ),
    ]
    manager.route(intents, model, tick=1)
    manager.exchange_remote(model)
    manager.exchange_acks(model)

    record = {
        "rank": rank,
        "inbox": sorted(m.payload["topic"] for m in moved_here.inbox),
        "acks": sorted(str(state.get("status")) for state in sender.pending_acks.values()),
    }
    gathered = comm.gather(record, root=0)
    if rank == 0:
        print(json.dumps(gathered))


if __name__ == "__main__":
    main()
