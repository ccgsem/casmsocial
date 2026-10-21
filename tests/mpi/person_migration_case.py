"""Migrate Person agents between two MPI ranks and report what survives."""

import json

import repast4py.context as ctx
from mpi4py import MPI

from casmsocial.activities import Act, make_plan
from casmsocial.communication.types import CommMessage, MessageIntent, MessageKind, build_message_payload
from casmsocial.person import BehaviorEngineV2, LLMBehaviorEngine, Person, ScheduleBehaviorEngine

PLACES = (100, 200)


def make_person(local_id, engine):
    plan = make_plan([Act(local_id, 0, 0, 0, 60, 100), Act(local_id, 1, 1, 120, 180, 200)])
    return Person.from_default_fields(local_id, 0, [plan], PLACES, behavior_engine=engine)


def message(receiver, msg_id):
    return CommMessage(
        msg_id=msg_id,
        sender_uid=(99, Person.TYPE, 0),
        sender_place_id=100,
        receiver_uid=receiver.uid,
        receiver_place_id=200,
        mode="local",
        payload=build_message_payload(MessageKind.WARNING, topic="heat", urgency=0.9),
        tick=7,
    )


def seed_cognitive_state(person):
    cognition = person.behavior_engine.cognition
    cognition.beliefs["routine_preference"] = 0.123
    cognition.beliefs["persona_prior"] = 0.456
    cognition.goals["stay_home"] = 0.9
    cognition.memory["last_hour_of_day"] = 13.0
    cognition.episodic_memory.append({"tick": 6, "event_type": "llm_proposal", "summary": "stay in", "data": {}})
    cognition.stress = 0.42
    cognition.last_llm_tick = 6
    cognition.last_llm_summary = "persona says stay in"
    person.receive(message(person, "inbox-1"))
    person.recent_messages = [message(person, "recent-1")]
    person.queue_message_intent(
        MessageIntent(
            sender_uid=person.uid,
            receiver_uid=(5, Person.TYPE, 0),
            receiver_place_id=200,
            mode="local",
            payload=build_message_payload(MessageKind.CHECK_IN),
        )
    )
    person.pending_acks["ack-1"] = 3


def describe(person):
    engine = person.behavior_engine
    record = {
        "engine": type(engine).__name__,
        "inbox": [msg.msg_id for msg in person.inbox],
        "recent_messages": [msg.msg_id for msg in person.recent_messages],
        "outbound_message_intents": [intent.receiver_uid for intent in person.outbound_message_intents],
        "pending_acks": person.pending_acks,
    }
    cognition = getattr(engine, "cognition", None)
    if cognition is not None:
        record.update(
            beliefs=cognition.beliefs,
            goals=cognition.goals,
            memory=cognition.memory,
            episodic_memory=cognition.episodic_memory,
            stress=cognition.stress,
            last_llm_tick=cognition.last_llm_tick,
            last_llm_summary=cognition.last_llm_summary,
        )
    return record


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == 2

    # The registered default is V2; agent 2 uses a different engine type than
    # the default, which restore() must not silently replace.
    Person.registerBehaviorEngine(BehaviorEngineV2)
    context = ctx.SharedContext(comm)
    people = {}
    if rank == 0:
        people = {
            "v2": make_person(1, BehaviorEngineV2),
            "llm": make_person(2, LLMBehaviorEngine),
            "schedule": make_person(3, ScheduleBehaviorEngine),
        }
        seed_cognitive_state(people["v2"])
        seed_cognitive_state(people["llm"])
        for person in people.values():
            context.add(person)

    sent = {name: describe(person) for name, person in people.items()}
    context.move_agents([(person.uid, 1) for person in people.values()] if rank == 0 else [], Person.restore)

    received = {}
    if rank == 1:
        by_id = {agent.id: agent for agent in context.agents()}
        received = {name: describe(by_id[local_id]) for name, local_id in (("v2", 1), ("llm", 2), ("schedule", 3))}

    gathered = comm.gather({"sent": sent, "received": received}, root=0)
    if rank == 0:
        print(json.dumps({"sent": gathered[0]["sent"], "received": gathered[1]["received"]}, default=list))


if __name__ == "__main__":
    main()
