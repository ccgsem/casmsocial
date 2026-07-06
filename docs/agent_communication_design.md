# Agent Communication Design for repast4py

## Purpose

This document defines a preferred target architecture for implementing agent communication in a distributed repast4py model with the following characteristics:

- `Person` agents move between `Place` agents at each simulation tick.
- Each `Place` is assigned to a fixed MPI rank for the duration of the simulation.
- A `Person` should be owned by the rank that owns its current `Place`.
- Communication supports three modes:
  1. **Local**: sender and receiver are in the same `Place`
  2. **One-way remote**: sender and receiver are in different `Place`s; message is delivered without confirmation
  3. **Two-way remote**: sender and receiver are in different `Place`s; receiver sends an acknowledgment confirming receipt

The goal is to provide a clean implementation path in Codex that preserves separation of concerns and fits naturally into the existing `Person` / `Place` model.

---

## Preferred Target Architecture

### Core design principle

Separate the model into four responsibilities:

- **`Person`** decides *what* to communicate and processes received communications
- **`Place`** organizes occupancy and defines co-location
- **`CommunicationManager`** handles routing, buffering, inter-rank transport, and acknowledgments
- **`Model`** orchestrates movement, synchronization, communication phases, and indexing

This keeps MPI and routing concerns out of the agent classes while still allowing communication to remain agent-driven.

---

## High-Level Component Model

```text
Model
 ├── SharedContext
 ├── place_to_rank: dict[place_id, rank]
 ├── place_members: dict[place_id, list[person_uid]]
 ├── CommunicationManager
 ├── Person agents
 └── Place agents

Person
 ├── place_id
 ├── inbox
 ├── pending_acks
 ├── decide_messages(model) -> list[MessageIntent]
 └── process_inbox(model)

Place
 ├── place_id
 ├── occupants (optional cached set / list)
 ├── add_occupant(person_uid)
 └── remove_occupant(person_uid)

CommunicationManager
 ├── route(intents)
 ├── deliver_local(...)
 ├── queue_remote(...)
 ├── exchange_remote()
 ├── generate_acks()
 ├── exchange_acks()
 └── clear_buffers()
```

---

## Why this architecture

This design is preferred because it:

- keeps `Person` behavior-focused
- avoids turning `Place` into a full distributed message broker
- centralizes MPI communication in one component
- matches the place-based partitioning of the simulation
- makes local communication cheap
- makes one-way and two-way remote communication explicit and testable
- supports incremental implementation

---

## Required Model Assumptions

The implementation should assume the following:

1. **Fixed place ownership**
   - `place_to_rank[place_id]` is established at initialization and does not change.

2. **Person ownership follows place**
   - After movement, a `Person` is owned by the rank that owns its current `Place`.

3. **Movement occurs before communication**
   - Communication mode is determined using the updated place assignments for the current tick.

4. **Communication is point-to-point**
   - At least initially, communication should target specific receivers rather than general broadcast.

5. **Acknowledgments are lightweight**
   - Acknowledgments confirm receipt only; they do not initially carry semantic reply content.

---

## Data Structures

## `MessageIntent`

Produced by `Person.decide_messages(...)`.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class MessageIntent:
    sender_uid: tuple
    receiver_uid: tuple
    receiver_place_id: int
    mode: str          # "local", "one_way", "two_way"
    payload: dict[str, Any]
```

This is an intent, not a transport packet.

---

## `CommMessage`

Routed and delivered by `CommunicationManager`.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class CommMessage:
    msg_id: str
    sender_uid: tuple
    sender_place_id: int
    receiver_uid: tuple
    receiver_place_id: int
    mode: str          # "local", "one_way", "two_way"
    payload: dict[str, Any]
    tick: int
```

---

## `AckMessage`

Used only for two-way remote communication.

```python
from dataclasses import dataclass

@dataclass
class AckMessage:
    msg_id: str
    original_sender_uid: tuple
    receiver_uid: tuple
    receiver_place_id: int
    tick: int
    status: str        # e.g. "received"
```

---

## Agent Responsibilities

## `Person`

### Required fields

```python
class Person(...):
    self.place_id: int
    self.inbox: list[CommMessage]
    self.pending_acks: dict[str, dict]
```

### Required methods

```python
def decide_messages(self, model) -> list[MessageIntent]:
    ...
```

This should:
- inspect the model state
- choose receivers
- choose mode
- emit intents only

It should **not** perform MPI communication.

```python
def receive(self, msg: CommMessage) -> None:
    self.inbox.append(msg)
```

```python
def process_inbox(self, model) -> None:
    ...
```

This should:
- process received messages
- update internal state
- optionally emit secondary behavioral effects
- not directly handle remote ack transport

### Ack tracking

For `two_way` messages, the sender should retain minimal metadata:

```python
self.pending_acks[msg_id] = {
    "receiver_uid": receiver_uid,
    "tick_sent": current_tick,
    "payload_summary": ...
}
```

This supports later timeouts or retries if desired.

---

## `Place`

### Required fields

```python
class Place(...):
    self.place_id: int
```

### Optional fields

```python
self.occupants: set[tuple] | list[tuple]
```

This can be maintained locally if useful, but a model-level `place_members` index is still recommended as the primary lookup structure.

### Role of `Place`

`Place` should remain lightweight. It should not own inter-rank communication logic.

It may optionally support local occupancy bookkeeping:

```python
def add_occupant(self, person_uid): ...
def remove_occupant(self, person_uid): ...
```

---

## CommunicationManager Responsibilities

`CommunicationManager` is the core integration component.

### It should own:

- local delivery
- remote message buffering by destination rank
- MPI exchange of remote messages
- remote acknowledgment buffering by destination rank
- MPI exchange of acknowledgments
- optional logging / counters / diagnostics

### It should not own:

- movement decisions
- communication content semantics
- person behavioral logic

---

## Proposed `CommunicationManager` interface

```python
class CommunicationManager:

    def __init__(self, model):
        self.model = model
        self.outgoing_msgs_by_rank = ...
        self.outgoing_acks_by_rank = ...

    def route(self, intents: list[MessageIntent], tick: int) -> None:
        ...

    def deliver_local(self, msg: CommMessage) -> None:
        ...

    def queue_remote(self, msg: CommMessage) -> None:
        ...

    def exchange_remote(self) -> None:
        ...

    def generate_acks(self, tick: int) -> None:
        ...

    def exchange_acks(self) -> None:
        ...

    def clear_buffers(self) -> None:
        ...
```

---

## Routing Rules

These routing rules should be implemented exactly.

### Rule 1: Local

If:

```python
sender.place_id == receiver.place_id
```

then:

- deliver directly to receiver inbox on the same rank
- do not use MPI
- do not create an acknowledgment unless local ack behavior is explicitly added later

### Rule 2: One-way remote

If sender and receiver are in different places:

- create a `CommMessage`
- determine destination rank using `place_to_rank[receiver_place_id]`
- add to `outgoing_msgs_by_rank[dest_rank]`
- exchange with MPI
- deliver to receiver inbox on destination rank
- no acknowledgment required

### Rule 3: Two-way remote

If sender and receiver are in different places and mode is `two_way`:

- same as one-way remote delivery
- sender records `msg_id` in `pending_acks`
- destination rank generates an `AckMessage` after delivery / receipt phase
- acknowledgment is sent back to sender rank
- sender removes `msg_id` from `pending_acks` when ack arrives

---

## Model-Level Indexes

The model should maintain a refreshed place-membership directory after movement and synchronization.

### Required index

```python
place_members: dict[int, list[tuple]]
```

Where values are `Person.uid` values or local agent references.

This index should be rebuilt every tick after movement.

### Optional global directory

For remote receiver selection, the model may also build a globally visible directory using `MPI.allgather`:

```python
global_place_members: dict[int, list[tuple]]
```

This is useful if agents need to choose receivers in remote places without ghosting all remote agents.

Recommendation: start with a global directory of `uid`s only, not full agent state.

---

## Tick Lifecycle

Implement communication using the following order of operations each tick.

```text
1. movement phase
2. rank migration for moved persons
3. synchronize shared context
4. rebuild place membership index
5. collect communication intents from persons
6. route and deliver local messages
7. exchange remote messages
8. deliver received remote messages
9. generate acknowledgments for two-way remote messages
10. exchange acknowledgments
11. process inboxes and ack receipts
12. clear temporary buffers
```

### Important constraint

Movement must happen **before** message classification into local vs remote.

---

## Proposed Model Orchestration

```python
def step(self):
    self.move_people()
    self.context.synchronize(restore_agent)

    self.rebuild_place_members()

    intents = []
    for person in self.people():
        intents.extend(person.decide_messages(self))

    self.comm_manager.route(intents, tick=self.current_tick)
    self.comm_manager.exchange_remote()
    self.comm_manager.generate_acks(tick=self.current_tick)
    self.comm_manager.exchange_acks()

    for person in self.people():
        person.process_inbox(self)

    self.comm_manager.clear_buffers()
```

This orchestration is the target design.

---

## repast4py Integration Notes

### Use `SharedContext`

The distributed model should be based on `repast4py.context.SharedContext`.

### Use rank migration for moved persons

When a `Person` changes places and the destination place is owned by another rank, that `Person` should migrate to the destination rank.

Codex should integrate with the existing movement mechanism already present in the codebase. If rank migration is not yet implemented, add it as part of this work.

### Use `restore_agent(...)`

If synchronization or movement requires restoration of agents from serialized state, define and use a `restore_agent` function compatible with both `Person` and `Place`.

### Do not use network projection as the transport layer

A repast4py `SharedNetwork` may later be useful to represent selective communication ties, but it should **not** be the primary transport mechanism for this first implementation.

The current implementation should use explicit buffering + MPI exchange for remote communications.

---

## Incremental Implementation Plan for Codex

Codex should implement in stages.

### Stage 1: Skeleton and data structures

Implement:

- `MessageIntent`
- `CommMessage`
- `AckMessage`
- `CommunicationManager` skeleton
- `Person` inbox / pending ack fields
- model-level `place_members`

### Stage 2: Local communication

Implement:

- `Person.decide_messages(...)`
- local same-place delivery
- `Person.process_inbox(...)`

Validate:
- same-place messages are delivered without MPI

### Stage 3: One-way remote communication

Implement:

- buffering by destination rank
- MPI message exchange
- remote inbox delivery

Validate:
- different-place messages are delivered across ranks
- no acks generated

### Stage 4: Two-way remote communication

Implement:

- sender-side `pending_acks`
- ack generation on receiver rank
- MPI exchange of acks
- sender-side ack completion

Validate:
- sender sees ack receipt
- pending ack entry is removed

### Stage 5: Logging and diagnostics

Add optional counters:

- local messages sent
- remote one-way messages sent
- remote two-way messages sent
- acknowledgments generated
- acknowledgments received
- unresolved pending acknowledgments

### Stage 6: Tests / example scenario

Add a minimal multi-rank example using `mpiexec`:

- at least 2 places on different ranks
- at least 1 moving person
- examples of all three communication modes

---

## Suggested File Layout

Codex may adapt to the existing repo, but a reasonable target structure is:

```text
project/
├── model/
│   ├── agents/
│   │   ├── person.py
│   │   └── place.py
│   ├── communication/
│   │   ├── messages.py
│   │   ├── manager.py
│   │   └── protocol.py
│   ├── model.py
│   └── restore.py
├── tests/
│   ├── test_local_communication.py
│   ├── test_remote_one_way.py
│   └── test_remote_two_way.py
└── docs/
    └── agent_communication_design.md
```

If the existing repo already has an organized module structure, preserve that instead.

---

## Suggested Codex Prompt

Use something close to the following in Codex:

```text
Implement the communication architecture described in docs/agent_communication_design.md.

Requirements:
- preserve existing Person and Place implementations where possible
- add a CommunicationManager for routing and MPI exchange
- support local, one-way remote, and two-way remote communication
- rebuild place membership after movement each tick
- ensure Person ownership follows the rank of the current Place
- keep communication content decisions on Person and routing on CommunicationManager
- add a minimal runnable example and basic tests
```

---

## Non-Goals for First Implementation

Do not include these in the first version unless they are already trivial in the existing codebase:

- retries / retransmission
- message ordering guarantees beyond tick-level processing
- complex response payloads for acknowledgments
- broadcast / multicast protocols
- network-projection-based routing
- durable communication logs
- fault tolerance beyond normal MPI execution

---

## Future Extensions

Once the first implementation is stable, possible extensions include:

- timeouts on pending acknowledgments
- retries for unacknowledged two-way messages
- delayed communication over multiple ticks
- place-level broadcast
- communication eligibility via `SharedNetwork`
- message priorities
- bandwidth / latency models by place or rank
- communication statistics logging to repast4py datasets

---

## Bottom Line

Implement the preferred design with:

- `Person` deciding communication intents
- `Place` representing occupancy context
- `CommunicationManager` handling routing and MPI transport
- `Model` orchestrating movement, synchronization, and communication phases

This is the preferred architecture because it is clear, modular, and aligned with place-based partitioning in a distributed repast4py model.
