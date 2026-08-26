# Runner Transport Architecture

Status: accepted — 2026-08-26

## Context

CASMSocial runs simulations through more than one backend, including
repast4py and external xDEVS runners. A runner needs a small control plane for
run lifecycle and a data plane for live observation batches. The control plane
is proposed as the versioned `casm.runner.v1.SimulatorControl` gRPC service.
The data plane uses Apache Arrow Flight, whose schema discovery and record-batch
transfer semantics are a better fit for Arrow observations than a bespoke
streaming RPC.

Arrow Flight is implemented by `pyarrow.flight`, while the proposed control
plane is implemented with `grpcio`. Although both use gRPC, their Python
servers are independently owned and cannot safely be registered onto one
listener.

## Decision

One runner process will expose two loopback-only listeners:

- a gRPC control listener implementing `SimulatorControl` for `Start`,
  `Cancel`, and `GetState`;
- an Arrow Flight listener exposing read-only observation channels through
  `list_flights`, `get_flight_info`, and `do_get`.

The process writes a single `runner_endpoints.json` manifest in its run
directory. The manifest records the control and Flight addresses, their
protocol versions, and the allocated run identifier. Callers must create the
run directory with owner-only permissions and treat the manifest as a local
trust-boundary artifact.

Both transports read from one bounded observation broker. Simulation backends
publish immutable Arrow tables to the broker; the broker assigns a strictly
increasing `batch_id` for each channel. `batch_id` is the only resume cursor.
It is not a simulation tick: one model step can publish zero, one, or many
batches.

The broker has explicit per-channel retention limits in rows, batches, and
bytes. A backend blocks or receives a defined overflow failure when those
limits are reached; it must never grow an in-memory queue without bound.
Once a batch is evicted, a reconnect using an older cursor receives an
out-of-range response rather than a partial, silently inconsistent replay.

Cancellation requests stop new simulation work at the backend's next safe
boundary. The broker closes streams after publishing already accepted batches
and records the terminal result. Disconnected consumers are removed promptly
and do not keep broker data alive.

## Backend adapter contract

Every backend adapter, including repast4py and xDEVS, supplies:

1. lifecycle operations: start, cooperative cancellation, state, and terminal
   result;
2. a validated mapping from model-owned observation-channel names to outputs;
3. immutable Arrow tables sent to the broker with their channel name; and
4. a backend progress value for `GetState` (repast4py normally reports its
   schedule tick; continuous-time backends report simulation time).

The transport contract does not contain backend-specific model configuration.
That configuration remains inside the validated `Start` request payload.

## Consequences

This design permits CASMSocial, repast4py, and xDEVS to share control and
observation semantics while retaining backend-specific execution code. It
avoids a custom Arrow-byte streaming protocol and does not require either
Python gRPC implementation to share a TCP listener with Arrow Flight.

The first implementation must include an integration test that starts one
reference run and verifies that Flight and the control-plane observation stream
produce the same channel names, schemas, batch order, and terminal outcome.
Because this is a material public capability addition, it requires review in
the appropriate Public Release System request before publication.
