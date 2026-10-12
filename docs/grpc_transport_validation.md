# gRPC Transport Validation

The loopback runner exposes a versioned `casm.runner.v1.SimulatorControl` gRPC
control endpoint and a read-only Arrow Flight observation endpoint. Both read
from one bounded observation broker.

Validation covers broker retention, Arrow Flight retrieval, loopback endpoint
manifest permissions, repast4py observer publication, and parity between the
gRPC observation stream and Arrow Flight output.

The control listener binds only to `127.0.0.1`; its endpoint manifest is stored
in an owner-only run directory with owner-only file permissions. The runner
accepts one run. The CASMSocial launcher connects the CASMSim cancellation hook
to the active adapter and model. Requests received during adapter/model startup
are retained; repeated requests are idempotent. Cancellation is cooperative,
not a forced process kill: an acknowledgement means the request was accepted,
not that execution has stopped. A blocked constructor, model tick or output
flush can still delay termination.

For MPI launches, only rank 0 receives the cancellation hook, and no MPI calls
are made from the control request thread. A startup request remains latched
until rank 0's model exists, then signals that model's cancellation flag. All
ranks still construct and start the CASMSocial model together; its first tick
check propagates cancellation through the existing collective. Multi-rank
cancellation therefore does not skip model initialization. Single-process
cancellation may still skip startup entirely.
After adapter execution and flushing, all ranks meet at a main-thread barrier
before rank 0 releases the control callback. Custom MPI adapters must likewise
preserve collective participation when cancelled during startup.

`tests/test_grpc_runner_launcher.py` and `tests/test_grpc_runner_main_thread.py`
cover rank ownership, normalized request broadcasting, callback completion and
main-thread enforcement with simulated communicators. `tests/test_grpc_runner_mpi.py`
runs the real CLI in one or two MPI processes with `MPI_THREAD_FUNNELED`
requested, then submits requests over loopback gRPC and retrieves output over
Flight. Its completion, cancellation and delayed-worker-flush cases verify that
only rank 0 opens listeners and writes endpoint files, all ranks receive the
normalized configuration, MPI calls stay on main threads, and gRPC remains
`RUNNING` while a worker is finishing output. The fixture stops its server after
retrieval and the test cleans up its own process group on timeout. MPI subprocess
tests skip when the MPI runtime or launcher is unavailable or when already
running under a multi-rank MPI launch; the CLI smoke fixture requires POSIX.

Run these tests from a single-process pytest invocation; they launch their own
MPI ranks:

```bash
uv run pytest tests/test_grpc_runner_mpi.py tests/test_runner_mpi_cancellation.py tests/test_runner_mpi_failures.py -q
```

The gRPC state remains `RUNNING` until execution and final flushing return,
then reports `CANCELLED` for a requested stop. Single-rank worker/flush exceptions
report `FAILED`, with sanitized client-facing error messages. Terminal requests
are unacknowledged and unknown run IDs return `NOT_FOUND`. CASMSim's legacy
four-state Python adapter enum has no cancelled member; adapter-level cancellation
retains its `Failed` mapping after shutdown. Use the gRPC state to distinguish
cancellation from a failure.

`tests/test_runner_cancellation.py` exercises the real loopback gRPC/Flight
servers with event-gated test models. It covers startup races, active execution,
final flushing, repeated cancellation, errors and completed-output retrieval.
`tests/test_runner_mpi_cancellation.py` launches two MPI processes with a small
collective model and the real `CasmPop.step()` cancellation check. It covers
queued requests, adapter resolution, construction, observer registration,
active execution and final flushing, including repeated and terminal requests.
The subprocesses have a timeout so a collective deadlock fails the test.
These tests do not exercise a live scientific model or durable retrieval after
runner/gateway restart; those remain separate gates.

## MPI failure handling

Before dispatch, malformed configuration, listener startup failures and startup
interrupts release worker ranks with an abort message. The broker closes and
pending or late control callbacks receive an error instead of waiting forever.
An accepted invalid configuration reports `FAILED` through gRPC. Main-thread
interrupts are translated to ordinary exceptions on the control callback so
CASMSim can report failure, while the original interrupt drives CLI shutdown.

After dispatch begins, an exception in adapter resolution, model construction,
execution, flushing or the completion barrier may leave peers in different
collectives. The failing rank logs the traceback and calls `MPI.Abort(1)` from
its main thread. A failed abort-message broadcast also aborts the job. This
terminates every rank with a failing process status; the supervisor must use
that status and runner stderr because the gRPC server may exit before a final
state can be read. The launcher does not attempt to recover the communicator or
claim that output was fully flushed after such a failure. Code that hangs
without raising still requires a supervisor timeout.

`tests/test_runner_mpi_failures.py` verifies pre-dispatch recovery and injects
constructor, execution, flush and completion failures on both rank 0 and rank 1
in two real MPI processes. Every subprocess has a time limit, and fatal cases
must exit nonzero with the failing rank's diagnostic.

## Agent-log output contract

The default `AgentLogger` emits non-null identity columns with stable Arrow
types: `run_id` string, `random_seed` int64, `tick` int32 (elapsed minutes),
`rank` int32 and `agent_id` int64. Default state columns are `x`/`y` float64
and `place_id` int64, also non-null. Safe casts reject fractional integer
identities/ticks, overflow and null required values before publication or writing.
Configured additional state columns retain inferred types. An empty snapshot
retains the previous schema; the default schema is also available before any
agents appear. Unknown custom-column types are not guessed on an initial empty
snapshot. Empty snapshots do not replay previous agent rows or create Parquet
files.

The bridge remembers the latest published table per channel and skips an
identical terminal snapshot. It still publishes changed or new terminal channels
and does not deduplicate identical batches from different simulation steps.
This is snapshot replay prevention, not a general row-key deduplication policy.

`tests/test_agent_log_transport_contract.py` covers these guarantees and checks
that live and Hive-reconstructed durable column types agree. Hive readers may
mark partition fields nullable even though the actual identity values are
validated as non-null; column ordering may also differ after reconstruction.

This transport is a material capability addition and requires Public Release
System approval before public distribution.
