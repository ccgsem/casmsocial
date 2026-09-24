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

The gRPC state remains `RUNNING` until execution and final flushing return,
then reports `CANCELLED` for a requested stop. Worker/flush exceptions still
report `FAILED`, with sanitized client-facing error messages. Terminal requests
are unacknowledged and unknown run IDs return `NOT_FOUND`. CASMSim's legacy
four-state Python adapter enum has no cancelled member; adapter-level cancellation
retains its `Failed` mapping after shutdown. Use the gRPC state to distinguish
cancellation from a failure.

`tests/test_runner_cancellation.py` exercises the real loopback gRPC/Flight
servers with event-gated test models. It covers startup races, active execution,
final flushing, repeated cancellation, errors and completed-output retrieval.
This is not evidence of multi-rank cancellation, a live scientific model run,
or durable retrieval after runner/gateway restart; those remain separate gates.

This transport is a material capability addition and requires Public Release
System approval before public distribution.
