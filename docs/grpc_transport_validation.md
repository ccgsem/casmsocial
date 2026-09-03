# gRPC Transport Validation

The loopback runner exposes a versioned `casm.runner.v1.SimulatorControl` gRPC
control endpoint and a read-only Arrow Flight observation endpoint. Both read
from one bounded observation broker.

Validation covers broker retention, Arrow Flight retrieval, loopback endpoint
manifest permissions, repast4py observer publication, and parity between the
gRPC observation stream and Arrow Flight output.

The control listener binds only to `127.0.0.1`; its endpoint manifest is stored
in an owner-only run directory with owner-only file permissions. The runner
accepts one run. Cooperative cancellation is not yet implemented and the
control API reports cancellation requests as unacknowledged.

This transport is a material capability addition and requires Public Release
System approval before public distribution.
