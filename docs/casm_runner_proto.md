# CASM runner protocol generation

`proto/casm_runner/v1/casm_runner.proto` is the authoritative loopback runner
control contract. Its generated Python modules are checked in under
`casmsocial/casmsim/proto/` so runtime users do not need a protobuf compiler.

After changing the protocol, regenerate and commit both modules:

```shell
uv run python scripts/generate_casm_runner_proto.py
```

The proto-generation test verifies the committed output is reproducible.
