# Runtime Launch Configs

Files in this directory are runnable launch configurations for local smoke
tests, examples, and operator workflows. They are not the canonical catalog
scenario definitions.

`casmsocial.yaml` is the candidate public default. It expects the Wake County
Heat fixture to be materialized into `data/datalakehouse` before running:

```bash
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --fixture-path testdata/wake_county_heat_1000_households \
  --ducklake-path data/datalakehouse
```

`mvp.yaml` is a smaller self-contained smoke configuration backed by
`scripts/create_mvp_ducklake.py`.

Canonical casmsocial scenarios live in `scenarios/casmsocial/*.yaml`. The
candidate public registered scenario is `wake_county_heat`. Update those files
when changing scenarios registered by `scripts/register_casmsocial.py`.

Use `config/*.yaml` when running the simulator directly, for example:

```bash
uv run mpirun -n 1 python -m casmsocial config/mvp.yaml
```
