# Runtime Launch Configs

Files in this directory are runnable launch configurations for local smoke
tests, examples, and operator workflows. They are not the canonical catalog
scenario definitions.

Canonical casmsocial scenarios live in `scenarios/casmsocial/*.yaml`. Update
those files when changing the named scenarios registered in casmdb by
`scripts/register_casmsocial.py`.

Use `config/*.yaml` when running the simulator directly, for example:

```bash
uv run mpirun -n 1 python -m casmsocial config/mvp.yaml
```

