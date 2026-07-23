# DC Metro Synthetic 100-Household Fixture

This is a deterministic, fictional agent-level input dataset for the local
CASMSocial `dc_metro_synthetic_100` scenario. It has 100 households, their
synthetic people, homes, workplaces, schools, and simple weekday activity
plans.

It is generated from `scripts/create_dc_metro_synthetic_fixture.py`; it is not
derived from OSF synthetic-population microdata. All coordinates are fictional
points used only for model mechanics.

Rebuild and validate the fixture:

```bash
uv run python scripts/create_dc_metro_synthetic_fixture.py
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --fixture-path testdata/dc_metro_synthetic_100_households \
  --ducklake-path data/datalakehouse_dc_metro_synthetic_100
```

Then run the scenario with the same DuckLake path configured in
`config/dc_metro_synthetic_100.yaml`.
