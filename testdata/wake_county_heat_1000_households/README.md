# Wake County Heat 1000-Household Fixture

This directory contains the minimum Wake County Heat dataset used to test casmsocial deployments.
The fixture is exported from the DuckLake catalog tables, not copied from DuckLake storage internals.

## Tables

- `wake_county_heat.persons_1000_households`
- `wake_county_heat.hh_1000_households`
- `wake_county_heat.activities_1000_households`
- `wake_county_heat.places`

## Intended Use

Use this fixture as the immutable source data for both supported deployment tests:

1. Local DuckLake materialized under `data/datalakehouse`.
2. Production-style DuckLake with a Postgres catalog and S3-compatible object storage in Docker.

`data/datalakehouse` remains generated runtime state and should stay out of Git. Loader scripts should rebuild deployment state from the Parquet files and validate row counts, schemas, and checksums against `manifest.yaml`.

## Local DuckLake Materialization

Run this from the repository root to create or replace the four fixture tables in a local DuckLake:

```bash
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --fixture-path testdata/wake_county_heat_1000_households \
  --ducklake-path data/datalakehouse
```

The materializer validates the manifest checksums, row counts, and schemas before loading, then verifies the loaded DuckLake tables.

See `docs/wake_county_heat_fixture.md` for the full operator runbook covering
local DuckLake materialization and production-style Postgres/S3 validation.

## Postgres Catalog and S3-Compatible Storage

The same fixture can be loaded into a production-style DuckLake deployment with a Postgres catalog and S3-compatible storage:

```bash
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --target postgres-s3 \
  --fixture-path testdata/wake_county_heat_1000_households \
  --postgres-connection "host=ducklake-postgres port=5432 dbname=casmsocial_ducklake user=casmsocial password=casmsocial" \
  --s3-data-path s3://casmsocial-ducklake/wake_county_heat \
  --s3-endpoint ducklake-minio:9000 \
  --s3-access-key-id casmsocial \
  --s3-secret-access-key casmsocial-secret \
  --s3-region us-east-1 \
  --s3-url-style path \
  --no-s3-use-ssl
```

For containerized MinIO, the endpoint should be the Docker service name and port without an `http://` prefix. The loader creates a DuckDB S3 secret, attaches `ducklake:postgres:<connection>`, and writes DuckLake table data to the configured S3 URI.

## Data Review Note

This fixture contains synthetic population, household, activity, and place records. It also includes precise coordinate/location fields, so data-owner and policy clearance should be confirmed before pushing this fixture to shared remotes.
