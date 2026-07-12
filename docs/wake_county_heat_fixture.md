# Wake County Heat Deployment Fixture

This runbook covers the minimum Wake County Heat dataset used to prove casmsocial
DuckLake deployments. The fixture lives at
`testdata/wake_county_heat_1000_households` and is the source of truth for both
local and production-style deployment tests.

## Scope

The fixture materializes these tables under the `wake_county_heat` schema:

| Table | Rows |
| --- | ---: |
| `persons_1000_households` | 2,223 |
| `hh_1000_households` | 1,000 |
| `activities_1000_households` | 5,264 |
| `places` | 380,396 |

The fixture contains synthetic population, household, activity, and place
records. It also contains coordinate/location fields, so confirm data-owner and
policy clearance before pushing or mirroring it outside approved repositories.

## Fixture Package

The package layout is:

```text
testdata/wake_county_heat_1000_households/
  README.md
  manifest.yaml
  tables/
    activities_1000_households.parquet
    hh_1000_households.parquet
    persons_1000_households.parquet
    places.parquet
```

`manifest.yaml` records the schema, file paths, row counts, DuckDB-visible
columns, sizes, and SHA-256 checksums. The loader validates the manifest before
writing any DuckLake tables.

## Local DuckLake

Use this path to create or replace the four fixture tables in the local
DuckLake deployment under `data/datalakehouse`:

```bash
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --fixture-path testdata/wake_county_heat_1000_households \
  --ducklake-path data/datalakehouse
```

`data/datalakehouse` is generated runtime state and is intentionally ignored by
Git. Rebuild it from the fixture instead of committing DuckLake metadata or
storage internals.

For a non-destructive check, write to a scratch ignored path:

```bash
uv run python scripts/materialize_wake_county_heat_fixture.py \
  --fixture-path testdata/wake_county_heat_1000_households \
  --ducklake-path data/datalakehouse_wake_county_heat_fixture_check
```

To validate the full local deployment path, including the shipped
`config/casmsocial.yaml` and a one-hour model smoke run, use:

```bash
uv run python scripts/validate_wake_county_heat_deployment.py \
  --fixture-path testdata/wake_county_heat_1000_households \
  --ducklake-path data/datalakehouse
```

The validation rematerializes the fixture, runs the default config with a
one-hour override, and checks that the agent log contains 2,223 agents for one
tick.

## Postgres Catalog And S3 Storage

Use this path to prove the production-style DuckLake deployment shape: Postgres
stores the DuckLake catalog and MinIO provides S3-compatible object storage.

The Compose stack builds the casmsocial production image, starts Postgres and
MinIO, creates the bucket, and runs the fixture loader:

```bash
UV_INSECURE_HOST=download.pytorch.org docker compose \
  -f docker-compose.ducklake-fixture.yaml \
  -p casmsocial-ducklake-fixture \
  up --build --abort-on-container-exit \
  --exit-code-from ducklake-fixture-loader \
  ducklake-fixture-loader
```

`UV_INSECURE_HOST=download.pytorch.org` is needed in environments whose
certificate store does not trust the PyTorch CPU wheel host. If your
environment trusts that host, omit the variable.

The expected loader output is:

```text
Materialized Wake County Heat fixture: target=postgres-s3 ... schema=wake_county_heat (persons_1000_households=2223, hh_1000_households=1000, activities_1000_households=5264, places=380396)
```

The loader redacts the Postgres password in its summary output.

Stop the stack after validation:

```bash
docker compose \
  -f docker-compose.ducklake-fixture.yaml \
  -p casmsocial-ducklake-fixture \
  down
```

This removes containers and the network but preserves the named volumes:

```text
casmsocial-ducklake-fixture_ducklake-minio-data
casmsocial-ducklake-fixture_ducklake-postgres-data
```

Remove those volumes only when you no longer need to inspect the loaded
catalog/storage state:

```bash
docker volume rm \
  casmsocial-ducklake-fixture_ducklake-minio-data \
  casmsocial-ducklake-fixture_ducklake-postgres-data
```

## Environment Overrides

The Compose file has defaults suitable for local validation. Override these
when needed:

| Variable | Default |
| --- | --- |
| `CASMSOCIAL_DUCKLAKE_POSTGRES_DB` | `casmsocial_ducklake` |
| `CASMSOCIAL_DUCKLAKE_POSTGRES_USER` | `casmsocial` |
| `CASMSOCIAL_DUCKLAKE_POSTGRES_PASSWORD` | `casmsocial` |
| `CASMSOCIAL_DUCKLAKE_POSTGRES_PORT` | `55432` |
| `CASMSOCIAL_DUCKLAKE_S3_BUCKET` | `casmsocial-ducklake` |
| `CASMSOCIAL_DUCKLAKE_S3_ACCESS_KEY_ID` | `casmsocial` |
| `CASMSOCIAL_DUCKLAKE_S3_SECRET_ACCESS_KEY` | `casmsocial-secret` |
| `CASMSOCIAL_DUCKLAKE_S3_REGION` | `us-east-1` |
| `CASMSOCIAL_DUCKLAKE_MINIO_API_PORT` | `59000` |
| `CASMSOCIAL_DUCKLAKE_MINIO_CONSOLE_PORT` | `59001` |

The direct `postgres-s3` loader mode also accepts equivalent CLI options:

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

For containerized MinIO, pass the endpoint as the Docker service name and port
without an `http://` prefix.

## Failure Triage

If fixture validation fails, inspect `manifest.yaml` first. The loader checks
checksums, row counts, and DuckDB-visible schemas before loading.

If Docker image build fails while downloading Torch, rerun with:

```bash
UV_INSECURE_HOST=download.pytorch.org
```

If the loader cannot attach DuckLake, check that Postgres is healthy and that
the MinIO init service completed. The Compose command exits with the loader's
status, so a nonzero exit means the fixture was not fully loaded.
