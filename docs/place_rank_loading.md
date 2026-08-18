# Loading place→rank assignments into a casmsocial run

## Context

`casmsocial.network_partitioner_ducklake` writes a table

```
partitions.metis_place_partitions(
    imputation INTEGER,
    n_ranks    INTEGER,
    rank       INTEGER,
    place_id   BIGINT
)
```

into the same DuckLake catalog that holds `<schema>.persons`, `<schema>.places`,
and, when configured, the source table named by `households.table` (commonly
`<schema>.hh` in current RTI-style inputs). This note recommends how that table
should be consumed at run time so that

```bash
uv run mpirun -n 4 python -m casmsocial config/casmsocial.yaml
```

honors the METIS partition.

## What's already in place

casmsocial is rank-aware end-to-end; the missing piece is just _populating_ each `Place`'s `rank`:

- `place.py: Place.__init__` reads `initDict["rank"]` (defaults to 0).
- `place.py: PlacesProjectionV2.add_place` only adds a place to the local set if `place.rank == self.rank`.
- `place.py: get_local_places` and `get_all_places` already filter by rank.
- `casmpop.py: create_places` (≈ line 1133) iterates the **`places` temp table** and does:

  ```python
  if "rank" not in place_record:
      place_record["rank"] = 0
  place = placeType(place_record, placeDataType)
  self.places_proj.add_place(place)
  ```

  i.e. **whatever value lives in `places.rank` is used directly**.

So the integration question is exclusively: _how does the `rank` column get into the `places` temp table?_

## Three approaches

### A. SQL join in `create_input_tables` (recommended)

Materialize the `places` temp table as a `LEFT JOIN` against `partitions.metis_place_partitions`, filtered by the run's `(imputation, n_ranks)`:

```python
# casmsocial/casmpop.py — inside create_input_tables(), where the places temp
# table is built today (around line 856).

from mpi4py import MPI

n_ranks = MPI.COMM_WORLD.Get_size()
imputation = self.params.get("Imputation", 1)

partition_table = self.params.get("partition.table")  # e.g. partitions.metis_place_partitions
default_rank   = int(self.params.get("partition.default_rank", 0))

if partition_table:
    self.conn.execute(
        f"""
        CREATE OR REPLACE TEMPORARY TABLE places AS
        SELECT
            p.* EXCLUDE (rank) IF EXISTS,
            COALESCE(part.rank, {default_rank}) AS rank
        FROM {quote_table_identifier(places_table)} p
        LEFT JOIN {quote_table_identifier(partition_table)} part
               ON part.place_id  = p.sp_id
              AND part.imputation = {imputation}
              AND part.n_ranks    = {n_ranks}
        """  # noqa: S608
    )
else:
    # existing behavior
    self.conn.execute(
        f"CREATE OR REPLACE TEMPORARY TABLE places AS "
        f"SELECT * FROM {quote_table_identifier(places_table)}"
    )
```

**Why this is the cleanest path**

- One SQL statement; no Python loop holding a `(place_id → rank)` dict in memory.
- Reuses the column `Place.__init__` and `add_place` _already_ honor.
- Works whether the partition was generated from `persons` or from `activities`; the partitioner's output table is the contract.
- `LEFT JOIN` lets unknown places fall back to `default_rank` (or you can swap in a hash-based default — see "Fallback strategies" below).
- Validation can be a single `SELECT COUNT(*) ... WHERE rank IS NULL` after the join.

The matching step for **household and person ownership** is the same idea, one
join further. `Household` is a social-unit agent loaded from the physical source
table named by the `households.table` config key. In current DMV-style inputs
that table is `hh`; the runtime materializes a temporary `households` table
after normalizing aliases. `Household.place_id` links the social unit to the
physical `Place`. If a household row omits `place_id`, casmsocial treats `sp_id`
as the physical place id. The temp `household_ranks` table preserves the
normalized household columns with their resolved physical place id and owner
rank:

```python
household_columns, place_expr = self._household_columns_expr(
    households_table,
    "h",
    exclude={"rank"},
)
self.conn.execute(
    f"""
    CREATE OR REPLACE TEMPORARY TABLE household_ranks AS
    SELECT
        {household_columns},
        place_ranks.rank
    FROM {quote_table_identifier(households_table)} h
    INNER JOIN place_ranks
            ON place_ranks.sp_id = {place_expr}
    WHERE h.Imputation = {imputation}
    """  # noqa: S608
)
```

`_household_columns_expr()` derives `household_id` from `sp_id` when the source
omits `household_id`, and derives `place_id` from `sp_home_id` or `sp_id` when
the source omits `place_id`. This keeps `hh`-style inputs compatible with the
runtime's normalized `households` temp table.

With `households.table` configured, `Person.sp_hh_id` references the household
id, not necessarily a physical place id. Persons therefore join through
`household_ranks` so they are owned by the rank of the household's linked place:

```python
self.conn.execute(
    f"""
    CREATE OR REPLACE TEMPORARY TABLE persons AS
    SELECT
        pe.*,
        COALESCE(hh.rank, {default_rank}) AS rank
    FROM {persons_identifier} pe
    LEFT JOIN household_ranks hh
           ON hh.household_id = pe.sp_hh_id
    WHERE Imputation = {imputation}
    """  # noqa: S608
)
```

If `households.table` is not configured, the legacy person rank path still treats
`sp_hh_id` as a physical place id and joins it directly to `place_ranks`.

In `create_persons`, filter to `WHERE rank = self.rank` so each rank only instantiates the persons it owns. (Same pattern as `create_places` already does implicitly through `add_place`.)

### B. Load partition map in Python and inject during `create_places`

Read the partition table once into a `{place_id: rank}` dict on rank 0, broadcast via `comm.bcast`, and look up in the per-row loop:

```python
partition_map: dict[int, int] = {}
if self.rank == 0 and partition_table:
    rows = self.conn.execute(
        f"SELECT place_id, rank FROM {quote_table_identifier(partition_table)} "
        f"WHERE imputation = ? AND n_ranks = ?",
        [imputation, n_ranks],
    ).fetchall()
    partition_map = dict(rows)
partition_map = self.comm.bcast(partition_map, root=0)
# ...later, inside create_places:
place_record["rank"] = partition_map.get(place_record["sp_id"], default_rank)
```

**Tradeoffs.** Easier to instrument (you can log mismatches mid-loop), but it duplicates the entire partition map on every rank, blocks on `bcast`, and pulls the rank assignment out of SQL where it's cheapest. Useful as a fallback if the partition table lives in a different store from the places table.

### C. Pre-materialized rank-partitioned places (parquet-Hive style)

Run an offline export job that writes `places_part_n_ranks=4/rank=2/part-0.parquet` directories, then point `places.table` at the partitioned directory. Each rank reads only its partition. This is no longer an in-tree casmsocial workflow; the maintained partitioning path writes place-to-rank assignments into DuckLake and joins them at startup.

**Tradeoffs.** Fastest at run time (zero join, zero broadcast) and zero changes to `create_input_tables`, but couples the catalog to a particular `n_ranks`. Re-running with a different rank count means materializing again. Good for benchmark / production runs against a fixed cohort; awkward for interactive work or for the GUI controller in casmservice that should be free to pick `n_ranks` per submission.

## YAML schema additions

Add a small `partition.*` stanza to the casmsocial YAML conventions, optional so existing configs keep working:

```yaml
# config/casmsocial.yaml (additions)

# Optional METIS-derived place→rank assignments. When set, the temporary
# places, households, and persons tables are joined against this table at
# startup; the `rank` column on each `Place`, each `Household`'s linked place,
# and the home rank for each `Person` come from here. When unset, every place
# falls back to `partition.default_rank` (0 by default) and the run behaves as
# it does today.
partition.table: 'partitions.metis_place_partitions'
partition.default_rank: 0          # used for places not present in the partition table
partition.require_full_coverage: true   # raise if any place has no assignment
```

`partition.imputation` is intentionally omitted — it always equals the run's `Imputation` parameter. `partition.n_ranks` is omitted because it always equals `MPI.COMM_WORLD.Get_size()` and would only be a foot-gun if the two could disagree.

## Validation at startup

After `create_input_tables`, before `create_places` runs, do three quick checks on rank 0:

1. **Partition exists for this `(imputation, n_ranks)`.**

   ```sql
   SELECT COUNT(*) FROM partitions.metis_place_partitions
   WHERE imputation = ? AND n_ranks = ?
   ```

   If 0, log a warning and either fall back (Option D below) or raise depending on `partition.require_partition`.

2. **Coverage of `places`.**

   ```sql
   SELECT COUNT(*) FROM places WHERE rank IS NULL
   ```

   With the SQL in Approach A, `rank` is `COALESCE`-d to `default_rank`, so check before the COALESCE — or run a separate query joining the source tables. If `partition.require_full_coverage = true` and any places fall through, raise `MissingPartitionAssignmentError`.

3. **Households or persons whose home is unpartitioned.**

   ```sql
   SELECT COUNT(*) FROM household_ranks WHERE rank IS NULL;
   SELECT COUNT(*) FROM persons WHERE rank IS NULL;
   ```

   Same policy. With `households.table` configured, a common cause is a
   household whose resolved `place_id` exists in `places` but was not connected
   to any other place in the activity graph. Without `households.table`, the
   older case is a person whose `sp_hh_id` references an unpartitioned physical
   place. Either treat them as `default_rank` or extend the partitioner to emit
   assignments for every place id in the places table, not just nodes that ended
   up in the graph.

These checks are cheap (count queries against the small partition + temp tables) and they catch the most common operator mistakes — running `mpirun -n 4` against a partition table generated for `n_ranks=8`, or pointing at a partition table that doesn't yet have an entry for the requested imputation.

## Fallback strategies

Three sensible fallbacks when the partition table is empty or partial:

1. **Fixed default rank** (current behavior, `rank = 0`). All places concentrate on rank 0; only useful for testing on a single rank.
2. **Hash-based round-robin.** `COALESCE(part.rank, abs(hashtext(CAST(sp_id AS VARCHAR))) % n_ranks)` in the SQL. Distributes uniformly without consulting any external state. A good default when the partition is partial.
3. **Soft-fail with required mode.** `partition.require_full_coverage = true` raises early if any place is unassigned, surfacing the operator error before the simulation starts.

I'd default to a hash-based fallback (option 2) with `partition.require_full_coverage = false` so casmsocial keeps working out of the box for users who haven't run the partitioner; flip to option 3 for production runs where data integrity matters.

## End-to-end run, recommended Option A

```bash
# 1. Generate the partition (one-time per (imputation, n_ranks) pair).
uv run python -m casmsocial.network_partitioner_ducklake \
    --schema wake_county_heat \
    --imputations all \
    --n-ranks 2,4,8 \
    --output-table partitions.metis_place_partitions \
    --restrict-to-places

# 2. Add the partition.* stanza to your YAML once (see above).

# 3. Run as you do today. The number of ranks must equal partition.n_ranks.
uv run mpirun -n 4 python -m casmsocial config/casmsocial.yaml
```

## Persons-side write of the rank column to the AgentLogger output

An additional benefit of Approach A: because the `rank` column flows all the way
through to `Place`, then through `Household.place_id` to `Person.sp_hh_id`, the
`AgentLogger`'s Hive-partitioned Parquet output already has `tick`/`rank`
partitions, but each row's `rank` directly reflects the METIS assignment rather
than just where it happened to land. That makes downstream cross-rank
consistency analysis a single GROUP BY on the output Parquet — no need to
re-derive ownership at analysis time.

## Recommendation summary

- **Adopt Approach A.** One SQL change in `create_input_tables`, plus a `WHERE rank = self.rank` filter on the persons load. Reuses the rank-aware machinery already in `Place` / `PlacesProjectionV2`. No Python broadcast, no offline parquet preprocessing.
- **Add the `partition.*` YAML stanza.** Keep all three keys optional so existing configs keep working.
- **Validate at startup.** Three count queries, executed on rank 0, against the partition and temp tables. Fail fast on n_ranks / imputation mismatch when `require_full_coverage = true`; log + hash-fallback otherwise.
- **Reserve Approach C for production runs at fixed `n_ranks`.** It pays off only when the same partition is reused many times.

The CLI command stays exactly what users already type:

```bash
uv run mpirun -n 4 python -m casmsocial config/casmsocial.yaml
```

with a small one-time YAML addition that points the run at the partition table.
