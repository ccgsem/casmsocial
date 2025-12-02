# Hive-Partitioned Persons Output Schema

## Overview

The network partitioner scripts now output persons data as a standard Hive-partitioned parquet dataset with root directory `persons.parquet`.

## Directory Structure

```
data/partitioned_8proc/
├── places_partition.parquet              (single file, not partitioned)
└── persons.parquet/                      (Hive-partitioned root directory)
    └── Imputation=1/                     (Imputation partition column)
        ├── process=0/
        │   └── *.parquet                 (parquet file(s))
        ├── process=1/
        │   └── *.parquet
        ├── process=2/
        │   └── *.parquet
        ├── process=3/
        │   └── *.parquet
        ├── process=4/
        │   └── *.parquet
        ├── process=5/
        │   └── *.parquet
        ├── process=6/
        │   └── *.parquet
        └── process=7/
            └── *.parquet
```

## Partition Columns

The dataset is partitioned by two columns:

| Column | Type | Values | Purpose |
|--------|------|--------|---------|
| **Imputation** | int32 | 1, 2, 3, ... | Distinguishes different synthetic population versions |
| **process** | int64 | 0-7 (for 8 ranks) | MPI process rank assignment |

Partition columns are **stored in the directory structure** (not in the parquet files), enabling:
- Efficient partition pruning when querying
- Standard Hive compatibility
- Easy filtering by imputation and process

## Data Columns

18 columns stored in the parquet files:

| Column | Type | Description | Null Count |
|--------|------|-------------|-----------|
| `sp_id` | int32 | Unique person identifier | 0 |
| `sp_hh_id` | int32 | Home place ID (household) | 0 |
| `sporder` | int32 | Order within household | 59 |
| `relationship` | int32 | Family relationship code | 59 |
| `age` | int32 | Age in years | 59 |
| `sex` | int32 | Sex/gender code | 59 |
| `race` | int32 | Race/ethnicity code | 59 |
| `income` | float64 | Household income | 52,549 |
| `student_level` | int32 | Education/student level | 139,680 |
| `job` | string | Job category | 108,726 |
| `industry` | string | Industry category | 108,726 |
| `soc` | string | SOC occupational code | 96,169 |
| `outside_worker` | bool | Works outside study area | 0 |
| `veteran` | int32 | Veteran status code | 196,302 |
| `sp_work_id` | int32 | Work place ID | 108,726 |
| `sp_school_id` | int32 | School place ID | 139,680 |
| `Imputation` | int32 | **Partition column** (in directory, not file) | 0 |
| `process` | int64 | **Partition column** (in directory, not file) | 0 |

**Note**: Columns `Imputation` and `process` are stored as directory names, not in the parquet file data.

## Reading the Data

### Method 1: Pandas (Simplest)

```python
import pandas as pd

# Pandas automatically detects Hive partitioning
persons = pd.read_parquet('data/partitioned_8proc/persons.parquet')

# Result: 1,045,389 rows × 18 columns
print(f"Shape: {persons.shape}")
print(f"Columns: {persons.columns.tolist()}")
```

### Method 2: DuckDB (Recommended for queries)

```python
import duckdb

conn = duckdb.connect()

# Query with automatic Hive partition discovery
result = conn.execute("""
    SELECT
        Imputation,
        process,
        COUNT(*) as person_count
    FROM read_parquet(
        'data/partitioned_8proc/persons.parquet/**/*.parquet',
        hive_partitioning = 1
    )
    GROUP BY Imputation, process
    ORDER BY process
""").df()
```

### Method 3: Filter Specific Partition

```python
import pandas as pd

# Read only process 0 for Imputation 1
persons_proc_0 = pd.read_parquet(
    'data/partitioned_8proc/persons.parquet/Imputation=1/process=0/'
)

print(f"Persons in process 0: {len(persons_proc_0):,}")
```

## Benefits of Hive Partitioning

1. **Standard Format**
   - Compatible with Apache Spark, Presto, Trino, Athena
   - Follows standard partition naming convention
   - Directory structure is self-documenting

2. **Partition Pruning**
   - Query engines automatically skip irrelevant partitions
   - Only reads data for specified Imputation/process values
   - Reduces I/O and improves query speed

3. **Scalability**
   - Easy to add new imputations without restructuring
   - Handles multiple processes automatically
   - Can extend to multiple imputation values

4. **Tool Compatibility**
   - Works with Spark SQL: `spark.read.parquet('persons.parquet')`
   - Works with Presto: `CREATE EXTERNAL TABLE persons LOCATION 'persons.parquet'`
   - Works with DuckDB, pandas, polars, etc.

## File Size Statistics

Based on test run with 1,045,389 persons across 8 processes:

```
Process 0: 203,630 persons → ~3.2 MB
Process 1: 203,970 persons → ~3.2 MB
Process 2: 157,278 persons → ~2.7 MB
Process 3: 171,849 persons → ~2.8 MB
Process 4:  76,144 persons → ~1.4 MB
Process 5:  73,618 persons → ~1.3 MB
Process 6:  78,859 persons → ~1.4 MB
Process 7:  80,041 persons → ~1.5 MB
────────────────────────────────────────
Total:   1,045,389 persons → ~18 MB (data files)
```

## Data Integrity Verification

All 1,045,389 persons correctly partitioned:

```
Imputation=1, process=0: 203,630 persons
Imputation=1, process=1: 203,970 persons
Imputation=1, process=2: 157,278 persons
Imputation=1, process=3: 171,849 persons
Imputation=1, process=4:  76,144 persons
Imputation=1, process=5:  73,618 persons
Imputation=1, process=6:  78,859 persons
Imputation=1, process=7:  80,041 persons
────────────────────────────────────
Total:                   1,045,389 persons ✓
```

- All persons have unique `sp_id`
- No missing partition values
- All partition assignments match METIS optimization

## Comparison: Old vs New Format

### Old Format (Manual Directory Structure)
```
Imputation=1/
├── partition=0/part-0.parquet
├── partition=1/part-0.parquet
├── partition=2/part-0.parquet
└── ...
```

**Issues:**
- Non-standard naming (partition=N instead of process=N)
- Manual directory creation
- Limited tool compatibility
- Confusing directory structure

### New Format (Hive Partitioning)
```
persons.parquet/
└── Imputation=1/
    ├── process=0/part-*.parquet
    ├── process=1/part-*.parquet
    ├── process=2/part-*.parquet
    └── ...
```

**Benefits:**
- Standard Hive naming convention
- Automatic partition handling by pandas/pyarrow
- Full ecosystem compatibility (Spark, Presto, DuckDB)
- Scalable to multiple imputations
- Self-documenting structure

## Implementation Details

### Output Code
```python
persons["Imputation"] = imputation

persons.to_parquet(
    path=str(persons_output_dir),
    engine="pyarrow",
    partition_cols=["Imputation", "process"],
    index=False,
)
```

### Reading Code
```python
# Automatic detection
persons_df = pd.read_parquet('persons.parquet')

# With DuckDB and Hive partitioning
conn.execute(
    "SELECT * FROM read_parquet('persons.parquet/**/*.parquet', hive_partitioning=1)"
)
```

## Multiple Imputations

When running with multiple imputations, the structure becomes:

```
persons.parquet/
├── Imputation=1/
│   ├── process=0/part-*.parquet
│   ├── process=1/part-*.parquet
│   └── ...
├── Imputation=2/
│   ├── process=0/part-*.parquet
│   ├── process=1/part-*.parquet
│   └── ...
└── Imputation=3/
    ├── process=0/part-*.parquet
    ├── process=1/part-*.parquet
    └── ...
```

**Reading specific imputation:**
```python
import duckdb

# Get all persons for Imputation 2
result = duckdb.execute("""
    SELECT * FROM read_parquet('persons.parquet/**/*.parquet', hive_partitioning=1)
    WHERE Imputation = 2
""").df()
```

## Important Notes

1. **Partition columns are directory structure**: `Imputation` and `process` are stored as directory paths, not in the data files. This is efficient for partition pruning.

2. **Automatic detection**: Both pandas and DuckDB automatically detect and read Hive partitions without additional configuration.

3. **Compatibility**: This format is compatible with standard Hive tools and should work with any parquet reader that supports Hive partitioning.

4. **Scalability**: Adding new imputations is as simple as re-running the partitioner with a different imputation number - the new data is automatically in the correct directory structure.

5. **Performance**: Hive partitioning enables efficient queries by automatically skipping irrelevant data partitions.
