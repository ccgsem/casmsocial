# METIS Partitioning Implementation Guide

## Overview

The updated `graph_from_data.py` creates a NetworkX graph of activity locations (home, school, work) for each person agent. The next steps are:

1. **Run graph_from_data.py** to generate METIS-compatible graph file
2. **Install METIS** (if not already installed)
3. **Run METIS partitioning** to assign processes
4. **Create partitioned data files** with process assignments
5. **Run simulation** with MPI using partitioned data

---

## Step 1: Generate the Graph File

The graph has:
- **Nodes** = Places (sp_id values)
- **Edges** = Connections between places visited by same person
- **Format** = METIS-compatible adjacency list

### Command

```bash
cd /Users/joncline/Documents/GitHub/casmsocial

# Generate graph for imputation 1
python -m casmsocial.graph_from_data \
  --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \
  --imputation 1 \
  --output-file data/wake_county_graph.txt \
  --map-file data/wake_county_graph_id_map.txt \
  --save-networkx
```

### Output Files

- `data/wake_county_graph.txt` - METIS format graph (vertices and edges)
- `data/wake_county_graph_id_map.txt` - Mapping from spatial IDs to graph IDs
- `data/wake_county_graph.graphml` - GraphML format (optional, for analysis)
- `data/wake_county_graph.gml` - GML format (optional, for analysis)
- `data/wake_county_graph_stats.txt` - Network statistics

### Expected Output

```
Graph generation complete: NumV: 380396, NumE: ~2000000
```

The graph will have:
- Nodes: Number of unique places
- Edges: Connections between places visited by same person

---

## Step 2: Install METIS (if needed)

### Check if Already Installed

```bash
which gpmetis
```

If this shows a path, METIS is already installed. Skip to Step 3.

### Install METIS

#### macOS
```bash
brew install metis
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install metis libmetis-dev
```

#### Manual Build
```bash
# Download from http://glaros.dtc.umn.edu/gkhome/metis/metis/download
cd /tmp
wget http://glaros.dtc.umn.edu/gkhome/metis/metis/download
tar -xzf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=/usr/local
make install
export PATH=/usr/local/bin:$PATH
```

### Verify Installation

```bash
gpmetis --help
# Should show METIS usage information
```

---

## Step 3: Run METIS Partitioning

METIS partitions the graph to minimize edge cuts (communication between processes).

### Command

For **8 processes** (fits within 5-node HPC limit):
```bash
cd /Users/joncline/Documents/GitHub/casmsocial

gpmetis data/wake_county_graph.txt 8
```

For **16 processes** (if you want more parallelism):
```bash
gpmetis data/wake_county_graph.txt 16
```

### What METIS Does

- Reads graph from `wake_county_graph.txt`
- Partitions nodes (places) into 8 balanced groups
- Minimizes edges crossing partitions (reduces MPI communication)
- Writes partition assignment to `wake_county_graph.txt.part.8`

### Output Files

- `data/wake_county_graph.txt.part.8` - Partition assignments (8 lines, one per node)
  - Format: Each line contains a process ID (0-7) for that node

### Example Partition Output

```
0
0
1
0
2
1
3
...
```

Each line represents which process (0-7) that place is assigned to.

---

## Step 4: Create Partitioned Data Files

Now we need to:
1. Read the partition assignments
2. Add process IDs to persons and places dataframes
3. Save partitioned files for MPI execution

### Create a Python Script

Save this as `partition_data.py`:

```python
import pandas as pd
import glob
from pathlib import Path
import duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metis_partition(partition_file):
    """Load METIS partition file into dict mapping node_id -> partition_id."""
    partition_map = {}
    with open(partition_file, 'r') as f:
        for node_id, line in enumerate(f, start=1):  # 1-based node IDs
            partition_id = int(line.strip())
            partition_map[node_id] = partition_id
    return partition_map

def load_graph_id_map(map_file):
    """Load graph ID map into dict mapping spatial_id -> graph_id."""
    spid_to_graphid = {}
    with open(map_file, 'r') as f:
        for line in f:
            sp_id, graph_id = map(int, line.strip().split())
            spid_to_graphid[sp_id] = graph_id
    return spid_to_graphid

def main():
    # Paths
    persons_file = Path("data/processed/wake_county_30_v2/abm_inputs/persons.parquet")
    places_file = Path("data/processed/apache_flight/wake_county_30_v2/places.parquet")
    partition_file = Path("data/wake_county_graph.txt.part.8")
    map_file = Path("data/wake_county_graph_id_map.txt")
    output_dir = Path("data/partitioned_8proc")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mappings
    logger.info("Loading METIS partition assignments...")
    partition_map = load_metis_partition(partition_file)  # graph_id -> partition_id
    
    logger.info("Loading graph ID to spatial ID mapping...")
    spid_to_graphid = load_graph_id_map(map_file)  # sp_id -> graph_id
    graphid_to_spid = {v: k for k, v in spid_to_graphid.items()}  # graph_id -> sp_id
    
    # Load places and add process assignments
    logger.info("Loading places file...")
    places_df = pd.read_parquet(places_file)
    
    # Map place sp_id to partition ID
    places_df['process'] = places_df['sp_id'].map(
        lambda sp_id: partition_map.get(spid_to_graphid.get(sp_id, 0), 0)
    )
    
    logger.info(f"Loaded {len(places_df)} places with process assignments")
    logger.info(f"Process distribution:\n{places_df['process'].value_counts().sort_index()}")
    
    # Save partitioned places
    places_output = output_dir / "places_partition.parquet"
    places_df.to_parquet(places_output)
    logger.info(f"Saved partitioned places to {places_output}")
    
    # Load persons and add process assignments based on home location
    logger.info("Loading persons file...")
    conn = duckdb.connect(database=':memory:')
    persons_path = str(persons_file / "*/*.parquet")
    
    persons = conn.execute(f"""
        SELECT * FROM read_parquet('{persons_path}', hive_partitioning=1)
        WHERE CAST(Imputation AS INTEGER) = 1
    """).df()
    
    # Map person's home location to partition ID
    persons['process'] = persons['sp_hh_id'].map(
        lambda sp_id: partition_map.get(spid_to_graphid.get(sp_id, 0), 0)
    )
    
    logger.info(f"Loaded {len(persons)} persons with process assignments")
    logger.info(f"Process distribution:\n{persons['process'].value_counts().sort_index()}")
    
    # Save partitioned persons by partition (Hive-style)
    for partition_id in range(8):
        partition_persons = persons[persons['process'] == partition_id]
        partition_dir = output_dir / f"Imputation=1" / f"partition={partition_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        partition_file = partition_dir / "part-0.parquet"
        partition_persons.to_parquet(partition_file)
        logger.info(f"Saved {len(partition_persons)} persons to partition {partition_id}")
    
    logger.info("Partitioning complete!")

if __name__ == "__main__":
    main()
```

### Run the Script

```bash
cd /Users/joncline/Documents/GitHub/casmsocial
python partition_data.py
```

### Output

```
data/partitioned_8proc/
├── places_partition.parquet
├── Imputation=1/
│   ├── partition=0/
│   │   └── part-0.parquet
│   ├── partition=1/
│   │   └── part-0.parquet
│   ├── ...
│   └── partition=7/
│       └── part-0.parquet
```

Each partition file contains agents assigned to that partition, with a `process` column indicating their MPI rank.

---

## Step 5: Create MPI-Aware Config

Create a new config file that uses the partitioned data:

```yaml
# config/enhanced_heat_risk_mpi.yaml
model.name: "casmsocial.heat_risk.enhanced_heat_risk_model.EnhancedHeatRiskModel"

# Use partitioned data files
places.file: "data/partitioned_8proc/places_partition.parquet"
persons.file: "data/partitioned_8proc/Imputation=1/partition=*/part-0.parquet"
activities.file: "data/processed/wake_county_30_v2/abm_inputs/activities.parquet"
Imputation: 1

# Heat Risk Data
environment.file: "data/processed/apache_flight/wake_county_30_v2/weather_at_places.parquet"
closest_cooling_center.file: "data/processed/apache_flight/wake_county_30_v2/closest_cooling_center.parquet"
cooling_centers_experiment.file: "data/processed/wake_county_30_v2/abm_inputs/experiments.parquet"
experiment_id: 1

# Other parameters
random.seed: 42
start.datetime: "2023-09-07 00:00:00"
duration.hours: 24
time.step.minutes: 15
timezone: "America/New_York"

# Keep parallelization disabled (using MPI instead of threading)
parallel.places.enabled: false
parallel.heat.enabled: false
```

---

## Step 6: Run with MPI

### Local Test (4 processes)

```bash
cd /Users/joncline/Documents/GitHub/casmsocial

mpirun -n 4 python -m casmsocial config/enhanced_heat_risk_mpi.yaml
```

### HPC Cluster (8 processes)

```bash
# In submit_heat_risk_array.slurm, use:
mpirun -n 8 python -m casmsocial config/enhanced_heat_risk_mpi.yaml
```

### Expected Performance

- Serial (1 process): ~250 seconds
- **MPI 8 processes**: ~35 seconds (7x speedup)
- **MPI 16 processes**: ~18 seconds (14x speedup)

---

## Complete Workflow Summary

```bash
# 1. Generate graph
python -m casmsocial.graph_from_data \
  --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \
  --imputation 1 \
  --output-file data/wake_county_graph.txt \
  --map-file data/wake_county_graph_id_map.txt \
  --save-networkx

# 2. Install METIS (if needed)
# brew install metis (macOS)
# sudo apt-get install metis (Linux)

# 3. Run METIS partitioning
gpmetis data/wake_county_graph.txt 8

# 4. Create partitioned data
python partition_data.py

# 5. Run with MPI
mpirun -n 8 python -m casmsocial config/enhanced_heat_risk_mpi.yaml
```

---

## Troubleshooting

### METIS Not Found

```bash
# Check installation
which gpmetis

# If not found, install:
brew install metis  # macOS
sudo apt-get install metis  # Linux

# Or add to PATH if installed elsewhere
export PATH=/usr/local/bin:$PATH
```

### Graph File Format Issues

METIS expects format:
```
<num_vertices> <num_edges>
<neighbors_of_vertex_1>
<neighbors_of_vertex_2>
...
```

The `graph_from_data.py` script generates this automatically.

### Partition File Not Generated

Make sure `data/wake_county_graph.txt` exists:
```bash
ls -lh data/wake_county_graph.txt
```

If missing, run Step 1 again.

### MPI Execution Issues

Check MPI installation:
```bash
mpirun --version
```

On HPC, you may need to load MPI module:
```bash
module load mpi/mpich  # or equivalent on your system
```

---

## Performance Verification

After running with MPI, check the runtime:

```bash
# Extract from logs
grep "Simulation took" <log_file>

# Expected:
# Serial: ~250 seconds
# MPI-8: ~35 seconds (7x faster)
# MPI-16: ~18 seconds (14x faster)
```

