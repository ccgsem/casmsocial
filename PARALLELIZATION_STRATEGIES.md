# Parallelization Strategies for casmsocial Heat Risk Model

## Executive Summary

Current threading-based parallelization (ThreadPoolExecutor) causes **23.7x slowdown** due to lock contention and GIL bottlenecks. The METIS-based graph partitioning approach offers a fundamentally different strategy: **MPI-based distributed parallelization across processes** rather than threads within a process.

---

## Current State

### ✅ What We Have (Implemented)
- ThreadPoolExecutor-based parallelization for place updates
- Configurable batch sizing and worker counts
- All parameters are now properly controllable via YAML config
- Infrastructure is flexible and can be toggled on/off

### ❌ Performance Problem
- **Threading overhead > computational benefits**
- GIL contention on Python operations
- Lock contention on shared MPI data structures
- Context switching costs exceed speedup

---

## Recommended Strategy: METIS-Based MPI Distribution

### Why METIS Partitioning is Better

**Instead of threading within a process:**
```
Single Process + ThreadPoolExecutor (current approach)
├─ GIL bottleneck (Python threads)
├─ Lock contention on shared data
├─ Context switching overhead
└─ Result: 23.7x slowdown
```

**Use METIS to distribute across processes:**
```
Multiple Processes (MPI) with METIS-optimized placement
├─ No GIL (separate Python interpreters)
├─ Minimal inter-process communication (METIS optimizes this)
├─ Each process runs independently
└─ Result: Near-linear speedup
```

### How METIS Partitioning Works

1. **Build Communication Graph:**
   - Vertices = Places (380,396 in Wake County dataset)
   - Edges = Agent movements between places
   - Example: If Person moves A→B→C, create edges A-B, A-C, B-C

2. **Run METIS Partitioning:**
   - Input: Communication graph
   - Output: Process assignment for each place
   - Goal: Minimize edges crossing partition boundaries

3. **Result:**
   - Agents mostly stay within their process's places
   - Minimal MPI communication required
   - Excellent scaling potential

---

## Implementation Roadmap

### Phase 1: Data Preparation (1-2 hours)
```bash
# 1. Use existing graph_from_data.py to create communication graph
python scripts/graph_from_data.py \
  --agents data/persons.parquet \
  --activities data/activities.parquet \
  --output wake_county_comm_graph

# 2. Run METIS partitioning (for N processes)
gpmetis wake_county_comm_graph.graph <num_processes>
# Output: wake_county_comm_graph.graph.part.<num_processes>

# 3. Add partition data to input files
python scripts/processes_from_partition.py \
  --agents data/persons.parquet \
  --places data/places.parquet \
  --partition wake_county_comm_graph.graph.part.8 \
  --id-mappings wake_county_comm_graph.id_map \
  --output-dir data/partitioned_8proc/
```

### Phase 2: Code Integration (2-4 hours)
```python
# In casmpop.py, read process assignments from partition data
# Instead of:
#   random assignment or geography-based
# Do:
#   self.process_assignment = persons_df['process'].tolist()

# METIS minimizes edges crossing process boundaries
# → Reduces MPI communication overhead
# → Each process mainly deals with local places
```

### Phase 3: MPI Execution (existing infrastructure)
```bash
# Run with optimized MPI distribution
mpirun -n 8 python -m casmsocial config/enhanced_heat_risk_example.yaml
# Each rank handles ~47,000 places (380k / 8)
# Most agent movements are intra-process
```

---

## Expected Performance

### Baseline Comparison
| Configuration | Runtime | Speedup |
|--------------|---------|---------|
| Serial (1 process) | 250s | 1.0x |
| Current threading (23.7x slower) | 5,929s | 0.04x |
| **METIS MPI (8 processes)** | **~35s** | **~7x** |
| **METIS MPI (16 processes)** | **~18s** | **~14x** |

### Why METIS Wins
1. **No GIL contention** - separate Python interpreters per process
2. **Minimal communication** - METIS optimizes edge cuts
3. **Load balancing** - METIS balances partition sizes
4. **Scaling** - Near-linear speedup expected

---

## Technical Details: METIS Approach

### Graph Construction
```python
# Pseudo-code for building communication graph
graph = {
    'vertices': list(places),  # 380,396 places
    'edges': []  # Agent movements
}

for person in persons:
    places_visited = person.schedule_places
    # Create edges between all pairs of visited places
    for i, place_a in enumerate(places_visited):
        for place_b in places_visited[i+1:]:
            graph['edges'].append((place_a, place_b))

# Result: Graph where edges represent communication needs
# METIS partitions to minimize edges crossing process boundaries
```

### Partition Result Example
```
Process 0: Places [1, 5, 12, 45, 67, ...]
Process 1: Places [2, 8, 15, 50, 72, ...]
...
Process 7: Places [3, 9, 18, 52, 75, ...]

When person moves from Place 45 → Place 67:
  → Both in Process 0 → No MPI communication needed! ✓

When person moves from Place 67 → Place 15:
  → Process 0 → Process 1 → Requires MPI send/recv (minimized by METIS)
```

---

## Advantages Over Current Threading

| Aspect | Threading (Current) | METIS MPI |
|--------|-------------------|-----------|
| GIL bottleneck | ✗ Major issue | ✓ No GIL |
| Lock contention | ✗ High | ✓ Low (MPI) |
| Scaling | ✗ Poor (23.7x slower!) | ✓ Near-linear |
| Communication | N/A (shared memory) | ✓ Optimized |
| Scalability | Limited | Unlimited (up to N processes) |
| Fault tolerance | Limited | Better (independent processes) |
| Memory overhead | Lower | Higher (separate interpreters) |

---

## Implementation Steps

### Step 1: Verify Graph Creation Tools Exist
```bash
ls -la scripts/graph_from_data.py
ls -la scripts/processes_from_partition.py
```

### Step 2: Install METIS (if not present)
```bash
# macOS
brew install metis

# Linux (Ubuntu/Debian)
sudo apt-get install libmetis-dev

# Or build from source:
# http://glaros.dtc.umn.edu/gkhome/metis/metis/download
```

### Step 3: Create Wake County Partition
```bash
cd scripts/
python graph_from_data.py \
  --agents ../data/persons.parquet \
  --activities ../data/activities.parquet \
  --output wake_county_graph

# Run METIS for 8 partitions (adjust as needed)
gpmetis wake_county_graph.graph 8

python processes_from_partition.py \
  --agents ../data/persons.parquet \
  --places ../data/places.parquet \
  --partition wake_county_graph.graph.part.8 \
  --output ../data/partitioned_wake_county_8proc/
```

### Step 4: Update Config
```yaml
# config/enhanced_heat_risk_example.yaml
persons.file: "data/partitioned_wake_county_8proc/persons_partition.parquet"
places.file: "data/partitioned_wake_county_8proc/places_partition.parquet"

# Disable threading (keep disabled)
parallel.places.enabled: false
parallel.heat.enabled: false
```

### Step 5: Run with MPI
```bash
# Test with 8 processes
mpirun -n 8 python -m casmsocial config/enhanced_heat_risk_example.yaml

# Expected: ~35 seconds vs 250 seconds = 7x faster
```

---

## Comparison: Threading vs METIS

### Threading Approach (Current - FAILS)
```
┌─────────────────────────────────────┐
│     Single Python Process           │
│  ┌──────────────────────────────┐   │
│  │ Main Thread (GIL holder)     │   │
│  │  - Controls interpreter      │   │
│  │  - Serializes all Python ops │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ Worker Thread 1 (blocked)    │   │
│  │ Worker Thread 2 (blocked)    │   │
│  │ Worker Thread 3 (blocked)    │   │
│  │ Worker Thread 4 (blocked)    │   │
│  └──────────────────────────────┘   │
│  Result: 23.7x SLOWER               │
└─────────────────────────────────────┘
```

### METIS MPI Approach (Recommended)
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Process 0   │  │  Process 1   │  │  Process 2   │  │  Process 3   │
│              │  │              │  │              │  │              │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │Python    │ │  │ │Python    │ │  │ │Python    │ │  │ │Python    │ │
│ │Interp 0  │ │  │ │Interp 1  │ │  │ │Interp 2  │ │  │ │Interp 3  │ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
│  Places:     │  │  Places:     │  │  Places:     │  │  Places:     │
│  ~95k        │  │  ~95k        │  │  ~95k        │  │  ~95k        │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │                │                │                │
       └────────────────┼────────────────┼────────────────┘
                   MPI Communication
                 (minimized by METIS)
Result: 7-10x FASTER
```

---

## Recommendation

**Immediately:** Keep threading disabled (current state is good)

**For Production HPC Runs:** Implement METIS partitioning
1. Creates balanced communication graph
2. Distributes load across processes
3. Minimal inter-process communication
4. Near-linear scaling with number of processes
5. Suitable for 750-job array on HPC cluster

**For Development:** Current serial execution is fast enough (250s for 24-hour simulation)

---

## Resources

- **METIS Documentation:** http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
- **METIS Manual:** http://glaros.dtc.umn.edu/gkhome/metis/metis/download
- **Partition scripts:** `scripts/graph_from_data.py`, `scripts/processes_from_partition.py`
- **casmsocial METIS guide:** `Using_METIS_For_Partitioning.md`

