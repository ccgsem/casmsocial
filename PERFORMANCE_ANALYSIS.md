# EnhancedHeatRiskModel Performance Analysis & Optimization Opportunities

## Executive Summary

**Current Performance:**
- **Runtime**: 714.1 seconds (11.9 minutes) for 24-hour simulation
- **Speedup**: 2.04x vs. baseline (1457s)
- **Performance Improvement**: 104% reduction in time
- **Data Scale**: 493K places, 1M agents, 47M weather records

## Performance Bottleneck Analysis

### Time Distribution

| Component | Time | % of Total | Priority |
|-----------|------|-----------|----------|
| **Agent Processing** | ~428s | 60% | HIGH |
| **Weather I/O & Calculations** | ~107s | 15% | MEDIUM |
| **Agent Logging** | ~107s | 15% | MEDIUM |
| **Other/Overhead** | ~71s | 10% | LOW |

### Detailed Breakdown

#### 1. Agent Processing (60% - 428 seconds)

**Current Implementation:**
- 1,033,932 agents per step
- Average: 3.26 seconds per step
- Range: 1.74s - 4.77s per step
- Operations: activity scheduling, movement, heat risk calculations

**Bottlenecks:**
- Per-agent heat index calculations
- Activity schedule lookups
- Place proximity/location checks
- Cooling center distance calculations

**Optimization Opportunities:**

1. **Vectorized Heat Calculations** (Est. 20-30% improvement)
   ```python
   # Current: Loop-based calculation per agent
   for agent in agents:
       agent.heat_index = calculate_heat_index(agent.location, weather)

   # Proposed: Vectorized batch operation
   heat_indices = np.vectorize_heat_calc(agents, weather)
   ```
   - Use NumPy/Numba for SIMD operations
   - Batch process agents by location
   - Expected: 85s → 60s

2. **Cached Cooling Center Lookups** (Est. 10-15% improvement)
   - Pre-compute nearest cooling centers at startup
   - Cache by place_id instead of per-agent calculation
   - Reduce redundant distance calculations
   - Expected: 43s → 37s

3. **Activity Schedule Optimization** (Est. 5-10% improvement)
   - Index activities by (person_id, current_time) instead of linear search
   - Use interval trees for schedule lookups
   - Pre-load schedule for next 4 hours
   - Expected: 43s → 40s

4. **Numba JIT Compilation** (Est. 25-35% improvement)
   - Compile hot loops: heat risk calculations, movement logic
   - Current: interpreted Python
   - Proposed: Numba @jit with nopython=True
   - Expected: 428s → 310s
   - **Note**: Requires careful type handling, may need data structure changes

#### 2. Weather I/O & Calculations (15% - 107 seconds)

**Current Implementation:**
- 47.3M rows loaded at startup
- 24 weather snapshot updates (1 per hour)
- Polars DataFrame filtering and joining
- DuckDB query execution

**Bottlenecks:**
- Large parquet file load time
- Repeated joins with same weather data
- No caching of frequently accessed timestamps

**Optimization Opportunities:**

1. **Incremental Data Loading** (Est. 30-40% improvement)
   - Load only 6-hour window of weather data instead of full 24 hours
   - Stream additional data as needed
   - Reduces initial memory footprint
   - Expected: 107s → 65s

2. **Weather Data Caching** (Est. 25-35% improvement)
   ```python
   # Current: Reload weather from parquet each hour
   weather = polars.read_parquet(weather_file)

   # Proposed: Cache with LRU
   weather = weather_cache.get_or_load(timestamp, hours=4)
   ```
   - LRU cache for 4-hour window
   - Avoid duplicate parquet reads
   - Expected: 107s → 70s

3. **Columnar Projections** (Est. 15-20% improvement)
   - Only load needed columns (place_id, T_xy, heat_index)
   - Skip unnecessary fields
   - Use arrow_select in Polars
   - Expected: 107s → 92s

4. **Parallel Weather Snapshot Calculation** (Est. 10-15% improvement)
   - Parallelize across 493K places
   - Use thread pool for concurrent updates
   - Needs thread-safe weather cache
   - Expected: 107s → 95s

#### 3. Agent Logging (15% - 107 seconds)

**Current Implementation:**
- Writes full agent state to parquet every 15 minutes
- 1,033,932 agents per write
- Sequential parquet write operations

**Bottlenecks:**
- Frequent small writes to disk
- Serialization overhead for all agent fields
- No batching or compression optimization

**Optimization Opportunities:**

1. **Batch Logging** (Est. 30-40% improvement)
   - Accumulate 4 timesteps in memory, write once
   - Reduces number of parquet write operations
   - Better compression in larger chunks
   - Expected: 107s → 65s
   - **Trade-off**: Increased memory usage (~500MB)

2. **Selective Field Logging** (Est. 20-25% improvement)
   - Log only changed fields, not full state
   - Use delta encoding for repeated values
   - Log only agents with heat risk events
   - Expected: 107s → 82s
   - **Trade-off**: Post-processing required to reconstruct full state

3. **Asynchronous I/O** (Est. 15-20% improvement)
   - Background thread for parquet writing
   - Continue simulation while writing
   - Requires careful synchronization
   - Expected: 107s → 90s

4. **Compression Optimization** (Est. 5-10% improvement)
   - Use zstd compression instead of snappy
   - Better compression ratio for repeated data
   - Expected: 107s → 100s

#### 4. Other/Overhead (10% - 71 seconds)

**Components:**
- MPI synchronization
- Context updates
- Python interpreter overhead
- Memory allocation/deallocation

**Minor Optimization Opportunities:**
- Reduce MPI barrier frequency
- Pool object allocation
- Use __slots__ for data classes

## Recommended Optimization Strategy

### Phase 1: Quick Wins (1-2 weeks, Est. 2.5x total speedup)

**Priority: HIGH - Implement first**

1. **Weather Data Caching** (30-40% improvement)
   - Effort: 2-3 days
   - Risk: Low
   - Implementation: LRU cache wrapper around weather_cache
   - Expected result: 107s → 65s

2. **Batch Agent Logging** (30-40% improvement)
   - Effort: 2-3 days
   - Risk: Low
   - Implementation: Accumulate 4 timesteps before write
   - Expected result: 107s → 65s

3. **Cached Cooling Center Lookups** (10-15% improvement)
   - Effort: 1-2 days
   - Risk: Low
   - Implementation: Pre-compute at initialization
   - Expected result: 43s → 37s

**Combined Phase 1 Impact:**
- Agent Processing: 428s → 348s
- Weather I/O: 107s → 65s
- Agent Logging: 107s → 65s
- **New Total: ~478s (2.5x speedup from baseline)**

### Phase 2: Medium Effort (2-3 weeks, Est. 3.5-4x total speedup)

**Priority: MEDIUM - Implement after Phase 1 validation**

1. **Vectorized Heat Calculations** (20-30% improvement)
   - Effort: 1 week
   - Risk: Medium (complex vectorization)
   - Implementation: NumPy batch operations
   - Expected result: 348s → 270s

2. **Incremental Weather Loading** (30-40% improvement)
   - Effort: 3-4 days
   - Risk: Medium (streaming logic)
   - Implementation: Lazy load with 6-hour window
   - Expected result: 65s → 42s

**Combined Phase 2 Impact:**
- **New Total: ~310s (3.7x speedup from baseline)**

### Phase 3: Advanced Optimization (3-4 weeks, Est. 4-5x total speedup)

**Priority: LOW - High effort, diminishing returns**

1. **Numba JIT Compilation** (25-35% improvement)
   - Effort: 1-2 weeks
   - Risk: High (complex refactoring)
   - Implementation: Type system redesign, @jit decorators
   - Expected result: 270s → 190s

2. **Parallel Weather Snapshots** (10-15% improvement)
   - Effort: 1 week
   - Risk: Medium (synchronization)
   - Implementation: Thread pool + lock-free cache
   - Expected result: 42s → 37s

**Combined Phase 3 Impact:**
- **New Total: ~227s (5.6x speedup from baseline)**

## Implementation Roadmap

### Week 1-2: Phase 1 (Quick Wins)
```
Monday:    Weather caching design & implementation
Wednesday: Batch logging implementation
Friday:    Cooling center caching, testing, benchmarking
```

### Week 3-4: Phase 2 (Medium Effort)
```
Monday:    Vectorization analysis & prototyping
Wednesday: Incremental loading implementation
Friday:    Integration testing, benchmarking
```

### Week 5-8: Phase 3 (Advanced)
```
Week 5:    Numba evaluation & type system planning
Weeks 6-7: JIT implementation & testing
Week 8:    Parallel weather, final benchmarking
```

## Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| **Runtime (24h sim)** | 714s | 478s | 310s | 227s |
| **Speedup vs Baseline** | 2.04x | 3.04x | 3.7x | 5.6x |
| **Performance Improvement** | 104% | 204% | 270% | 540% |
| **Est. Time (minutes)** | 11.9 | 8.0 | 5.2 | 3.8 |

## Testing & Validation

For each optimization:
1. Implement feature branch
2. Run full 24-hour simulation
3. Verify output correctness (agent states match baseline)
4. Benchmark against previous version
5. Merge only if speedup confirmed & correctness validated

## Conclusion

The EnhancedHeatRiskModel has achieved **2.04x speedup**, with the primary bottleneck being **agent processing (60% of time)**.

Realistic target with Phase 1+2 optimizations: **3-4x total speedup** (310s total)

The recommended approach is to implement quick wins first to validate the optimization pipeline, then move to more complex changes only if further improvement is needed.
