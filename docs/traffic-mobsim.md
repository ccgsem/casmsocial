# Vehicle mobsim prototype (`casmsocial.traffic`)

Status: **prototype**. It runs standalone, under MPI, and behind routed
`CasmPop` legs when `traffic.enabled` is set (off by default).

## Why

`casmsocial` legs carry a routed free-flow travel time from `RoadNetwork`.
There is no congestion, and a person on a leg has no position. This package
adds a distributed, congestion-aware vehicle simulation that legs can hand
trips to. It combines two published approaches:

| | MATSim | BATSim (smart-facility/BATSim) | This prototype |
|---|---|---|---|
| Loop | Repeat the day until plans settle | One within-day pass | One pass per run; runs chain through a travel-time profile |
| Link model | FIFO queues with spillback | BPR on instantaneous link count | BPR on implied flow (default) or BATSim's form |
| Adaptation | Score, then replan between iterations | Strategic agents reroute at nodes | BATSim rule, optionally priced with live local loads |
| Parallelism | JVM threads | Repast HPC, grid partition, agents migrate | repast4py, place-aligned partition, vehicles migrate |

## Design

**Vehicles, not persons, migrate.** `Vehicle` (`traffic/vehicle.py`) is a
small repast4py agent: person id, trip id, remaining edge path, current edge,
next event time, and strategy state. A `Person` keeps its `rank_place_id`
anchor during a leg, as casmsocial already does for communication. Vehicles
live in their own `SharedContext`, so casmsocial's place projections never see
them.

**Edge ownership makes link loads local.** Each node is owned by one rank.
A vehicle stays on the rank that owns the start node of its current edge
until it reaches the end node, then migrates if that node is owned elsewhere.
Each edge's vehicle count is therefore complete on one rank, so no global
link-load reduction is needed. This is BATSim's invariant.

**Partition follows places.** `assign_node_ranks` gives each node the
majority rank of the places snapped to it. Unlabelled nodes take the rank of
the nearest labelled node by hop count. Trips start on the rank that holds
the traveller.

**Event-driven time.** A vehicle's exit time is fixed with BPR when it enters
an edge. `advance_to(t_end)` loops:

1. `Allreduce(MIN)` of `[next local event, -pending migrations]`. One
   collective gives both the global next event time and whether anyone has
   vehicles to send.
2. If any rank has migrations, `context.move_agents` runs, then step 1 again.
3. Every rank processes its events before `global_next + sync_window_s`.

`advance_to` runs inside one casmsocial tick, so the outer model keeps its
15-minute step. `sync_window_s` trades exactness for collectives: a migrating
vehicle can enter its next edge up to one window late.

**Link model.** BATSim's `ff·(1 + 0.15·(n/c)^4)` with the raw count `n`
explodes under queues: 50 cars give about 50× free flow. The default
`implied_flow` model reads `n` vehicles each spending `t` seconds as flow
`n/t`, and solves `t = ff·(1 + α·(n/(c·t))^β)` by Newton. `link_model="batsim"`
keeps the original. Neither model has spillback.

**Strategic rerouting** uses BATSim's rule at each node: reroute when
`x1·cos α + x2·sin α > θ`, where `x1` is elapsed time over free-flow time so
far and `x2` is the next edge's saturation. The new path avoids the next
edge. With `reroute_with_live_loads` (default) it also prices edges this rank
holds at their current BPR time. `strategic_share` picks strategic trips with
a hash of `(seed, person, trip)`, so the choice doesn't depend on rank count.

**MATSim-style feedback.** Each rank records per-edge mean traversal times in
`record_bin_s` bins. `gather_travel_time_profile()` collects them, and the
rows never collide because edges have one owner.
`TrafficNetwork.set_travel_time_profile(rows)` makes the next run's router use
them. Each casmdb experiment can therefore be one iteration of the day.

## Usage

```python
from mpi4py import MPI
from casmsocial.traffic import TrafficNetwork, VehicleMobsim, assign_node_ranks

network = TrafficNetwork.from_tables(nodes, edges, place_snaps)    # roads.* artifacts
node_rank = assign_node_ranks(network, place_rank)                 # place_id -> rank
mobsim = VehicleMobsim(MPI.COMM_WORLD, network, node_rank, sync_window_s=5.0, strategic_share=0.1)

mobsim.spawn_between_places(person_id, trip_id, origin_place, dest_place, depart_s)
for tick_end in range(900, 86_400 + 1, 900):
    mobsim.advance_to(tick_end)            # collective
    for arrival in mobsim.drain_arrivals():
        ...                                # hand the person to their next activity
```

Edge tables may carry `capacity_vph`. Otherwise the capacity comes from
`road_type` (`DEFAULT_CAPACITY_VPH`), which is a placeholder.

## Legs in CasmPop

Set `traffic.enabled: true` together with `roads.enabled: true`:

```yaml
roads.enabled: true            # routed legs, as before
traffic.enabled: true          # carry them on the vehicle mobsim
traffic.link_model: 'implied_flow'
traffic.sync_window_s: 5.0
traffic.strategic_share: 0.0
```

`LegTrafficCoupler` (`traffic/legs.py`) joins the two:

* **Spawn.** When a person starts a leg in `roads.mode` that has origin and
  destination nodes, `Person._transition_to_leg_at_minute` hands it to the
  coupler. The vehicle departs when the previous activity ended, and the trip
  id is that minute, so it is the same on any rank count. Unrouted legs and
  other modes behave as before.
* **Hold.** `Person.move_at` returns early for a held person, so neither the
  environment step nor a behavior engine can move them into the next activity
  while the car is on the road.
* **Release.** At the start of each tick, before anyone moves,
  `CasmPop.advance_traffic` runs the mobsim to the tick time and releases
  people whose car arrived. A car that arrives early changes nothing: the
  person still enters at the scheduled start. A late car means the person
  enters on the first tick after arrival.
* **Ranks.** A held person keeps the origin place as `rank_place_id`, so they
  stay on that rank while only the vehicle migrates. Vehicles carry a
  `home_rank`, and `VehicleMobsim.route_arrivals_home` sends each arrival
  there with one `alltoall` per tick.
* **Records.** Each completed leg is a `TripRecord` with departure, scheduled
  and actual arrival, delay, free-flow time, reroutes and migrations.
  `trip_rows()` returns them as dicts, and the end of the run logs
  `global_summary()` (trips, delayed trips, total delay).

Known approximation: a leg is found during the tick it starts in, after the
mobsim has already run to that tick. Its car keeps its true departure time,
but it sees link loads as of the tick, so congestion can lag departures by up
to one tick. Finding departures one tick ahead would remove this, at the cost
of a second plan scan per person.

## What the tests show

`tests/test_traffic_mobsim.py` (single rank) and
`tests/test_traffic_mobsim_mpi.py` (1, 2 and 3 ranks, 4,000 trips, 12×12 grid):

* Every trip arrives once. Profile rows are unique per edge and bin, and
  traversals equal edge entries.
* 2- and 3-rank runs match the 1-rank mean trip time within 0.5%, and over
  95% of trips match within 2%.
* `tests/test_traffic_legs.py` runs the MVP day with traffic on. Free-flow
  legs keep the schedule. Squeezing one edge's capacity delays one person
  past their 09:00 start, and they stay in transit until the tick after their
  car arrives. `tests/test_traffic_legs_mpi.py` runs the same day on 2 ranks,
  where person 2's cars cross ranks. The trips and every person's place at
  every tick match the 1-rank run.
* A 30% strategic share cuts mean trip time on a bottleneck by more than 5%.
  A 100% share herds drivers onto the same detours: more than twice the link
  entries and longer trips. This is the effect BATSim models.

Benchmark on this Mac (4 performance cores, Python 3.12, 50×50 grid,
40,000 trips over 4 h), timing only the `advance_to` loop:

| Ranks | 1 s window | 10 s window |
|---|---|---|
| 1 | 4.7 s | 4.5 s |
| 2 | 3.3 s | 2.8 s |
| 4 | 3.8 s | 2.2 s |

Routing all 40,000 trips at spawn on one rank took about 24 s, several times
the mobsim loop itself.

## Open items before Wake County

1. **Routing cost.** One A* per trip in pure Python will not scale to ~4M
   Wake trips on a county network. Route on the spawning rank, cache by
   (origin node, destination node, time bin), and consider a compiled router
   (contraction hierarchies).
2. **CasmPop integration.** Done for routed legs (see *Legs in CasmPop*).
   Still to do:
   * Look ahead one tick for departures.
   * Decide what a person does when the car arrives after the next activity
     has ended. Today they stay off-place until a later activity starts.
   * Let behavior engines react to the delay. Delay is recorded but no engine
     reads it yet.
3. **In-transit heat exposure.** Vehicles know their current edge, so the
   Wake model can look up weather along the route instead of skipping
   in-transit agents.
4. **Capacities.** Wake edges need lanes and capacity from OSM tags, or a
   calibration against NCDOT counts.
5. **Spillback.** Add MATSim-style queues only if BPR proves too optimistic
   in validation. This needs per-edge storage limits and ordered link exits.
6. **Outputs.** Add a casmdb channel for the link travel-time profile and the
   `TripRecord` rows (delay, reroutes), and a casmservice map layer for
   vehicles.
