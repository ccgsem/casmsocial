"""Run the vehicle mobsim on N ranks and print a JSON summary from rank 0.

Usage: traffic_mobsim_case.py <grid_width> <n_trips> <sync_window_s> <strategic_share>

Every rank builds the same grid and trip list; each trip is spawned on rank 0
regardless of where it starts, so the first exchange also exercises
forwarding vehicles to the rank that owns their origin node.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mpi4py import MPI  # noqa: E402

from casmsocial.traffic import VehicleMobsim, assign_node_ranks, boundary_edge_share  # noqa: E402
from tests.traffic_fixtures import grid_network, split_columns  # noqa: E402


def main() -> None:
    width = int(sys.argv[1])
    n_trips = int(sys.argv[2])
    window = float(sys.argv[3])
    share = float(sys.argv[4])

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    network = grid_network(width, width, capacity_vph=400)
    node_rank = assign_node_ranks(network, split_columns(network, width, size))
    mobsim = VehicleMobsim(comm, network, node_rank, sync_window_s=window, strategic_share=share, seed=11)

    rng = random.Random(5)
    trips = []
    for person in range(n_trips):
        origin, dest = rng.randrange(width * width), rng.randrange(width * width)
        while dest == origin:
            dest = rng.randrange(width * width)
        trips.append((person, origin, dest, rng.uniform(0, 1800)))

    if rank == 0:
        for person, origin, dest, depart in trips:
            mobsim.spawn(person, 0, origin, dest, depart)

    comm.Barrier()  # spawning (routing) on rank 0 is not part of the timed loop
    started = time.perf_counter()
    arrivals = []
    for tick_end in range(900, 4 * 3600 + 1, 900):  # 15-minute casmsocial ticks
        mobsim.advance_to(float(tick_end))
        arrivals.extend(mobsim.drain_arrivals())
    elapsed = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)

    stats = mobsim.global_stats()
    profile = mobsim.gather_travel_time_profile()
    all_arrivals = comm.gather([(a.person_id, a.arrive_s, a.rank, a.n_migrations) for a in arrivals], root=0)

    if rank == 0:
        flat = sorted(item for part in all_arrivals for item in part)
        print(
            json.dumps({
                "ranks": size,
                "boundary_edge_share": boundary_edge_share(network, node_rank),
                "stats": stats,
                "arrivals": [[p, t] for p, t, _, _ in flat],
                "arrival_ranks": sorted({r for _, _, r, _ in flat}),
                "max_vehicle_migrations": max((m for *_, m in flat), default=0),
                "profile_rows": len(profile),
                "profile_edges_unique": len(profile) == len({(r["edge_id"], r["bin_start_s"]) for r in profile}),
                "profile_traversals": sum(r["n_traversals"] for r in profile),
                "elapsed_s": elapsed,
            })
        )


if __name__ == "__main__":
    main()
