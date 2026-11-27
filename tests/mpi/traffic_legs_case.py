"""Run the MVP day with routed legs on the vehicle mobsim and print a JSON summary from rank 0.

Usage: traffic_legs_case.py <jam_edge_capacity_vph>

On two ranks the MVP partition puts place 200 on rank 1 and places 100 and
300 on rank 0, so person 2's vehicles cross ranks and their arrivals must be
routed back to the rank holding person 2. A capacity <= 0 leaves edge 1 as
built; otherwise edge 1 (person 1's morning route) gets that capacity.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402
from mpi4py import MPI  # noqa: E402
from repast4py.parameters import init_params  # noqa: E402

from casmsocial.casmpop import CasmPop  # noqa: E402
from casmsocial.person import Person  # noqa: E402
from scripts.create_mvp_ducklake import create_mvp_tables  # noqa: E402


def main() -> None:
    jam_capacity = float(sys.argv[1])
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    conn = duckdb.connect(":memory:")
    create_mvp_tables(conn)
    data_path = Path(tempfile.mkdtemp())

    def use_test_data_resources(model: CasmPop) -> None:
        model.data_path = data_path
        model.conn = conn

    CasmPop._set_data_resources = use_test_data_resources
    CasmPop.register_activity_names([])
    CasmPop.register_planned_activity_names([])

    params = init_params(str(Path(__file__).resolve().parents[2] / "config" / "mvp.yaml"), "{}")
    params.update({
        "roads.enabled": True,
        "roads.nodes.file": "casmsocial_mvp.road_nodes",
        "roads.edges.file": "casmsocial_mvp.road_edges",
        "roads.place_snap.file": "casmsocial_mvp.place_road_snap",
        "behavior.engine": "schedule",
        "behavior.llm.enabled": False,
        "communication.enabled": False,
        "observers.agent_log.enabled": False,
        "observers.behavior_log.enabled": False,
        "time.step.minutes": 15,
        "traffic.enabled": True,
        "partition.table": "partitions.mvp_two_rank_place_partitions" if size == 2 else "",
    })
    model = CasmPop(comm, params)
    model.build_context()
    network = model.traffic.mobsim.network
    if jam_capacity > 0:
        network.edges[1] = dataclasses.replace(network.edges[1], capacity_vph=jam_capacity)

    timeline = []  # (minute_of_day, person_id, place_id, rank)
    for _ in range(24 * 60 // model.time_step_minutes):
        model.step()
        for person in model.context.agents(agent_type=Person.TYPE):
            timeline.append((model.cal.minute_of_day, person.id, person.place_id, rank))

    summary = model.traffic.global_summary()
    records = [
        (r.person_id, r.trip_id, r.depart_s, r.arrive_s, r.scheduled_arrival_s, r.n_migrations, rank)
        for r in model.traffic.trip_records
    ]
    all_records = comm.gather(records, root=0)
    all_timeline = comm.gather(timeline, root=0)
    vehicle_node_rank = model.traffic.mobsim.node_rank

    if rank == 0:
        print(
            json.dumps({
                "ranks": size,
                "summary": summary,
                "node_rank": {str(k): v for k, v in sorted(vehicle_node_rank.items())},
                "trips": sorted(item for part in all_records for item in part),
                "timeline": sorted(item for part in all_timeline for item in part),
            })
        )


if __name__ == "__main__":
    main()
