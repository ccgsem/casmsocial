from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow.dataset as ds
from mpi4py import MPI
from repast4py.parameters import init_params

from casmsocial.casmpop import CasmPop
from casmsocial.model import Model
from casmsocial.person import Person
from casmsocial.road_network import RoadNetwork
from scripts.create_mvp_ducklake import create_mvp_tables
from scripts.validate_mvp_routed_plans import build_routed_mvp_plan_report, validate_routed_mvp_plans


def _read_hive_dataset(path):
    return ds.dataset(path, format="parquet", partitioning="hive").to_table()


def test_create_mvp_tables_writes_two_rank_partition_table():
    conn = duckdb.connect(":memory:")
    try:
        create_mvp_tables(conn)

        households = conn.execute("""
            SELECT sp_id, hh_size
            FROM rti_synth_pop_v2_dmv_100.hh
            ORDER BY sp_id
            """).fetchall()
        rows = conn.execute("""
            SELECT imputation, n_ranks, rank, place_id
            FROM partitions.mvp_two_rank_place_partitions
            ORDER BY place_id
            """).fetchall()
        rank_count = conn.execute("""
            SELECT COUNT(DISTINCT rank)
            FROM partitions.mvp_two_rank_place_partitions
            WHERE imputation = 1 AND n_ranks = 2
            """).fetchone()[0]

        assert households == [(100, 1), (200, 1)]
        assert rows == [(1, 2, 0, 100), (1, 2, 1, 200), (1, 2, 0, 300)]
        assert rank_count == 2
    finally:
        conn.close()


def test_create_mvp_tables_writes_routable_road_artifacts():
    conn = duckdb.connect(":memory:")
    try:
        create_mvp_tables(conn)

        nodes = conn.execute("""
            SELECT node_id, x, y
            FROM rti_synth_pop_v2_dmv_100.road_nodes
            ORDER BY node_id
            """).arrow().read_all().to_pylist()
        edges = conn.execute("""
            SELECT edge_id, from_node_id, to_node_id, length_m, travel_time_min, mode, road_type
            FROM rti_synth_pop_v2_dmv_100.road_edges
            ORDER BY edge_id
            """).arrow().read_all().to_pylist()
        snaps = conn.execute("""
            SELECT place_id, road_node_id
            FROM rti_synth_pop_v2_dmv_100.place_road_snap
            ORDER BY place_id
            """).arrow().read_all().to_pylist()

        road_network = RoadNetwork.from_tables(nodes, edges, snaps)
        route = road_network.route_between_places(100, 300)

        assert len(nodes) == 3
        assert len(edges) == 6
        assert snaps == [
            {"place_id": 100, "road_node_id": 1},
            {"place_id": 200, "road_node_id": 2},
            {"place_id": 300, "road_node_id": 3},
        ]
        assert route is not None
        assert route.origin_node_id == 1
        assert route.destination_node_id == 3
        assert route.distance_m == 5000.0
        assert route.travel_time_min == 12
    finally:
        conn.close()


def test_mvp_config_runs_and_writes_behavior_log(tmp_path, monkeypatch):
    params = init_params("config/mvp.yaml", "{}")
    params["observers.output_dir"] = str(tmp_path)
    params["observers.agent_log_file"] = "mvp_agent_log.parquet"
    params["observers.behavior_log_file"] = "mvp_behavior_log.parquet"

    conn = duckdb.connect(":memory:")
    create_mvp_tables(conn)

    def use_test_data_resources(model: CasmPop) -> None:
        model.data_path = tmp_path
        model.conn = conn

    previous_model = Model.get_model()
    previous_behavior_engine = Person.getBehaviorEngine()
    previous_activity_names = CasmPop.get_activity_names()
    previous_planned_activity_names = CasmPop.get_planned_activity_names()
    previous_activities_data_type = CasmPop._CasmPop__activities_data_type

    monkeypatch.setattr(CasmPop, "_set_data_resources", use_test_data_resources)
    CasmPop.register_activity_names([])
    CasmPop.register_planned_activity_names([])
    CasmPop._CasmPop__activities_data_type = None

    try:
        model = CasmPop(MPI.COMM_SELF, params)
        model.start()

        agent_log_path = tmp_path / "mvp_agent_log.parquet"
        behavior_log_path = tmp_path / "mvp_behavior_log.parquet"
        assert list(agent_log_path.rglob("*.parquet"))
        assert list(behavior_log_path.rglob("*.parquet"))

        behavior_table = _read_hive_dataset(behavior_log_path)
        assert behavior_table.num_rows > 0
        assert {
            "run_id",
            "random_seed",
            "tick",
            "rank",
            "agent_id",
            "place_id",
            "last_decision",
            "last_llm_summary",
            "last_memory_event_type",
            "last_plan_adjustment_requested_kind",
            "last_plan_adjustment_applied",
            "safety_signal",
            "social_signal",
            "obligation_signal",
            "schedule_signal",
            "reply_signal",
        }.issubset(set(behavior_table.column_names))
    finally:
        CasmPop.register_activity_names(previous_activity_names)
        CasmPop.register_planned_activity_names(previous_planned_activity_names)
        CasmPop._CasmPop__activities_data_type = previous_activities_data_type
        Person.registerBehaviorEngine(previous_behavior_engine)
        Model.theModel = previous_model
        conn.close()


def test_mvp_config_builds_routed_plans_with_fixture_tables(tmp_path, monkeypatch):
    params = init_params("config/mvp.yaml", "{}")
    params["roads.enabled"] = True
    params["roads.nodes.file"] = "rti_synth_pop_v2_dmv_100.road_nodes"
    params["roads.edges.file"] = "rti_synth_pop_v2_dmv_100.road_edges"
    params["roads.place_snap.file"] = "rti_synth_pop_v2_dmv_100.place_road_snap"
    params["behavior.engine"] = "schedule"
    params["behavior.llm.enabled"] = False
    params["communication.enabled"] = False
    params["observers.agent_log.enabled"] = False
    params["observers.behavior_log.enabled"] = False

    conn = duckdb.connect(":memory:")
    create_mvp_tables(conn)

    def use_test_data_resources(model: CasmPop) -> None:
        model.data_path = tmp_path
        model.conn = conn

    previous_model = Model.get_model()
    previous_behavior_engine = Person.getBehaviorEngine()
    previous_activity_names = CasmPop.get_activity_names()
    previous_planned_activity_names = CasmPop.get_planned_activity_names()
    previous_activities_data_type = CasmPop._CasmPop__activities_data_type

    monkeypatch.setattr(CasmPop, "_set_data_resources", use_test_data_resources)
    CasmPop.register_activity_names([])
    CasmPop.register_planned_activity_names([])
    CasmPop._CasmPop__activities_data_type = None

    try:
        model = CasmPop(MPI.COMM_SELF, params)
        model.build_context()

        summary = validate_routed_mvp_plans(model)
        report = build_routed_mvp_plan_report(
            summary,
            config_path=Path("config/mvp.yaml"),
            ducklake_path=tmp_path / "mvp.ducklake",
            generated_at="2026-05-31T00:00:00+00:00",
        )

        assert summary["agents"] == 2
        assert summary["legs"] == 4
        assert summary["routes"]["100->300"]["travel_time_min"] == 12
        assert report["validation"] == summary
        assert report["version"] == 1
    finally:
        CasmPop.register_activity_names(previous_activity_names)
        CasmPop.register_planned_activity_names(previous_planned_activity_names)
        CasmPop._CasmPop__activities_data_type = previous_activities_data_type
        Person.registerBehaviorEngine(previous_behavior_engine)
        Model.theModel = previous_model
        conn.close()
