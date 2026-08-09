from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmsocial.road_network import RoadNetwork
from scripts.build_road_network import (
    build_graph,
    compute_edge_travel_times,
    load_osm_extract,
    load_places,
    main,
    snap_places_to_graph,
)


def _write_osm_extract(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="38.0000" lon="-77.0000" />
  <node id="2" lat="38.0000" lon="-77.0100" />
  <node id="3" lat="38.0000" lon="-77.0200" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <tag k="highway" v="residential" />
    <tag k="maxspeed" v="30 mph" />
  </way>
  <way id="20">
    <nd ref="3" />
    <nd ref="1" />
    <tag k="highway" v="footway" />
  </way>
  <way id="30">
    <nd ref="2" />
    <nd ref="1" />
    <tag k="highway" v="service" />
    <tag k="oneway" v="yes" />
  </way>
</osm>
""",
        encoding="utf-8",
    )


def _write_places(path: Path) -> None:
    table = pa.table(
        {
            "sp_id": [100, 300],
            "longitude": [-77.0001, -77.0199],
            "latitude": [38.0001, 38.0001],
        }
    )
    pq.write_table(table, path)


def test_build_graph_extracts_drive_edges_from_osm_xml(tmp_path):
    osm_path = tmp_path / "roads.osm"
    _write_osm_extract(osm_path)

    graph = build_graph(load_osm_extract(osm_path), "drive")
    compute_edge_travel_times(graph)

    edge_pairs = [(edge.from_node_id, edge.to_node_id) for edge in graph.edges]
    assert set(graph.nodes) == {1, 2, 3}
    assert len(graph.edges) == 5
    assert (1, 2) in edge_pairs
    assert (2, 1) in edge_pairs
    assert (2, 3) in edge_pairs
    assert (3, 2) in edge_pairs
    assert (3, 1) not in edge_pairs
    assert all(edge.length_m > 0 for edge in graph.edges)
    assert all(edge.travel_time_min > 0 for edge in graph.edges)

    residential_edge = next(edge for edge in graph.edges if edge.road_type == "residential")
    speed_kph = residential_edge.length_m / residential_edge.travel_time_min * 60.0 / 1000.0
    assert speed_kph == pytest.approx(48.28, abs=0.1)


def test_snap_places_to_nearest_road_node(tmp_path):
    osm_path = tmp_path / "roads.osm"
    places_path = tmp_path / "places.parquet"
    _write_osm_extract(osm_path)
    _write_places(places_path)

    graph = build_graph(load_osm_extract(osm_path), "drive")
    snaps = snap_places_to_graph(load_places(places_path), graph)

    assert snaps == [
        {"place_id": 100, "road_node_id": 1},
        {"place_id": 300, "road_node_id": 3},
    ]


def test_cli_writes_parquet_artifacts_consumed_by_road_network(tmp_path):
    osm_path = tmp_path / "roads.osm"
    places_path = tmp_path / "places.parquet"
    nodes_path = tmp_path / "road_nodes.parquet"
    edges_path = tmp_path / "road_edges.parquet"
    snaps_path = tmp_path / "place_road_snap.parquet"
    report_path = tmp_path / "road_artifacts.json"
    _write_osm_extract(osm_path)
    _write_places(places_path)

    main(
        [
            "--osm-file",
            str(osm_path),
            "--places-file",
            str(places_path),
            "--nodes-out",
            str(nodes_path),
            "--edges-out",
            str(edges_path),
            "--snaps-out",
            str(snaps_path),
            "--report-out",
            str(report_path),
        ]
    )

    nodes = pq.read_table(nodes_path).to_pylist()
    edges = pq.read_table(edges_path).to_pylist()
    snaps = pq.read_table(snaps_path).to_pylist()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    road_network = RoadNetwork.from_tables(nodes, edges, snaps)

    route = road_network.route_between_places(100, 300)

    assert report["version"] == 1
    assert report["mode"] == "drive"
    assert report["inputs"]["places"] == 2
    assert report["outputs"]["nodes_file"] == str(nodes_path)
    assert report["validation"]["nodes"] == len(nodes)
    assert report["validation"]["edges"] == len(edges)
    assert report["validation"]["snaps"] == len(snaps)
    assert report["validation"]["missing_place_ids"] == []
    assert report["validation"]["road_type_counts"] == {"residential": 4, "service": 1}
    assert route is not None
    assert route.node_ids == (1, 2, 3)
    assert route.distance_m > 0
    assert route.travel_time_min >= 1


def test_mvp_source_fixture_builds_expected_routed_smoke_routes(tmp_path):
    nodes_path = tmp_path / "mvp_built_road_nodes.parquet"
    edges_path = tmp_path / "mvp_built_road_edges.parquet"
    snaps_path = tmp_path / "mvp_built_place_road_snap.parquet"
    report_path = tmp_path / "mvp_built_road_artifacts.json"

    main(
        [
            "--osm-file",
            "examples/mvp/roads.osm",
            "--places-file",
            "examples/mvp/road_builder_places.csv",
            "--nodes-out",
            str(nodes_path),
            "--edges-out",
            str(edges_path),
            "--snaps-out",
            str(snaps_path),
            "--report-out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    road_network = RoadNetwork.from_tables(
        pq.read_table(nodes_path).to_pylist(),
        pq.read_table(edges_path).to_pylist(),
        pq.read_table(snaps_path).to_pylist(),
    )

    home_a_to_work = road_network.route_between_places(100, 300)
    home_b_to_work = road_network.route_between_places(200, 300)

    assert home_a_to_work is not None
    assert report["validation"]["nodes"] == 3
    assert report["validation"]["edges"] == 4
    assert report["validation"]["snaps"] == 3
    assert report["validation"]["road_type_counts"] == {"residential": 4}
    assert home_a_to_work.node_ids == (1, 3)
    assert home_a_to_work.travel_time_min == 12
    assert home_b_to_work is not None
    assert home_b_to_work.node_ids == (2, 3)
    assert home_b_to_work.travel_time_min == 10


def test_build_graph_rejects_unsupported_mode(tmp_path):
    osm_path = tmp_path / "roads.osm"
    _write_osm_extract(osm_path)

    with pytest.raises(ValueError, match="Unsupported road-network mode"):
        build_graph(load_osm_extract(osm_path), "walk")
