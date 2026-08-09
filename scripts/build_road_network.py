from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EARTH_RADIUS_M = 6_371_000.0
DRIVE_SPEED_KPH_BY_HIGHWAY = {
    "motorway": 105.0,
    "motorway_link": 65.0,
    "trunk": 90.0,
    "trunk_link": 55.0,
    "primary": 70.0,
    "primary_link": 45.0,
    "secondary": 60.0,
    "secondary_link": 40.0,
    "tertiary": 50.0,
    "tertiary_link": 35.0,
    "unclassified": 40.0,
    "residential": 35.0,
    "living_street": 10.0,
    "service": 25.0,
}
NON_DRIVE_HIGHWAYS = {
    "bridleway",
    "corridor",
    "cycleway",
    "elevator",
    "footway",
    "path",
    "pedestrian",
    "platform",
    "steps",
    "track",
}


@dataclass(frozen=True)
class OsmNode:
    node_id: int
    x: float
    y: float


@dataclass(frozen=True)
class OsmWay:
    way_id: int
    node_ids: tuple[int, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class OsmExtract:
    nodes: dict[int, OsmNode]
    ways: tuple[OsmWay, ...]


@dataclass(frozen=True)
class PlaceRecord:
    place_id: int
    x: float
    y: float


@dataclass
class RoadEdgeRecord:
    edge_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    travel_time_min: float
    mode: str
    road_type: str


@dataclass
class RoadGraph:
    nodes: dict[int, OsmNode]
    edges: list[RoadEdgeRecord] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build road-network artifacts for casmsocial from an OSM extract.")
    parser.add_argument("--osm-file", required=True, help="Path to the input OpenStreetMap extract.")
    parser.add_argument("--places-file", required=True, help="Path to the place records used for snapping.")
    parser.add_argument("--nodes-out", required=True, help="Output path for road node records.")
    parser.add_argument("--edges-out", required=True, help="Output path for road edge records.")
    parser.add_argument("--snaps-out", required=True, help="Output path for place-to-road-node snap records.")
    parser.add_argument("--report-out", help="Optional path for a JSON road-artifact build report.")
    parser.add_argument(
        "--mode",
        default="drive",
        help="Travel mode to extract from the road graph. Defaults to 'drive'.",
    )
    return parser.parse_args(argv)


def load_osm_extract(osm_file: Path) -> OsmExtract:
    """Load the source OSM extract.

    The MVP builder supports OSM XML extracts. It intentionally avoids a heavy
    GIS dependency while the production preprocessing stack is still settling.
    """
    if not osm_file.exists():
        raise FileNotFoundError(osm_file)
    if osm_file.suffix.lower() not in {".osm", ".xml"}:
        raise ValueError(f"Unsupported OSM extract format: {osm_file}")

    root = ET.parse(osm_file).getroot()
    nodes: dict[int, OsmNode] = {}
    for node in root.findall("node"):
        node_id = int(_required_attr(node, "id"))
        nodes[node_id] = OsmNode(
            node_id=node_id,
            x=float(_required_attr(node, "lon")),
            y=float(_required_attr(node, "lat")),
        )

    ways = []
    for way in root.findall("way"):
        way_id = int(_required_attr(way, "id"))
        node_ids = tuple(int(nd.attrib["ref"]) for nd in way.findall("nd") if "ref" in nd.attrib)
        tags = {
            tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag") if "k" in tag.attrib and "v" in tag.attrib
        }
        if len(node_ids) >= 2:
            ways.append(OsmWay(way_id=way_id, node_ids=node_ids, tags=tags))

    return OsmExtract(nodes=nodes, ways=tuple(ways))


def load_places(places_file: Path) -> list[PlaceRecord]:
    """Load the place records that will be snapped to the road network."""
    if not places_file.exists():
        raise FileNotFoundError(places_file)

    suffix = places_file.suffix.lower()
    if suffix == ".parquet":
        records = pq.read_table(places_file).to_pylist()
    elif suffix == ".csv":
        with places_file.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    elif suffix == ".json":
        payload = json.loads(places_file.read_text(encoding="utf-8"))
        records = payload["places"] if isinstance(payload, dict) and "places" in payload else payload
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list of places in {places_file}")
    else:
        raise ValueError(f"Unsupported place record format: {places_file}")

    return [_place_record_from_mapping(record, places_file) for record in records]


def build_graph(osm_data: OsmExtract, mode: str) -> RoadGraph:
    """Build a directed road graph filtered for the requested travel mode."""
    if mode != "drive":
        raise ValueError(f"Unsupported road-network mode: {mode}")

    graph = RoadGraph(nodes={})
    edge_id = 1
    for way in osm_data.ways:
        highway = way.tags.get("highway")
        if highway is None or highway in NON_DRIVE_HIGHWAYS:
            continue

        speed_kph = _speed_kph_for_way(way)
        direction = _oneway_direction(way)
        for from_node_id, to_node_id in zip(way.node_ids, way.node_ids[1:]):
            from_node = osm_data.nodes.get(from_node_id)
            to_node = osm_data.nodes.get(to_node_id)
            if from_node is None or to_node is None:
                continue

            graph.nodes[from_node.node_id] = from_node
            graph.nodes[to_node.node_id] = to_node
            if direction in {"forward", "both"}:
                graph.edges.append(_edge_record(edge_id, from_node, to_node, speed_kph, mode, highway))
                edge_id += 1
            if direction in {"reverse", "both"}:
                graph.edges.append(_edge_record(edge_id, to_node, from_node, speed_kph, mode, highway))
                edge_id += 1

    if not graph.edges:
        raise ValueError("No routable road edges were found in the OSM extract")
    return graph


def compute_edge_travel_times(graph: RoadGraph) -> None:
    """Populate edge travel-time attributes on the graph in minutes."""
    for edge in graph.edges:
        if edge.length_m <= 0 or edge.travel_time_min <= 0:
            raise ValueError(f"Invalid road edge travel time for edge {edge.edge_id}")


def snap_places_to_graph(places: list[PlaceRecord], graph: RoadGraph) -> list[dict[str, int]]:
    """Snap places to the nearest road nodes and return snap records."""
    if not graph.nodes:
        raise ValueError("Cannot snap places to an empty road graph")

    snaps = []
    for place in places:
        nearest_node = min(
            graph.nodes.values(),
            key=lambda node: (_haversine_m(place.x, place.y, node.x, node.y), node.node_id),
        )
        snaps.append({"place_id": place.place_id, "road_node_id": nearest_node.node_id})
    return snaps


def write_nodes(graph: RoadGraph, output_path: Path) -> None:
    """Write road-node records to the configured output."""
    rows = [
        {"node_id": node.node_id, "x": node.x, "y": node.y}
        for node in sorted(graph.nodes.values(), key=lambda node: node.node_id)
    ]
    _write_parquet_records(rows, output_path)


def write_edges(graph: RoadGraph, output_path: Path) -> None:
    """Write road-edge records to the configured output."""
    rows = [
        {
            "edge_id": edge.edge_id,
            "from_node_id": edge.from_node_id,
            "to_node_id": edge.to_node_id,
            "length_m": edge.length_m,
            "travel_time_min": edge.travel_time_min,
            "mode": edge.mode,
            "road_type": edge.road_type,
        }
        for edge in graph.edges
    ]
    _write_parquet_records(rows, output_path)


def write_snaps(snaps: list[dict[str, int]], output_path: Path) -> None:
    """Write place-to-road-node snap records to the configured output."""
    _write_parquet_records(snaps, output_path)


def build_road_artifact_report(
    *,
    osm_file: Path,
    places_file: Path,
    nodes_out: Path,
    edges_out: Path,
    snaps_out: Path,
    osm_data: OsmExtract,
    places: list[PlaceRecord],
    graph: RoadGraph,
    snaps: list[dict[str, int]],
    mode: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable summary for generated road artifacts."""
    road_type_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    total_length_m = 0.0
    total_travel_time_min = 0.0
    for edge in graph.edges:
        road_type_counts[edge.road_type] = road_type_counts.get(edge.road_type, 0) + 1
        mode_counts[edge.mode] = mode_counts.get(edge.mode, 0) + 1
        total_length_m += edge.length_m
        total_travel_time_min += edge.travel_time_min

    place_ids = {place.place_id for place in places}
    snapped_place_ids = {int(snap["place_id"]) for snap in snaps}
    missing_place_ids = sorted(place_ids - snapped_place_ids)

    return {
        "version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "inputs": {
            "osm_file": str(osm_file),
            "places_file": str(places_file),
            "osm_nodes": len(osm_data.nodes),
            "osm_ways": len(osm_data.ways),
            "places": len(places),
        },
        "outputs": {
            "nodes_file": str(nodes_out),
            "edges_file": str(edges_out),
            "snaps_file": str(snaps_out),
        },
        "validation": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "snaps": len(snaps),
            "missing_place_ids": missing_place_ids,
            "mode_counts": dict(sorted(mode_counts.items())),
            "road_type_counts": dict(sorted(road_type_counts.items())),
            "total_length_m": total_length_m,
            "total_travel_time_min": total_travel_time_min,
        },
    }


def write_road_artifact_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a road-artifact build report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_attr(element: ET.Element, name: str) -> str:
    try:
        return element.attrib[name]
    except KeyError as exc:
        raise ValueError(f"OSM element is missing required attribute {name!r}") from exc


def _place_record_from_mapping(record: dict[str, Any], source_path: Path) -> PlaceRecord:
    place_id = _first_present(record, ("place_id", "sp_id", "id"), source_path)
    x = _first_present(record, ("x", "longitude", "lon"), source_path)
    y = _first_present(record, ("y", "latitude", "lat"), source_path)
    return PlaceRecord(place_id=int(place_id), x=float(x), y=float(y))


def _first_present(record: dict[str, Any], names: tuple[str, ...], source_path: Path) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    raise ValueError(f"Place record in {source_path} is missing one of: {', '.join(names)}")


def _speed_kph_for_way(way: OsmWay) -> float:
    maxspeed = way.tags.get("maxspeed")
    parsed_maxspeed = _parse_maxspeed_kph(maxspeed) if maxspeed else None
    if parsed_maxspeed is not None:
        return parsed_maxspeed
    return DRIVE_SPEED_KPH_BY_HIGHWAY.get(way.tags.get("highway", ""), 35.0)


def _parse_maxspeed_kph(maxspeed: str) -> float | None:
    token = maxspeed.split(";")[0].strip().lower()
    if token in {"none", "signals", "variable", "walk"}:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", token)
    if match is None:
        return None

    speed = float(match.group(1))
    if "mph" in token:
        return speed * 1.609344
    return speed


def _oneway_direction(way: OsmWay) -> str:
    oneway = way.tags.get("oneway", "").strip().lower()
    if oneway in {"yes", "true", "1"} or way.tags.get("junction") == "roundabout":
        return "forward"
    if oneway in {"-1", "reverse"}:
        return "reverse"
    return "both"


def _edge_record(
    edge_id: int,
    from_node: OsmNode,
    to_node: OsmNode,
    speed_kph: float,
    mode: str,
    highway: str,
) -> RoadEdgeRecord:
    length_m = _haversine_m(from_node.x, from_node.y, to_node.x, to_node.y)
    speed_m_per_min = speed_kph * 1000.0 / 60.0
    travel_time_min = length_m / speed_m_per_min
    return RoadEdgeRecord(
        edge_id=edge_id,
        from_node_id=from_node.node_id,
        to_node_id=to_node.node_id,
        length_m=length_m,
        travel_time_min=travel_time_min,
        mode=mode,
        road_type=highway,
    )


def _haversine_m(origin_x: float, origin_y: float, destination_x: float, destination_y: float) -> float:
    origin_lat = math.radians(origin_y)
    destination_lat = math.radians(destination_y)
    delta_lat = math.radians(destination_y - origin_y)
    delta_lon = math.radians(destination_x - origin_x)
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(origin_lat) * math.cos(destination_lat) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _write_parquet_records(records: list[dict[str, Any]], output_path: Path) -> None:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Road artifact outputs must be parquet files: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), output_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    osm_file = Path(args.osm_file)
    places_file = Path(args.places_file)
    nodes_out = Path(args.nodes_out)
    edges_out = Path(args.edges_out)
    snaps_out = Path(args.snaps_out)
    report_out = Path(args.report_out) if args.report_out else None

    osm_data = load_osm_extract(osm_file)
    places = load_places(places_file)
    graph = build_graph(osm_data, args.mode)
    compute_edge_travel_times(graph)
    snaps = snap_places_to_graph(places, graph)

    write_nodes(graph, nodes_out)
    write_edges(graph, edges_out)
    write_snaps(snaps, snaps_out)
    if report_out is not None:
        report = build_road_artifact_report(
            osm_file=osm_file,
            places_file=places_file,
            nodes_out=nodes_out,
            edges_out=edges_out,
            snaps_out=snaps_out,
            osm_data=osm_data,
            places=places,
            graph=graph,
            snaps=snaps,
            mode=args.mode,
        )
        write_road_artifact_report(report, report_out)
    report_text = f", report={report_out}" if report_out is not None else ""
    print(
        "Road network artifacts written: "
        f"nodes={len(graph.nodes)}, edges={len(graph.edges)}, snaps={len(snaps)}"
        f"{report_text}"
    )


if __name__ == "__main__":
    main()
