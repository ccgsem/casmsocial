import glob
import os
from pathlib import Path
from typing import Optional

import duckdb
import networkx as nx
import pandas as pd
import typer
from dotenv import load_dotenv
from loguru import logger


def isNumber(v):
    return isinstance(v, (int, float))


def map_spid_to_graph_id(places_f):
    map_df = pd.DataFrame()

    for f in places_f:
        df = pd.read_parquet(f)

        df = df["sp_id"]
        logger.debug(len(df))

        map_df = pd.concat([map_df, df], axis=0, ignore_index=True)

    map_df.columns = ["sp_id"]
    map_df["graph_id"] = range(len(map_df))
    map_df["graph_id"] = map_df["graph_id"] + 1

    ret_map = dict(zip(map_df["sp_id"], map_df["graph_id"]))
    return ret_map


def add_to_dict(key, val, d):
    if key is not None and val is not None:
        if key in d:
            d[key].append(val)
        else:
            d[key] = [val]
    return d


def doubly_add_to_dict(val1, val2, d):
    d = add_to_dict(val1, val2, d)
    d = add_to_dict(val2, val1, d)
    return d


def full_map_of_adjacent_places(persons_file: Path, imputation: int) -> nx.Graph:
    """
    Create a NetworkX graph of adjacent places from persons data.

    Returns:
        nx.Graph: NetworkX graph where nodes are place IDs and edges connect
                 places that are visited by the same person (home-school,
                 home-work, work-school connections).
    """
    # Create an undirected graph
    graph = nx.Graph()

    persons_files_path = str(persons_file / "*/*.parquet")
    persons_files = glob.glob(persons_files_path)
    logger.info(f"Using imputation {imputation} for persons file {persons_files_path}...")

    conn = duckdb.connect(database=":memory:")
    persons = conn.execute(
        """
        SELECT
            -- Cast Hive partition column from VARCHAR to INTEGER
            CAST(Imputation AS INTEGER) AS Imputation,
            * EXCLUDE (Imputation)
        FROM read_parquet(
            ?,          -- directory with Hive-style partitions
            hive_partitioning = 1       -- tell DuckDB to read dir names as columns

        )
        WHERE CAST(Imputation AS INTEGER) = ?;
        """,
        [persons_files, imputation],
    ).df()

    logger.info(f"Processing {len(persons)} persons to build network")

    # Process each person's places and add edges
    for hh, school, work in zip(persons["sp_hh_id"], persons["sp_school_id"], persons["sp_work_id"]):
        # Handle NaN/NA values from parquet files (can be NaN, None, or pd.NA)
        hh_id = None
        school_id = None
        work_id = None

        # Check for various NA representations
        if pd.notna(hh) and hh is not None:
            hh_id = int(hh)

        if pd.notna(school) and school is not None:
            school_id = int(school)

        if pd.notna(work) and work is not None:
            work_id = int(work)

        # Add nodes to graph (NetworkX handles duplicates automatically)
        valid_places = [place for place in [hh_id, school_id, work_id] if place is not None]
        graph.add_nodes_from(valid_places)

        # Add edges between all pairs of places for this person
        if hh_id is not None and school_id is not None:
            graph.add_edge(hh_id, school_id)
        if hh_id is not None and work_id is not None:
            graph.add_edge(hh_id, work_id)
        if school_id is not None and work_id is not None:
            graph.add_edge(school_id, work_id)

    logger.info(f"Created network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def get_num_v_and_e_from_graph(graph: nx.Graph) -> tuple[int, int]:
    """Get number of vertices and edges from NetworkX graph."""
    return graph.number_of_nodes(), graph.number_of_edges()


def make_spid_to_graphid_map_from_graph(graph: nx.Graph) -> dict[int, int]:
    """Create mapping from spatial IDs to sequential graph IDs."""
    spid_to_graphid = {}
    for i, node in enumerate(graph.nodes(), start=1):
        spid_to_graphid[node] = i
    return spid_to_graphid


def write_graph_file_from_networkx(output_f: Path, graph: nx.Graph, spid_to_graphid: dict[int, int]):
    """Write graph in adjacency list format from NetworkX graph."""
    num_v = graph.number_of_nodes()
    num_e = graph.number_of_edges()

    with open(output_f, "w") as f:
        f.write(f"{num_v} {num_e}\n")

        # Write adjacency list for each node (in graph ID order)
        for node in sorted(graph.nodes()):
            neighbors = list(graph.neighbors(node))
            # Convert to graph IDs and sort
            neighbor_graph_ids = sorted([spid_to_graphid[neighbor] for neighbor in neighbors])
            f.write(" ".join(map(str, neighbor_graph_ids)))
            f.write("\n")


def write_map_file(output_f: Path, spid_to_graphid: dict[int, int]):
    """Write spatial ID to graph ID mapping file."""
    with open(output_f, "w") as f:
        for spatial_id, graph_id in spid_to_graphid.items():
            f.write(f"{spatial_id} {graph_id}\n")


def save_networkx_formats(graph: nx.Graph, base_path: Path):
    """Save graph in various NetworkX formats for analysis."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GraphML (XML format, preserves attributes)
    nx.write_graphml(graph, f"{base_path.stem}.graphml")
    logger.info(f"Saved GraphML format: {base_path.stem}.graphml")

    # Save as GML (simple text format)
    nx.write_gml(graph, f"{base_path.stem}.gml")
    logger.info(f"Saved GML format: {base_path.stem}.gml")

    # Save basic network statistics
    stats_file = f"{base_path.stem}_stats.txt"
    with open(stats_file, "w") as f:
        f.write("Network Statistics\n")
        f.write("==================\n")
        f.write(f"Nodes: {graph.number_of_nodes()}\n")
        f.write(f"Edges: {graph.number_of_edges()}\n")
        f.write(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}\n")
        f.write(f"Density: {nx.density(graph):.6f}\n")

        if nx.is_connected(graph):
            f.write("Connected: Yes\n")
            f.write(f"Average path length: {nx.average_shortest_path_length(graph):.2f}\n")
        else:
            f.write("Connected: No\n")
            f.write(f"Connected components: {nx.number_connected_components(graph)}\n")

    logger.info(f"Saved network statistics: {stats_file}")


def main(
    persons_file: Optional[Path] = None,
    imputation: Optional[int] = None,
    output_file: Optional[Path] = None,
    map_file: Optional[Path] = None,
    save_networkx: bool = typer.Option(False, "--save-networkx", help="Save graph in NetworkX formats (GraphML, GML)"),
):
    """
    Generate graph data from agent/person parquet files using NetworkX.

    This script reads agent data from a parquet file and generates a graph
    representation along with ID mappings. Uses NetworkX for robust graph
    handling and optional analysis formats.
    """
    # Set defaults if not provided
    if persons_file is None:
        persons_file = Path("data/persons.parquet")
    if imputation is None:
        imputation = 1
    if output_file is None:
        output_file = Path("data/graph.txt")
    if map_file is None:
        map_file = Path("data/graph_id_map.txt")

    logger.info(f"Processing agent file: {persons_file}")
    logger.info(f"Output graph file: {output_file}")
    logger.info(f"Output map file: {map_file}")

    # Load environment variables
    load_dotenv()
    data_path = os.environ.get("CASMSOCIAL_DATA_PATH")
    if data_path:
        logger.info(f"Using CASMSOCIAL_DATA_PATH: {data_path}")
        persons_file = Path(data_path) / persons_file

    # Build NetworkX graph
    graph = full_map_of_adjacent_places(persons_file, imputation)

    # Generate mappings and write output files
    num_v, num_e = get_num_v_and_e_from_graph(graph)
    spid_to_graphid_map = make_spid_to_graphid_map_from_graph(graph)

    write_graph_file_from_networkx(output_file, graph, spid_to_graphid_map)
    write_map_file(map_file, spid_to_graphid_map)

    # Optionally save NetworkX formats
    if save_networkx:
        save_networkx_formats(graph, output_file)

    logger.info(f"Graph generation complete: NumV: {num_v}, NumE: {num_e}")


if __name__ == "__main__":
    typer.run(main)
