"""
Partition activity location network using METIS via pymetis.

Alternative implementation using the pymetis wrapper (pure Python METIS wrapper).
This provides a simpler interface compared to metispy, with no environment variable setup needed.

It automates the METIS workflow:
    1. graph_from_data.py → Create NetworkX graph
    2. pymetis.part_graph() → Run METIS partitioning
    3. partition_data.py → Create Hive-partitioned data files

Usage:
    python -m casmsocial.network_partitioner_pymetis \\
        --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
        --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \\
        --imputation 1 \\
        --output-dir data/partitioned_8proc \\
        --n-ranks 8

Features:
    - Pure Python METIS wrapper (via pymetis)
    - No environment variables needed (METIS library bundled with pymetis)
    - Creates Hive-partitioned data files for MPI
    - Support for CSR (Compressed Sparse Row) format for large graphs
    - Supports edge weights and node weights
"""

from pathlib import Path

import typer
from loguru import logger

from casmsocial.graph_from_data import full_map_of_adjacent_places
from casmsocial.partition_data import partition_persons, partition_places


class NetworkPartitionerError(Exception):
    """Error during network partitioning."""

    pass


def partition_with_pymetis(
    persons_file: Path,
    places_file: Path,
    imputation: int,
    output_dir: Path,
    n_ranks: int = 8,
) -> None:
    """
    Partition activity location network using pymetis (pure Python METIS wrapper).

    This orchestrates the three-step METIS workflow:
    1. Build NetworkX graph from person activities
    2. Use pymetis.part_graph() to partition the graph (using CSR format)
    3. Create Hive-partitioned data files for MPI

    Args:
        persons_file: Path to persons parquet directory (Hive-partitioned)
        places_file: Path to places parquet file
        imputation: Imputation number to use
        output_dir: Output directory for partitioned data
        n_ranks: Number of MPI processes to partition for

    Requires:
        - pymetis: Pure Python METIS wrapper
          (pip install pymetis or uv add pymetis)
    """
    logger.info(f"Building activity location network for imputation {imputation}...")

    # Step 1: Build NetworkX graph from person activities
    graph = full_map_of_adjacent_places(persons_file, imputation)
    logger.info(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 2: Partition with METIS using pymetis
    logger.info(f"Partitioning graph for {n_ranks} MPI processes using pymetis...")

    try:
        import pymetis
    except ImportError as e:
        error_msg = "pymetis not installed. Install with:\n  pip install pymetis\n  or\n  uv add pymetis\n"
        logger.error(error_msg)
        raise NetworkPartitionerError(error_msg) from e

    try:
        # Create node mapping (spatial ID to 1-indexed graph ID)
        node_list = sorted(graph.nodes())
        node_mapping = {node: i + 1 for i, node in enumerate(node_list)}

        # Create METIS options with ufactor (load imbalance tolerance)
        # ufactor=30 means (1+30)/1000 = 1.03 (3% load imbalance tolerance)
        options = pymetis.Options(ufactor=30)

        # Build CSR (Compressed Sparse Row) format
        # pymetis requires 0-indexed node IDs (0 to n-1)
        logger.debug("Building CSR format for METIS partitioning")

        xadj = [0]
        adjncy = []

        for node in node_list:
            # Get neighbors and convert to 0-indexed positions
            # node_mapping[n] gives 1-indexed position, subtract 1 for 0-indexed
            neighbors = sorted([node_mapping[n] - 1 for n in graph.neighbors(node)])
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))

        # Run METIS partitioning using CSR format
        # pymetis.part_graph returns a GraphPartition with fields:
        # - edge_cuts (number of edges cut)
        # - vertex_part (partition assignment for each vertex as array)
        partition_result = pymetis.part_graph(n_ranks, xadj=xadj, adjncy=adjncy, options=options)
        edge_cut = partition_result.edge_cuts
        partition = partition_result.vertex_part

        logger.info(f"METIS partitioning complete. Edge cut: {edge_cut}")

        # Convert partition array to partition_map
        # partition[i] is the partition for the i-th node (0-indexed)
        partition_map = {}
        for i, node in enumerate(node_list):
            graph_id = node_mapping[node]
            partition_map[graph_id] = partition[i]

        # Log partition distribution
        logger.info("Partition distribution:")
        for pid in range(n_ranks):
            count = sum(1 for p in partition_map.values() if p == pid)
            if count > 0:
                logger.info(f"  Partition {pid}: {count} places")

    except Exception as e:
        error_msg = f"METIS partitioning failed: {e}"
        logger.error(error_msg)
        raise NetworkPartitionerError(error_msg) from e

    # Step 3: Create partitioned data files
    logger.info("Creating partitioned data files...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create spid_to_graphid mapping (spatial ID to 1-indexed graph ID)
    spid_to_graphid = dict(node_mapping)

    # Partition places
    partition_places(places_file, partition_map, spid_to_graphid, output_dir)

    # Partition persons
    partition_persons(persons_file, partition_map, spid_to_graphid, output_dir, imputation, n_ranks)

    logger.info(f"Network partitioned and saved to {output_dir}")
    logger.info(f"Ready for MPI execution with {n_ranks} processes")
    logger.info(
        f"Use: mpirun -n {n_ranks} python -m wake_county_heat_risk config/enhanced_heat_risk_mpi.yaml  # see wake-county-heat-risk repo"
    )


def main(
    persons_file: str = typer.Option(
        "data/processed/wake_county_30_v2/abm_inputs/persons.parquet",
        "--persons-file",
        help="Path to persons parquet directory",
    ),
    places_file: str = typer.Option(
        "data/processed/apache_flight/wake_county_30_v2/places.parquet",
        "--places-file",
        help="Path to places parquet file",
    ),
    imputation: int = typer.Option(1, "--imputation", help="Imputation number to partition"),
    output_dir: str = typer.Option(
        "data/partitioned_8proc", "--output-dir", help="Output directory for partitioned data"
    ),
    n_ranks: int = typer.Option(8, "--n-ranks", help="Number of MPI processes"),
) -> None:
    """
    Partition activity location network using pymetis.

    This automates the METIS workflow using the pure Python pymetis wrapper.
    No environment variables or system library configuration needed - pymetis
    comes with METIS bundled.

    Uses CSR (Compressed Sparse Row) format for efficient graph representation
    and partitioning.

    Example:
        python -m casmsocial.network_partitioner_pymetis \\
            --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
            --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \\
            --imputation 1 \\
            --output-dir data/partitioned_8proc \\
            --n-ranks 8
    """
    persons_path = Path(persons_file)
    places_path = Path(places_file)
    output_path = Path(output_dir)

    logger.info("Starting METIS partitioning with pymetis...")
    logger.info(f"Persons file: {persons_path}")
    logger.info(f"Places file: {places_path}")
    logger.info(f"Imputation: {imputation}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"MPI Processes: {n_ranks}")

    partition_with_pymetis(persons_path, places_path, imputation, output_path, n_ranks)


if __name__ == "__main__":
    typer.run(main)
