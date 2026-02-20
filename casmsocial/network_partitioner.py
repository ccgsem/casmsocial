"""
Partition activity location network using METIS.

This script creates METIS-partitioned data ready for MPI execution.
It automates the manual workflow:
    1. graph_from_data.py → Create NetworkX graph
    2. metispy.part_graph() → Run METIS partitioning
    3. partition_data.py → Create Hive-partitioned data files

Usage:
    python -m casmsocial.network_partitioner \\
        --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
        --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \\
        --imputation 1 \\
        --output-dir data/partitioned_8proc \\
        --n-ranks 8

Features:
    - Pure Python METIS partitioning (via metispy)
    - Creates Hive-partitioned data files for MPI
    - No external system commands needed (after METIS library installed)
"""

import os
from pathlib import Path

import typer
from loguru import logger

from casmsocial.graph_from_data import full_map_of_adjacent_places
from casmsocial.partition_data import partition_persons, partition_places


class NetworkPartitionerError(Exception):
    """Error during network partitioning."""

    pass


def partition_with_metispy(
    persons_file: Path,
    places_file: Path,
    imputation: int,
    output_dir: Path,
    n_ranks: int = 8,
) -> None:
    """
    Partition activity location network using metispy (metis-python).

    This orchestrates the three-step METIS workflow:
    1. Build NetworkX graph from person activities
    2. Use metispy.part_graph() to partition the NetworkX graph
    3. Create Hive-partitioned data files for MPI

    Args:
        persons_file: Path to persons parquet directory (Hive-partitioned)
        places_file: Path to places parquet file
        imputation: Imputation number to use
        output_dir: Output directory for partitioned data
        n_ranks: Number of MPI processes to partition for

    Requires:
        - metis-python: Python METIS wrapper
          (pip install metis-python or uv add metis-python)
        - METIS system library with METIS_DLL environment variable set
          (brew install metis on macOS, apt-get install libmetis-dev on Linux)
    """
    # Check METIS_DLL is set
    metis_dll = os.environ.get("METIS_DLL")
    if not metis_dll:
        error_msg = (
            "METIS_DLL environment variable not set.\n"
            "\nSet it with:\n"
            '  export METIS_DLL="$(brew --prefix metis)/lib/libmetis.dylib"  # macOS\n'
            "  export METIS_DLL=/usr/lib/x86_64-linux-gnu/libmetis.so  # Linux\n"
            "\nThen run network_partitioner again."
        )
        logger.error(error_msg)
        raise NetworkPartitionerError(error_msg)

    logger.info(f"Building activity location network for imputation {imputation}...")

    # Step 1: Build NetworkX graph from person activities
    graph = full_map_of_adjacent_places(persons_file, imputation)
    logger.info(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 2: Partition with METIS using metispy
    logger.info(f"Partitioning graph for {n_ranks} MPI processes using METIS...")

    try:
        import metis
    except ImportError as e:
        error_msg = (
            "metis-python not installed. Install with:\n  pip install metis-python\n  or\n  uv add metis-python\n"
        )
        logger.error(error_msg)
        raise NetworkPartitionerError(error_msg) from e

    try:
        # Create node mapping (spatial ID to 1-indexed graph ID)
        node_list = sorted(graph.nodes())
        node_mapping = {node: i + 1 for i, node in enumerate(node_list)}

        # Run METIS partitioning directly on NetworkX graph
        # metispy.part_graph() accepts NetworkX graphs and returns (objval, partition)
        # where partition is a list indexed by node order (0-indexed)
        objval, partition = metis.part_graph(graph, nparts=n_ranks, ufactor=30)
        logger.info(f"METIS partitioning complete. Edge cut: {objval}")

        # Convert partition array to partition_map
        # partition[i] is the partition for the i-th node (0-indexed)
        # We need to map from 1-indexed graph IDs to partition assignments
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
    Partition activity location network using METIS.

    This automates the manual METIS workflow:
    1. Build NetworkX graph from person activities
    2. Run metispy.part_graph() to partition the graph
    3. Create Hive-partitioned data files for MPI

    Example:
        export METIS_DLL="$(brew --prefix metis)/lib/libmetis.dylib"
        python -m casmsocial.network_partitioner \\
            --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
            --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \\
            --imputation 1 \\
            --output-dir data/partitioned_8proc \\
            --n-ranks 8
    """
    persons_path = Path(persons_file)
    places_path = Path(places_file)
    output_path = Path(output_dir)

    logger.info("Starting METIS partitioning with metispy...")
    logger.info(f"Persons file: {persons_path}")
    logger.info(f"Places file: {places_path}")
    logger.info(f"Imputation: {imputation}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"MPI Processes: {n_ranks}")

    partition_with_metispy(persons_path, places_path, imputation, output_path, n_ranks)


if __name__ == "__main__":
    typer.run(main)
