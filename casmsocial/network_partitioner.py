"""
Partition activity location network using repast4py.network with METIS.

This script replaces the previous three-step process (graph_from_data.py →
gpmetis → partition_data.py) with a single integrated step using repast4py's
built-in network partitioning support.

Usage:
    python -m casmsocial.network_partitioner \\
        --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
        --imputation 1 \\
        --output-file data/place_network.txt \\
        --n-ranks 8

Features:
    - Automatic METIS graph partitioning
    - Ghost node management for MPI
    - One-step process (graph building + partitioning)
    - NetworkX full compatibility
    - Edge attribute support
"""

from pathlib import Path

import typer
from loguru import logger

from casmsocial.graph_from_data import full_map_of_adjacent_places


class NetworkPartitionerError(Exception):
    """Error during network partitioning."""

    pass


def partition_with_repast4py(
    persons_file: Path,
    imputation: int,
    output_file: Path,
    n_ranks: int = 8,
) -> None:
    """
    Build and partition activity location network using repast4py.network.

    Args:
        persons_file: Path to persons parquet directory (Hive-partitioned)
        imputation: Imputation number to use
        output_file: Output file path for partitioned network
        n_ranks: Number of MPI processes to partition for

    Requires:
        - repast4py: Core framework
        - networkx-metis: Python wrapper for METIS partitioning
        - METIS: System library (brew install metis on macOS, apt-get on Linux)
    """
    # Defer import error to execution time with helpful message
    try:
        from repast4py.network import write_network
    except ImportError as e:
        error_msg = (
            "Missing required dependencies for network partitioning.\n"
            "\nRequired packages:\n"
            "  1. repast4py: pip install repast4py or uv add repast4py\n"
            "  2. networkx-metis: pip install networkx-metis or uv add networkx-metis\n"
            "\nRequired system library:\n"
            "  - METIS: brew install metis (macOS) or sudo apt-get install metis (Linux)\n"
            "\nFull installation:\n"
            "  uv add networkx-metis\n"
            "  brew install metis\n"
            "\nVerify installation:\n"
            "  python -c 'import networkx_metis; print(\"OK\")'\n"
        )
        logger.error(error_msg)
        raise NetworkPartitionerError(error_msg) from e

    logger.info(f"Building activity location network for imputation {imputation}...")

    # Step 1: Build NetworkX graph from person activities
    # (uses existing graph_from_data.py function)
    graph = full_map_of_adjacent_places(persons_file, imputation)

    logger.info(f"Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 2: Partition with METIS and serialize using repast4py.network
    logger.info(f"Partitioning graph for {n_ranks} MPI processes using METIS...")

    try:
        write_network(
            graph,
            network_name="place_network",
            fpath=str(output_file),
            n_ranks=n_ranks,
            partition_method="metis",
        )
    except Exception as e:
        logger.error(
            f"Error during METIS partitioning: {e}\n"
            "This may indicate METIS library is not installed.\n"
            "On macOS: brew install metis\n"
            "On Linux: sudo apt-get install metis libmetis-dev\n"
        )
        raise

    logger.info(f"Network partitioned and written to {output_file}")
    logger.info(f"Ready for MPI execution with {n_ranks} processes")
    logger.info(f"Use: mpirun -n {n_ranks} python -m casmsocial config/enhanced_heat_risk_mpi_repast4py.yaml")


def main(
    persons_file: str = typer.Option(
        "data/processed/wake_county_30_v2/abm_inputs/persons.parquet",
        "--persons-file",
        help="Path to persons parquet directory",
    ),
    imputation: int = typer.Option(1, "--imputation", help="Imputation number to partition"),
    output_file: str = typer.Option(
        "data/place_network.txt",
        "--output-file",
        help="Output network file path",
    ),
    n_ranks: int = typer.Option(8, "--n-ranks", help="Number of MPI processes"),
) -> None:
    """
    Partition activity location network using repast4py.network with METIS.

    This single-step process replaces the previous workflow:
    - graph_from_data.py (create graph)
    - gpmetis (external partitioning)
    - partition_data.py (serialize for MPI)

    All functionality is now integrated here.

    Example:
        python -m casmsocial.network_partitioner \\
            --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
            --imputation 1 \\
            --output-file data/place_network.txt \\
            --n-ranks 8
    """
    persons_path = Path(persons_file)
    output_path = Path(output_file)

    logger.info("Starting repast4py.network METIS partitioning...")
    logger.info(f"Input: {persons_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"MPI Processes: {n_ranks}")

    partition_with_repast4py(persons_path, imputation, output_path, n_ranks)


if __name__ == "__main__":
    typer.run(main)
