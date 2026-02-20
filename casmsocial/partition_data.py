"""
Create partitioned data files based on METIS graph partitioning.

This script takes the METIS partition assignment file and creates
partitioned dataframes for persons and places, ready for MPI execution.

Usage:
    python -m casmsocial.partition_data \
        --partition-file data/wake_county_graph.txt.part.8 \
        --map-file data/wake_county_graph_id_map.txt \
        --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \
        --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \
        --output-dir data/partitioned_8proc \
        --imputation 1 \
        --num-partitions 8
"""

import glob
from pathlib import Path

import duckdb
import pandas as pd
import typer
from loguru import logger


def load_metis_partition(partition_file: Path) -> dict[int, int]:
    """Load METIS partition file into dict mapping node_id -> partition_id.

    Args:
        partition_file: Path to METIS partition file (graph_id -> partition assignment)

    Returns:
        Dictionary mapping graph_id (1-based) to partition_id (0-based)
    """
    partition_map = {}
    with open(partition_file) as f:
        for node_id, line in enumerate(f, start=1):  # 1-based node IDs
            partition_id = int(line.strip())
            partition_map[node_id] = partition_id
    logger.info(f"Loaded {len(partition_map)} partition assignments from {partition_file}")
    return partition_map


def load_graph_id_map(map_file: Path) -> dict[int, int]:
    """Load graph ID map into dict mapping spatial_id -> graph_id.

    Args:
        map_file: Path to graph ID map file (spatial_id graph_id)

    Returns:
        Dictionary mapping spatial_id to graph_id
    """
    spid_to_graphid = {}
    with open(map_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sp_id, graph_id = map(int, parts)
                spid_to_graphid[sp_id] = graph_id
    logger.info(f"Loaded {len(spid_to_graphid)} spatial ID to graph ID mappings")
    return spid_to_graphid


def partition_places(
    places_file: Path,
    partition_map: dict[int, int],
    spid_to_graphid: dict[int, int],
    output_dir: Path,
) -> None:
    """Partition places dataframe and save to output directory.

    Args:
        places_file: Path to places parquet file
        partition_map: Mapping from graph_id to partition_id
        spid_to_graphid: Mapping from spatial_id to graph_id
        output_dir: Output directory for partitioned places
    """
    logger.info(f"Loading places from {places_file}...")
    places_df = pd.read_parquet(places_file)
    logger.info(f"Loaded {len(places_df)} places")

    # Map place sp_id to partition ID through graph_id
    places_df["process"] = places_df["sp_id"].map(lambda sp_id: partition_map.get(spid_to_graphid.get(sp_id, 0), 0))

    logger.info("Assigned processes to places:")
    for partition_id in sorted(places_df["process"].unique()):
        count = (places_df["process"] == partition_id).sum()
        logger.info(f"  Partition {partition_id}: {count} places")

    # Save partitioned places
    output_file = output_dir / "places_partition.parquet"
    places_df.to_parquet(output_file)
    logger.info(f"Saved partitioned places to {output_file}")


def partition_persons(
    persons_file: Path,
    partition_map: dict[int, int],
    spid_to_graphid: dict[int, int],
    output_dir: Path,
    imputation: int,
    num_partitions: int,
) -> None:
    """Partition persons dataframe by process and save as Hive-partitioned dataset.

    Args:
        persons_file: Path to persons parquet directory (Hive-partitioned by Imputation)
        partition_map: Mapping from graph_id to partition_id
        spid_to_graphid: Mapping from spatial_id to graph_id
        output_dir: Output directory for partitioned data
        imputation: Imputation number to load
        num_partitions: Number of partitions (processes)
    """
    logger.info(f"Loading persons from {persons_file} (Imputation={imputation})...")

    # Load Hive-partitioned persons data
    conn = duckdb.connect(database=":memory:")
    persons_path = str(persons_file / "*/*.parquet")
    persons_files = glob.glob(persons_path)
    logger.debug(f"Found persons files: {len(persons_files)}")

    persons = conn.execute(
        """
        SELECT
            CAST(Imputation AS INTEGER) AS Imputation,
            * EXCLUDE (Imputation)
        FROM read_parquet(
            ?,
            hive_partitioning = 1
        )
        WHERE CAST(Imputation AS INTEGER) = ?;
        """,
        [persons_files, imputation],
    ).df()

    logger.info(f"Loaded {len(persons)} persons")

    # Map person's home location to partition ID
    persons["process"] = persons["sp_hh_id"].map(lambda sp_id: partition_map.get(spid_to_graphid.get(sp_id, 0), 0))

    logger.info("Assigned processes to persons:")
    for partition_id in sorted(persons["process"].unique()):
        count = (persons["process"] == partition_id).sum()
        logger.info(f"  Partition {partition_id}: {count} persons")

    # Save partitioned persons as Hive-partitioned dataset
    # Root directory: persons.parquet
    # Partition columns: Imputation, process
    persons_output_dir = output_dir / "persons.parquet"
    logger.info(f"Saving {len(persons)} persons to Hive-partitioned dataset at {persons_output_dir}...")

    # Add Imputation column back for Hive partitioning
    persons["Imputation"] = imputation

    # Write as Hive-partitioned parquet dataset
    # This creates directory structure: persons.parquet/Imputation=N/process=M/part-*.parquet
    persons.to_parquet(
        path=str(persons_output_dir),
        engine="pyarrow",
        partition_cols=["Imputation", "process"],
        index=False,
    )

    logger.info(f"Saved Hive-partitioned persons dataset to {persons_output_dir}")
    logger.info(f"  Structure: {persons_output_dir}/Imputation={imputation}/process=*/part-*.parquet")
    logger.info("Persons partitioning complete!")


def main(
    partition_file: str = typer.Argument("data/wake_county_graph.txt.part.8", help="Path to METIS partition file"),
    map_file: str = typer.Argument("data/wake_county_graph_id_map.txt", help="Path to graph ID map file"),
    persons_file: str = typer.Argument(
        "data/processed/wake_county_30_v2/abm_inputs/persons.parquet", help="Path to persons parquet directory"
    ),
    places_file: str = typer.Argument(
        "data/processed/apache_flight/wake_county_30_v2/places.parquet", help="Path to places parquet file"
    ),
    output_dir: str = typer.Argument("data/partitioned_8proc", help="Output directory for partitioned data"),
    imputation: int = typer.Argument(1, help="Imputation number to partition"),
    num_partitions: int = typer.Argument(8, help="Number of partitions"),
) -> None:
    """Create partitioned data files based on METIS graph partitioning.

    This script reads METIS partition assignments and creates Hive-partitioned
    dataframes suitable for MPI execution.

    Example:
        python -m casmsocial.partition_data \\
            --partition-file data/wake_county_graph.txt.part.8 \\
            --map-file data/wake_county_graph_id_map.txt \\
            --persons-file data/processed/wake_county_30_v2/abm_inputs/persons.parquet \\
            --places-file data/processed/apache_flight/wake_county_30_v2/places.parquet \\
            --output-dir data/partitioned_8proc \\
            --imputation 1 \\
            --num-partitions 8
    """
    # Convert string arguments to Path objects
    partition_file_path = Path(partition_file)
    map_file_path = Path(map_file)
    persons_file_path = Path(persons_file)
    places_file_path = Path(places_file)
    output_dir_path = Path(output_dir)

    logger.info("Starting METIS partition data creation...")
    logger.info(f"Partition file: {partition_file_path}")
    logger.info(f"Map file: {map_file_path}")
    logger.info(f"Persons file: {persons_file_path}")
    logger.info(f"Places file: {places_file_path}")
    logger.info(f"Output directory: {output_dir_path}")
    logger.info(f"Imputation: {imputation}")
    logger.info(f"Number of partitions: {num_partitions}")

    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load mappings
    partition_map = load_metis_partition(partition_file_path)
    spid_to_graphid = load_graph_id_map(map_file_path)

    # Partition places and persons
    partition_places(places_file_path, partition_map, spid_to_graphid, output_dir_path)
    partition_persons(persons_file_path, partition_map, spid_to_graphid, output_dir_path, imputation, num_partitions)

    logger.info("Partitioning complete! Ready for MPI execution.")
    logger.info("Use config: config/enhanced_heat_risk_mpi.yaml  (see wake-county-heat-risk repo)")
    logger.info(
        f"Run with: mpirun -n {num_partitions} python -m wake_county_heat_risk config/enhanced_heat_risk_mpi.yaml"
    )


if __name__ == "__main__":
    typer.run(main)
