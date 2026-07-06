"""
Partition activity location network using METIS, sourcing input data from a
DuckLake catalog.

Inputs (``persons``, ``places``) live as catalog tables under the convention
already used by the rest of casmsocial (see ``ducklake_utils.py`` and the YAML
keys ``persons.table`` / ``places.table``). The partition assignment is written
back to the same DuckLake as a table with one row per place, suitable for
joining against the places table at run time and for keeping a history of
partitions across experiments.

Workflow:
    1. Attach the DuckLake catalog (``casmsocial.ducklake_utils.get_ducklake_connection``).
    2. Pull persons rows for each requested imputation and build a NetworkX graph
       whose nodes are place IDs and whose edges connect places visited by the
       same person (home, work, school). Home-person counts are retained as
       optional vertex weights for workload-balanced partitioning.
    3. Partition each graph with ``pymetis.part_graph`` using the CSR format for
       each requested MPI rank count.
    4. Append rows ``(imputation, n_ranks, rank, place_id)`` to the output
       table, replacing any pre-existing rows for the same
       ``(imputation, n_ranks)`` so re-running is idempotent.

Usage:
    python -m casmsocial.network_partitioner_ducklake \\
        --ducklake-path data/datalakehouse \\
        --schema rti_synth_pop_v2_dmv_100 \\
        --imputations all \\
        --n-ranks 2,4,8 \\
        --output-table partitions.metis_place_partitions
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import typer
from dotenv import load_dotenv
from loguru import logger

from casmsocial.data_utilities import check_if_table_exists, quote_table_identifier
from casmsocial.ducklake_utils import get_ducklake_connection


class NetworkPartitionerError(Exception):
    """Error during DuckLake-driven network partitioning."""


def _no_persons_message(persons_table: str, imputation: int) -> str:
    return f"No persons returned from {persons_table} for imputation={imputation}"


def _invalid_rank_count_message(n_ranks: int) -> str:
    return f"n_ranks must be >= 1, got {n_ranks}"


def _missing_pymetis_message() -> str:
    return "pymetis is not installed. Install with `uv add pymetis` or `pip install pymetis`."


def _missing_ducklake_message(ducklake_path: Path) -> str:
    return f"DuckLake path does not exist: {ducklake_path}"


def _invalid_output_table_message(qualified: str) -> str:
    return f"Output table name must be 'table' or 'schema.table', got '{qualified}'"


PERSON_WEIGHT_ATTR = "person_weight"
_WEIGHT_BY_OPTIONS = {
    "none": None,
    "places": None,
    "place": None,
    "persons": PERSON_WEIGHT_ATTR,
    "person": PERSON_WEIGHT_ATTR,
    "home_persons": PERSON_WEIGHT_ATTR,
}


def _parse_positive_int_list(value: str, *, name: str) -> list[int]:
    """Parse a comma-separated positive integer list, preserving first occurrence order."""
    parsed_values: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise NetworkPartitionerError(f"{name} must be comma-separated integers, got '{value}'") from exc
        if parsed < 1:
            raise NetworkPartitionerError(f"{name} values must be >= 1, got {parsed}")
        if parsed not in parsed_values:
            parsed_values.append(parsed)

    if not parsed_values:
        raise NetworkPartitionerError(f"{name} must include at least one value")
    return parsed_values


def _weight_attribute_for(weight_by: str | None) -> str | None:
    normalized = (weight_by or "none").strip().lower()
    if normalized in _WEIGHT_BY_OPTIONS:
        return _WEIGHT_BY_OPTIONS[normalized]
    valid_options = ", ".join(sorted(_WEIGHT_BY_OPTIONS))
    raise NetworkPartitionerError(f"Invalid weight_by '{weight_by}'. Valid options: {valid_options}")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    table_identifier = quote_table_identifier(table_name)
    return {column[0].lower() for column in conn.execute(f"SELECT * FROM {table_identifier} LIMIT 0").description}


def _load_person_place_columns(
    conn: duckdb.DuckDBPyConnection,
    persons_table: str,
    imputation: int,
) -> pd.DataFrame:
    persons_identifier = quote_table_identifier(persons_table)
    columns = _table_columns(conn, persons_table)
    if "imputation" not in columns:
        logger.info(f"Table {persons_table} has no Imputation column; loading all persons rows.")
        query = f"""
            SELECT sp_hh_id, sp_school_id, sp_work_id
            FROM {persons_identifier}
            """  # noqa: S608 - table identifier is validated by quote_table_identifier.
        return conn.execute(query).df()

    query = f"""
        SELECT sp_hh_id, sp_school_id, sp_work_id
        FROM {persons_identifier}
        WHERE CAST(Imputation AS INTEGER) = ?
        """  # noqa: S608 - table identifiers are validated by quote_table_identifier.
    return conn.execute(query, [imputation]).df()


def _load_place_ids(conn: duckdb.DuckDBPyConnection, places_table: str) -> set[int]:
    places_identifier = quote_table_identifier(places_table)
    query = f"SELECT sp_id FROM {places_identifier}"  # noqa: S608 - validated identifier.
    place_ids_df = conn.execute(query).df()
    return {int(place_id) for place_id in place_ids_df["sp_id"].tolist()}


def _coerce_place_id(value: object, valid_place_ids: set[int] | None) -> int | None:
    """Convert a parquet/duckdb value to int, treating NaN/NA as None."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    coerced = int(value)
    if valid_place_ids is not None and coerced not in valid_place_ids:
        return None
    return coerced


def _add_person_place_edges(
    graph: nx.Graph,
    persons: pd.DataFrame,
    valid_place_ids: set[int] | None,
) -> None:
    for hh, school, work in zip(persons["sp_hh_id"], persons["sp_school_id"], persons["sp_work_id"]):
        home_id = _coerce_place_id(hh, valid_place_ids)
        place_ids = [home_id, _coerce_place_id(school, valid_place_ids), _coerce_place_id(work, valid_place_ids)]
        valid_places = [place_id for place_id in place_ids if place_id is not None]
        graph.add_nodes_from(valid_places)
        if home_id is not None:
            graph.nodes[home_id][PERSON_WEIGHT_ATTR] = int(graph.nodes[home_id].get(PERSON_WEIGHT_ATTR, 0)) + 1
        graph.add_edges_from(combinations(valid_places, 2))


def build_graph_from_ducklake(
    conn: duckdb.DuckDBPyConnection,
    persons_table: str,
    imputation: int,
    *,
    places_table: str | None = None,
    restrict_to_places: bool = False,
) -> nx.Graph:
    """Build a NetworkX graph of co-visited places from a DuckLake catalog.

    Args:
        conn: An attached DuckLake connection (see
            ``casmsocial.ducklake_utils.get_ducklake_connection``).
        persons_table: Catalog-qualified persons table (e.g.
            ``rti_synth_pop_v2_dmv_100.persons``).
        imputation: Imputation column value to filter on.
        places_table: Optional places table. When supplied, every ``sp_id`` in
            the table is seeded as a graph node, including isolated places not
            referenced by persons.
        restrict_to_places: When True, person-referenced IDs outside
            ``places_table`` are dropped. When False, those references are
            still admitted for permissive compatibility with earlier partitioning behavior.

    Returns:
        A NetworkX undirected graph with one node per ``sp_id`` and edges
        between every pair of places that share at least one person.
    """
    logger.info(
        f"Building activity location network from DuckLake " f"(table={persons_table}, imputation={imputation})"
    )

    persons = _load_person_place_columns(conn, persons_table, imputation)

    logger.info(f"Loaded {len(persons):,} person rows for imputation {imputation}")
    if persons.empty:
        raise NetworkPartitionerError(_no_persons_message(persons_table, imputation))

    graph: nx.Graph = nx.Graph()
    valid_place_ids: set[int] | None = None
    if places_table is not None:
        place_ids = _load_place_ids(conn, places_table)
        graph.add_nodes_from(place_ids)
        if restrict_to_places:
            valid_place_ids = place_ids
        logger.info(f"Seeded graph with {len(place_ids):,} known places from {places_table}")

    _add_person_place_edges(graph, persons, valid_place_ids)

    logger.info(f"Graph built: {graph.number_of_nodes():,} nodes, " f"{graph.number_of_edges():,} edges")
    return graph


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def _vertex_weight(graph: nx.Graph, node: int, weight_attribute: str | None) -> int:
    if weight_attribute is None:
        return 1
    return max(1, int(graph.nodes[node].get(weight_attribute, 0) or 0))


def partition_graph_with_pymetis(
    graph: nx.Graph,
    n_ranks: int,
    ufactor: int = 30,
    *,
    weight_attribute: str | None = None,
) -> dict[int, int]:
    """Partition a NetworkX graph into ``n_ranks`` parts via pymetis.

    Returns a mapping ``place_id -> rank`` (0-indexed rank assignments).
    """
    if n_ranks < 1:
        raise NetworkPartitionerError(_invalid_rank_count_message(n_ranks))

    try:
        import pymetis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NetworkPartitionerError(_missing_pymetis_message()) from exc

    if graph.number_of_nodes() == 0:
        raise NetworkPartitionerError("Cannot partition an empty graph")  # noqa: TRY003

    if n_ranks == 1:
        # Trivial case: pymetis fails on n_parts=1; everyone goes to rank 0.
        return {int(node): 0 for node in graph.nodes()}

    node_list = sorted(graph.nodes())
    node_to_csr_idx = {node: i for i, node in enumerate(node_list)}

    xadj: list[int] = [0]
    adjncy: list[int] = []
    for node in node_list:
        neighbors = sorted(node_to_csr_idx[n] for n in graph.neighbors(node))
        adjncy.extend(neighbors)
        xadj.append(len(adjncy))

    vweights = None
    if weight_attribute is not None:
        vweights = [_vertex_weight(graph, node, weight_attribute) for node in node_list]

    options = pymetis.Options(ufactor=ufactor)

    logger.info(
        f"Running pymetis.part_graph (n_ranks={n_ranks}, ufactor={ufactor}, "
        f"|V|={graph.number_of_nodes():,}, |E|={graph.number_of_edges():,}, "
        f"weight_attribute={weight_attribute or 'none'})"
    )

    if hasattr(pymetis, "CSRAdjacency"):
        adjacency = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
        result = pymetis.part_graph(n_ranks, adjacency=adjacency, vweights=vweights, options=options)
    else:  # pragma: no cover - compatibility with older pymetis releases.
        result = pymetis.part_graph(n_ranks, xadj=xadj, adjncy=adjncy, vweights=vweights, options=options)
    edge_cut: int = result.edge_cuts
    vertex_part = result.vertex_part

    logger.info(f"METIS partitioning complete. Edge cut: {edge_cut:,}")

    partition: dict[int, int] = {int(node): int(vertex_part[i]) for i, node in enumerate(node_list)}

    distribution = pd.Series(list(partition.values())).value_counts().reindex(range(n_ranks), fill_value=0)
    rank_weights = dict.fromkeys(range(n_ranks), 0)
    for node, rank in partition.items():
        rank_weights[rank] += _vertex_weight(graph, node, weight_attribute)

    for rank, count in distribution.items():
        if weight_attribute is None:
            logger.info(f"  rank {int(rank)}: {int(count):,} places")
        else:
            logger.info(f"  rank {int(rank)}: {int(count):,} places, weight={rank_weights[int(rank)]:,}")

    return partition


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------


def _split_qualified_table(qualified: str) -> tuple[str | None, str]:
    """Split a (possibly schema-qualified) table name into (schema, table)."""
    parts = qualified.split(".")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise NetworkPartitionerError(_invalid_output_table_message(qualified))


def ensure_output_table(conn: duckdb.DuckDBPyConnection, qualified_name: str) -> None:
    """Create the partition output table (and its schema) if it does not exist."""
    schema, _ = _split_qualified_table(qualified_name)
    output_identifier = quote_table_identifier(qualified_name)
    if schema is not None:
        schema_identifier = quote_table_identifier(schema)
        query = f"CREATE SCHEMA IF NOT EXISTS {schema_identifier}"
        conn.execute(query)

    query = f"""
        CREATE TABLE IF NOT EXISTS {output_identifier} (
            imputation INTEGER NOT NULL,
            n_ranks    INTEGER NOT NULL,
            rank       INTEGER NOT NULL,
            place_id   BIGINT  NOT NULL
        )
        """
    conn.execute(query)


def count_existing_partition_rows(
    conn: duckdb.DuckDBPyConnection,
    qualified_name: str,
    imputation: int,
    n_ranks: int,
) -> int:
    """Return existing partition rows for an ``(imputation, n_ranks)`` pair."""
    if not check_if_table_exists(conn, qualified_name):
        return 0

    output_identifier = quote_table_identifier(qualified_name)
    query = f"""
        SELECT COUNT(*)
        FROM {output_identifier}
        WHERE imputation = ? AND n_ranks = ?
        """  # noqa: S608 - table identifier is validated by quote_table_identifier.
    result = conn.execute(query, [imputation, n_ranks]).fetchone()
    return int(result[0]) if result else 0


def write_partition_table(
    conn: duckdb.DuckDBPyConnection,
    qualified_name: str,
    imputation: int,
    n_ranks: int,
    partition: dict[int, int],
) -> int:
    """Write the partition assignment to the DuckLake.

    Existing rows for the same ``(imputation, n_ranks)`` combination are
    deleted first so re-running the partitioner is idempotent.

    Returns the number of rows inserted.
    """
    ensure_output_table(conn, qualified_name)
    output_identifier = quote_table_identifier(qualified_name)

    df = pd.DataFrame(
        {
            "imputation": pd.Series([imputation] * len(partition), dtype="int32"),
            "n_ranks": pd.Series([n_ranks] * len(partition), dtype="int32"),
            "rank": pd.Series(list(partition.values()), dtype="int32"),
            "place_id": pd.Series(list(partition.keys()), dtype="int64"),
        }
    )

    delete_query = f"""
        DELETE FROM {output_identifier}
        WHERE imputation = ? AND n_ranks = ?
        """  # noqa: S608 - table identifier is validated by quote_table_identifier.
    deleted = conn.execute(
        delete_query,
        [imputation, n_ranks],
    ).fetchone()

    deleted_count = int(deleted[0]) if deleted else 0
    if deleted_count:
        logger.info(
            f"Replaced {deleted_count:,} existing rows in {qualified_name} for "
            f"(imputation={imputation}, n_ranks={n_ranks})"
        )

    conn.register("partition_df", df)
    insert_query = f"""
        INSERT INTO {output_identifier} (imputation, n_ranks, rank, place_id)
        SELECT imputation, n_ranks, rank, place_id FROM partition_df
        """  # noqa: S608 - table identifier is validated by quote_table_identifier.
    conn.execute(insert_query)
    conn.unregister("partition_df")

    logger.info(
        f"Inserted {len(df):,} partition rows into {qualified_name} " f"(imputation={imputation}, n_ranks={n_ranks})"
    )
    return len(df)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def list_person_imputations(conn: duckdb.DuckDBPyConnection, persons_table: str) -> list[int]:
    """Return all distinct imputation values from the persons table.

    If the persons table has no ``Imputation`` column, return ``[1]`` so older
    single-imputation datasets keep the current behavior.
    """
    if "imputation" not in _table_columns(conn, persons_table):
        logger.info(f"Table {persons_table} has no Imputation column; defaulting to imputation 1.")
        return [1]

    persons_identifier = quote_table_identifier(persons_table)
    query = f"""
        SELECT DISTINCT CAST(Imputation AS INTEGER) AS imputation
        FROM {persons_identifier}
        WHERE Imputation IS NOT NULL
        ORDER BY imputation
        """  # noqa: S608 - table identifier is validated by quote_table_identifier.
    imputations = [int(row[0]) for row in conn.execute(query).fetchall()]
    if not imputations:
        raise NetworkPartitionerError(f"No imputation values found in {persons_table}")
    return imputations


def resolve_imputations(
    conn: duckdb.DuckDBPyConnection,
    persons_table: str,
    imputation_spec: str,
) -> list[int]:
    """Resolve an imputation CLI spec into concrete imputation values."""
    normalized = imputation_spec.strip().lower()
    if normalized == "all":
        return list_person_imputations(conn, persons_table)
    return _parse_positive_int_list(imputation_spec, name="imputations")


def _partition_rank_values_from_connection(
    conn: duckdb.DuckDBPyConnection,
    *,
    imputation: int,
    n_rank_values: list[int],
    persons_table: str,
    places_table: str,
    output_table: str,
    restrict_to_places: bool,
    ufactor: int,
    force: bool,
    weight_attribute: str | None,
) -> None:
    pending_rank_values = []
    for n_ranks in n_rank_values:
        existing_rows = count_existing_partition_rows(conn, output_table, imputation, n_ranks)
        if existing_rows and not force:
            logger.info(
                f"Partition rows already exist in {output_table} for "
                f"(imputation={imputation}, n_ranks={n_ranks}); "
                f"skipping regeneration ({existing_rows:,} rows). Use --force to replace them."
            )
            continue
        pending_rank_values.append(n_ranks)

    if not pending_rank_values:
        return

    graph = build_graph_from_ducklake(
        conn,
        persons_table,
        imputation,
        places_table=places_table,
        restrict_to_places=restrict_to_places,
    )
    for n_ranks in pending_rank_values:
        partition = partition_graph_with_pymetis(
            graph,
            n_ranks,
            ufactor=ufactor,
            weight_attribute=weight_attribute,
        )
        write_partition_table(conn, output_table, imputation, n_ranks, partition)


def partition_from_ducklake(
    ducklake_path: Path,
    schema: str,
    imputation: int,
    n_ranks: int,
    output_table: str,
    *,
    persons_table: str | None = None,
    places_table: str | None = None,
    restrict_to_places: bool = False,
    ufactor: int = 30,
    database_name: str = "insights_ducklake",
    force: bool = False,
    weight_by: str = "none",
) -> None:
    """End-to-end: connect, build graph, run METIS, write partition table.

    Args:
        ducklake_path: Directory containing ``metadata.sqlite`` and ``storage/``.
        schema: Catalog schema where the input tables live, e.g.
            ``rti_synth_pop_v2_dmv_100``. Used to build default
            ``persons_table`` / ``places_table`` names if not supplied.
        imputation: Imputation column value to partition for.
        n_ranks: Number of MPI processes the partition is being computed for.
        output_table: Destination table inside the DuckLake. May be
            ``schema.table`` or a bare ``table``.
        persons_table: Override for the persons table (defaults to
            ``{schema}.persons``).
        places_table: Override for the places table (defaults to
            ``{schema}.places``).
        restrict_to_places: When True, drops persons-only IDs that do not
            appear in the places table.
        ufactor: Load-imbalance tolerance for METIS (1000 + ufactor)/1000.
        database_name: DuckLake catalog name (matches
            ``get_ducklake_connection``).
        force: When True, regenerate and replace existing rows for the same
            ``(imputation, n_ranks)`` pair. When False, existing rows cause the
            run to skip before graph construction.
        weight_by: Workload weight strategy. ``"none"`` / ``"places"``
            preserves place-count balancing. ``"persons"`` balances METIS
            vertex weights by home-person counts per place.
    """
    persons_table = persons_table or f"{schema}.persons"
    places_table_resolved = places_table or f"{schema}.places"
    weight_attribute = _weight_attribute_for(weight_by)

    if not ducklake_path.exists():
        raise NetworkPartitionerError(_missing_ducklake_message(ducklake_path))

    logger.info(f"Connecting to DuckLake at {ducklake_path}")
    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    try:
        _partition_rank_values_from_connection(
            conn,
            imputation=imputation,
            n_rank_values=[n_ranks],
            persons_table=persons_table,
            places_table=places_table_resolved,
            output_table=output_table,
            restrict_to_places=restrict_to_places,
            ufactor=ufactor,
            force=force,
            weight_attribute=weight_attribute,
        )
    finally:
        conn.close()


def partition_many_from_ducklake(
    ducklake_path: Path,
    schema: str,
    imputation_spec: str,
    n_rank_values: list[int],
    output_table: str,
    *,
    persons_table: str | None = None,
    places_table: str | None = None,
    restrict_to_places: bool = False,
    ufactor: int = 30,
    database_name: str = "insights_ducklake",
    force: bool = False,
    weight_by: str = "none",
) -> None:
    """Partition one or more imputations for one or more MPI rank counts."""
    persons_table = persons_table or f"{schema}.persons"
    places_table_resolved = places_table or f"{schema}.places"
    weight_attribute = _weight_attribute_for(weight_by)

    if not ducklake_path.exists():
        raise NetworkPartitionerError(_missing_ducklake_message(ducklake_path))

    logger.info(f"Connecting to DuckLake at {ducklake_path}")
    conn = get_ducklake_connection(ducklake_path, database_name=database_name)
    try:
        imputation_values = resolve_imputations(conn, persons_table, imputation_spec)
        logger.info(f"Resolved imputations: {', '.join(str(value) for value in imputation_values)}")

        for imputation in imputation_values:
            _partition_rank_values_from_connection(
                conn,
                imputation=imputation,
                n_rank_values=n_rank_values,
                persons_table=persons_table,
                places_table=places_table_resolved,
                output_table=output_table,
                restrict_to_places=restrict_to_places,
                ufactor=ufactor,
                force=force,
                weight_attribute=weight_attribute,
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    ducklake_path: str | None = typer.Option(
        None,
        "--ducklake-path",
        help=(
            "Path to the DuckLake directory (contains metadata.sqlite + storage/). "
            "Defaults to $CASMSOCIAL_DUCKLAKE_PATH if set."
        ),
    ),
    schema: str = typer.Option(
        "rti_synth_pop_v2_dmv_100",
        "--schema",
        help="DuckLake schema for the input tables (e.g. rti_synth_pop_v2_dmv_100).",
    ),
    imputations: str = typer.Option(
        "all",
        "--imputations",
        "--imputation",
        help=(
            "Imputations to partition: 'all' or comma-separated values such as '1,2,3'. "
            "--imputation is kept as a backward-compatible alias."
        ),
    ),
    n_ranks: str = typer.Option("8", "--n-ranks", help="Comma-separated MPI rank counts, e.g. '2,4,8'."),
    output_table: str = typer.Option(
        "partitions.metis_place_partitions",
        "--output-table",
        help=(
            "Destination table inside the DuckLake. May be 'schema.table' or "
            "a bare 'table'. The schema is auto-created if needed."
        ),
    ),
    persons_table: str | None = typer.Option(
        None,
        "--persons-table",
        help="Override for the persons table; defaults to '{schema}.persons'.",
    ),
    places_table: str | None = typer.Option(
        None,
        "--places-table",
        help="Override for the places table; defaults to '{schema}.places'.",
    ),
    restrict_to_places: bool = typer.Option(
        False,
        "--restrict-to-places/--no-restrict-to-places",
        help=(
            "If set, only place IDs that appear in the places table are kept "
            "in the partition graph (drops persons-only IDs)."
        ),
    ),
    ufactor: int = typer.Option(
        30,
        "--ufactor",
        help="METIS load-imbalance tolerance: actual tolerance = (1000 + ufactor)/1000.",
    ),
    database_name: str = typer.Option(
        "insights_ducklake",
        "--database-name",
        help="DuckLake catalog name passed to get_ducklake_connection.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Regenerate and replace rows even if the output already has this imputation and rank count.",
    ),
    weight_by: str = typer.Option(
        "none",
        "--weight-by",
        help=(
            "Partition balance target: 'none' / 'places' balances place count; "
            "'persons' balances home-person vertex weights."
        ),
    ),
) -> None:
    """Partition the activity-location network for one or more imputations and write the
    assignment to a DuckLake table.

    Example:
        python -m casmsocial.network_partitioner_ducklake \\
            --schema rti_synth_pop_v2_dmv_100 \\
            --imputations all \\
            --n-ranks 2,4,8 \\
            --output-table partitions.metis_place_partitions
    """
    load_dotenv()

    if ducklake_path is None:
        ducklake_path = os.environ.get("CASMSOCIAL_DUCKLAKE_PATH")
    if ducklake_path is None:
        raise typer.BadParameter("--ducklake-path or CASMSOCIAL_DUCKLAKE_PATH is required")
    ducklake_dir = Path(ducklake_path).expanduser()
    n_rank_values = _parse_positive_int_list(n_ranks, name="n_ranks")

    logger.info("Starting METIS partitioning from DuckLake source")
    logger.info(f"  DuckLake:      {ducklake_dir}")
    logger.info(f"  Schema:        {schema}")
    logger.info(f"  Imputations:   {imputations}")
    logger.info(f"  Ranks:         {', '.join(str(value) for value in n_rank_values)}")
    logger.info(f"  Output table:  {output_table}")
    logger.info(f"  Weight by:     {weight_by}")
    if force:
        logger.info("  Force:         existing partition rows will be replaced")
    if restrict_to_places:
        logger.info("  Restriction:   place IDs limited to those in places table")

    partition_many_from_ducklake(
        ducklake_dir,
        schema,
        imputations,
        n_rank_values,
        output_table,
        persons_table=persons_table,
        places_table=places_table,
        restrict_to_places=restrict_to_places,
        ufactor=ufactor,
        database_name=database_name,
        force=force,
        weight_by=weight_by,
    )

    logger.info(f"Done. Partition rows are in {output_table}.")


if __name__ == "__main__":
    typer.run(main)
