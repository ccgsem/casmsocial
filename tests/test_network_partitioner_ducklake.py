import importlib.util
import warnings

import duckdb
import networkx as nx
import pytest

import casmsocial.network_partitioner_ducklake as partitioner
from casmsocial.network_partitioner_ducklake import (
    build_graph_from_ducklake,
    count_existing_partition_rows,
    list_person_imputations,
    partition_from_ducklake,
    partition_graph_with_pymetis,
    partition_many_from_ducklake,
    resolve_imputations,
    write_partition_table,
)

# pymetis has no prebuilt wheel for every platform (e.g. manylinux aarch64 at
# the time of writing) and is therefore an optional `partitioning` extra, not
# a core dependency -- guard the tests that actually exercise it.
PYMETIS_AVAILABLE = importlib.util.find_spec("pymetis") is not None


def _create_connection():
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE persons (
            Imputation INTEGER,
            sp_hh_id BIGINT,
            sp_school_id BIGINT,
            sp_work_id BIGINT
        )
        """)
    conn.execute("CREATE TABLE places (sp_id BIGINT)")
    return conn


class _NoCloseConnection:
    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def register(self, *args, **kwargs):
        return self._conn.register(*args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self._conn.unregister(*args, **kwargs)


@pytest.mark.skipif(not PYMETIS_AVAILABLE, reason="pymetis not installed (partitioning extra)")
def test_partition_graph_with_pymetis_uses_non_deprecated_csr_adjacency():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        partition = partition_graph_with_pymetis(graph, n_ranks=2)

    xadj_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, DeprecationWarning) and "xadj/adjncy" in str(warning.message)
    ]
    assert xadj_warnings == []
    assert sorted(partition) == [1, 2, 3, 4]
    assert set(partition.values()) <= {0, 1}


@pytest.mark.skipif(not PYMETIS_AVAILABLE, reason="pymetis not installed (partitioning extra)")
def test_partition_graph_with_pymetis_passes_vertex_weights(monkeypatch):
    import pymetis

    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    graph.nodes[1][partitioner.PERSON_WEIGHT_ATTR] = 4
    captured = {}

    class _Result:
        edge_cuts = 0
        vertex_part = [0, 1]

    def fake_part_graph(nparts, **kwargs):
        captured["nparts"] = nparts
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(pymetis, "part_graph", fake_part_graph)

    partition = partition_graph_with_pymetis(
        graph,
        n_ranks=2,
        weight_attribute=partitioner.PERSON_WEIGHT_ATTR,
    )

    assert partition == {1: 0, 2: 1}
    assert captured["nparts"] == 2
    assert captured["vweights"] == [4, 1]


def test_count_existing_partition_rows_returns_matching_rows_only():
    conn = duckdb.connect(":memory:")
    try:
        write_partition_table(
            conn,
            "partition_output",
            imputation=1,
            n_ranks=2,
            partition={1: 0, 2: 1},
        )
        write_partition_table(
            conn,
            "partition_output",
            imputation=2,
            n_ranks=2,
            partition={3: 0},
        )

        assert count_existing_partition_rows(conn, "partition_output", imputation=1, n_ranks=2) == 2
        assert count_existing_partition_rows(conn, "partition_output", imputation=2, n_ranks=2) == 1
        assert count_existing_partition_rows(conn, "partition_output", imputation=1, n_ranks=4) == 0
        assert count_existing_partition_rows(conn, "missing_output", imputation=1, n_ranks=2) == 0
    finally:
        conn.close()


def test_parse_positive_int_list_deduplicates_values():
    assert partitioner._parse_positive_int_list("2, 4, 2,8", name="n_ranks") == [2, 4, 8]


def test_parse_positive_int_list_rejects_invalid_values():
    try:
        partitioner._parse_positive_int_list("2,nope", name="n_ranks")
    except partitioner.NetworkPartitionerError as exc:
        assert "comma-separated integers" in str(exc)
    else:
        raise AssertionError("invalid rank list should fail")


def test_resolve_imputations_all_loads_distinct_values():
    conn = _create_connection()
    try:
        conn.execute("""
            INSERT INTO persons VALUES
                (2, 1, NULL, NULL),
                (1, 1, NULL, NULL),
                (2, 2, NULL, NULL)
            """)

        assert list_person_imputations(conn, "persons") == [1, 2]
        assert resolve_imputations(conn, "persons", "all") == [1, 2]
        assert resolve_imputations(conn, "persons", "2,1,2") == [2, 1]
    finally:
        conn.close()


def test_resolve_imputations_all_defaults_to_one_without_imputation_column():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("""
            CREATE TABLE persons (
                sp_hh_id BIGINT,
                sp_school_id BIGINT,
                sp_work_id BIGINT
            )
            """)

        assert list_person_imputations(conn, "persons") == [1]
        assert resolve_imputations(conn, "persons", "all") == [1]
    finally:
        conn.close()


def test_partition_from_ducklake_skips_when_output_already_generated(monkeypatch, tmp_path):
    conn = duckdb.connect(":memory:")
    write_partition_table(
        conn,
        "partition_output",
        imputation=1,
        n_ranks=2,
        partition={1: 0, 2: 1},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("graph should not be rebuilt when output exists")

    monkeypatch.setattr(partitioner, "get_ducklake_connection", lambda *args, **kwargs: conn)
    monkeypatch.setattr(partitioner, "build_graph_from_ducklake", fail_if_called)

    partition_from_ducklake(
        tmp_path,
        schema="demo",
        imputation=1,
        n_ranks=2,
        output_table="partition_output",
    )


def test_partition_many_from_ducklake_reuses_graph_per_imputation(monkeypatch, tmp_path):
    conn = _create_connection()
    conn.execute("""
        INSERT INTO persons VALUES
            (1, 1, 2, NULL),
            (2, 2, 3, NULL)
        """)
    conn.execute("INSERT INTO places VALUES (1), (2), (3)")
    build_calls = []

    def fake_build_graph(conn_arg, persons_table, imputation, **kwargs):
        build_calls.append((persons_table, imputation, kwargs))
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3)])
        return graph

    def fake_partition_graph(graph, n_ranks, **kwargs):
        return {node: index % n_ranks for index, node in enumerate(sorted(graph.nodes()))}

    monkeypatch.setattr(partitioner, "get_ducklake_connection", lambda *args, **kwargs: _NoCloseConnection(conn))
    monkeypatch.setattr(partitioner, "build_graph_from_ducklake", fake_build_graph)
    monkeypatch.setattr(partitioner, "partition_graph_with_pymetis", fake_partition_graph)

    partition_many_from_ducklake(
        tmp_path,
        schema="demo",
        imputation_spec="all",
        n_rank_values=[2, 4],
        output_table="partition_output",
        persons_table="persons",
        places_table="places",
    )

    assert [(call[0], call[1]) for call in build_calls] == [("persons", 1), ("persons", 2)]
    rows = conn.execute("""
        SELECT imputation, n_ranks, COUNT(*)
        FROM partition_output
        GROUP BY imputation, n_ranks
        ORDER BY imputation, n_ranks
        """).fetchall()
    assert rows == [(1, 2, 3), (1, 4, 3), (2, 2, 3), (2, 4, 3)]


def test_build_graph_from_ducklake_loads_persons_without_imputation_column():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("""
            CREATE TABLE persons (
                sp_hh_id BIGINT,
                sp_school_id BIGINT,
                sp_work_id BIGINT
            )
            """)
        conn.execute("CREATE TABLE places (sp_id BIGINT)")
        conn.execute("INSERT INTO persons VALUES (1, 2, NULL)")
        conn.execute("INSERT INTO places VALUES (1), (2)")

        graph = build_graph_from_ducklake(
            conn,
            "persons",
            1,
            places_table="places",
        )

        assert sorted(graph.nodes()) == [1, 2]
        assert graph.has_edge(1, 2)
    finally:
        conn.close()


def test_build_graph_from_ducklake_seeds_isolated_places():
    conn = _create_connection()
    try:
        conn.execute("INSERT INTO persons VALUES (1, 1, 2, NULL)")
        conn.execute("INSERT INTO places VALUES (1), (2), (3)")

        graph = build_graph_from_ducklake(
            conn,
            "persons",
            1,
            places_table="places",
        )

        assert sorted(graph.nodes()) == [1, 2, 3]
        assert graph.has_edge(1, 2)
        assert graph.degree[3] == 0
    finally:
        conn.close()


def test_build_graph_from_ducklake_records_home_person_weights():
    conn = _create_connection()
    try:
        conn.execute("""
            INSERT INTO persons VALUES
                (1, 1, 2, NULL),
                (1, 1, NULL, NULL),
                (1, 2, NULL, NULL)
            """)
        conn.execute("INSERT INTO places VALUES (1), (2), (3)")

        graph = build_graph_from_ducklake(
            conn,
            "persons",
            1,
            places_table="places",
        )

        assert graph.nodes[1][partitioner.PERSON_WEIGHT_ATTR] == 2
        assert graph.nodes[2][partitioner.PERSON_WEIGHT_ATTR] == 1
        assert graph.nodes[3].get(partitioner.PERSON_WEIGHT_ATTR, 0) == 0
    finally:
        conn.close()


def test_build_graph_from_ducklake_keeps_unknown_person_places_by_default():
    conn = _create_connection()
    try:
        conn.execute("INSERT INTO persons VALUES (1, 1, 99, NULL)")
        conn.execute("INSERT INTO places VALUES (1), (2)")

        graph = build_graph_from_ducklake(
            conn,
            "persons",
            1,
            places_table="places",
        )

        assert sorted(graph.nodes()) == [1, 2, 99]
        assert graph.has_edge(1, 99)
    finally:
        conn.close()


def test_build_graph_from_ducklake_drops_unknown_person_places_when_restricted():
    conn = _create_connection()
    try:
        conn.execute("INSERT INTO persons VALUES (1, 1, 99, NULL)")
        conn.execute("INSERT INTO places VALUES (1), (2)")

        graph = build_graph_from_ducklake(
            conn,
            "persons",
            1,
            places_table="places",
            restrict_to_places=True,
        )

        assert sorted(graph.nodes()) == [1, 2]
        assert not graph.has_node(99)
        assert not graph.has_edge(1, 99)
    finally:
        conn.close()


def test_write_partition_table_replaces_existing_partition_rows():
    conn = duckdb.connect(":memory:")
    try:
        first_count = write_partition_table(
            conn,
            "partition_output",
            imputation=1,
            n_ranks=2,
            partition={1: 0, 2: 1},
        )
        second_count = write_partition_table(
            conn,
            "partition_output",
            imputation=1,
            n_ranks=2,
            partition={1: 1, 2: 1, 3: 0},
        )

        rows = conn.execute("""
            SELECT place_id, rank
            FROM partition_output
            WHERE imputation = 1 AND n_ranks = 2
            ORDER BY place_id
            """).fetchall()

        assert first_count == 2
        assert second_count == 3
        assert rows == [(1, 1), (2, 1), (3, 0)]
    finally:
        conn.close()
