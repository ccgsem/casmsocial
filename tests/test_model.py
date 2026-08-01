from __future__ import annotations

import pathlib
from datetime import datetime
from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from loguru import logger
from mpi4py import MPI

from casmsocial.activities import DEFAULT_TRAVEL_LEG, Act, Leg
from casmsocial.casmpop import (
    AgentLogger,
    BehaviorLogger,
    CasmPop,
    DeltaAgentStateLogger,
    InvalidObserverOutputFileError,
    InvalidPartitionRankError,
    InvalidSocialNetworkTableError,
    MissingPartitionAssignmentError,
    MissingRequiredTableError,
    SimEnvironment,
    ScheduleOccupancyLogger,
    SocialInteractionLogger,
)
from casmsocial.communication.types import CommMessage, MessageKind, build_message_payload
from casmsocial.household import Household
from casmsocial.model import Model
from casmsocial.person import BehaviorEngineV2, LLMBehaviorEngine, Person, ScheduleBehaviorEngine
from casmsocial.place import EnhancedPlacesProjection, Place
from casmsocial.road_network import RoadRoute
from casmsocial.sim_time import SimTime
from casmsocial.social_interactions import InteractionEvent, PresenceInterval, SocialTie


class DummyModel(Model):
    def __init__(self, comm, params):
        self.comm = comm
        self.params = params

    def start(self) -> None:
        pass

    def step(self) -> None:
        pass


def test_model_singleton_helpers():
    Model.theModel = None
    dummy = DummyModel(None, {})

    dummy.set_model(dummy)

    assert Model.get_model() is dummy


class FakeRoadNetwork:
    def route_between_places(self, origin_place_id, destination_place_id, mode="drive"):
        return RoadRoute(
            origin_place_id=origin_place_id,
            destination_place_id=destination_place_id,
            origin_node_id=11,
            destination_node_id=22,
            distance_m=5000.0,
            travel_time_min=12,
            node_ids=(11, 22),
        )


class SlowFakeRoadNetwork:
    def route_between_places(self, origin_place_id, destination_place_id, mode="drive"):
        return RoadRoute(
            origin_place_id=origin_place_id,
            destination_place_id=destination_place_id,
            origin_node_id=11,
            destination_node_id=22,
            distance_m=5000.0,
            travel_time_min=45,
            node_ids=(11, 22),
        )


class _StubContext:
    def __init__(self, projection):
        self.projection = projection

    def get_projection(self, name):
        assert name == "places_projection"
        return self.projection


class _AgentContext:
    def __init__(self, agents):
        self._agents = agents

    def add(self, agent):
        self._agents.append(agent)

    def agents(self, agent_type=None):
        if agent_type is None:
            return iter(self._agents)
        return (agent for agent in self._agents if agent.type == agent_type)


class _EmptyTypedContext:
    def agents(self, agent_type=None):
        raise KeyError(agent_type)


class _FastAgentManager:
    def __init__(self):
        self.added = []
        self._local_agents = {}
        self.rank = 0

    def add_local(self, agent):
        self.added.append(agent)
        self._local_agents[agent.uid] = agent


class _FastContext:
    def __init__(self, projection):
        self._agent_manager = _FastAgentManager()
        self._agents_by_type = {}
        self.projections = {"places_projection": projection}
        self.added_via_public_api = []

    def add(self, agent):
        self.added_via_public_api.append(agent)


class _AllreduceComm:
    def __init__(self, total: int):
        self.total = total
        self.allreduce_calls = 0
        self.last_value = None

    def allreduce(self, value, op=None):
        self.allreduce_calls += 1
        self.last_value = value
        return self.total


class _GatherComm:
    def __init__(self, gathered):
        self.gathered = gathered
        self.gather_calls = 0
        self.last_value = None
        self.last_root = None

    def gather(self, value, root=0):
        self.gather_calls += 1
        self.last_value = value
        self.last_root = root
        return self.gathered


class _BcastComm:
    def __init__(self, value=None):
        self.value = value
        self.bcast_calls = 0
        self.last_value = None
        self.last_root = None

    def bcast(self, value, root=0):
        self.bcast_calls += 1
        self.last_value = value
        self.last_root = root
        return self.value if self.value is not None else value


class _TrackingCommunicationManager:
    def __init__(self):
        self.clear_calls = 0
        self.route_calls = 0
        self.exchange_remote_calls = 0
        self.exchange_ack_calls = 0

    def clear_buffers(self):
        self.clear_calls += 1

    def route(self, intents, model, tick):
        self.route_calls += 1
        self.routed_intents = list(intents)

    def exchange_remote(self, model):
        self.exchange_remote_calls += 1

    def exchange_acks(self, model):
        self.exchange_ack_calls += 1


class _TickCal:
    tick = 12


class _MigrationContext:
    def __init__(self):
        self.move_agents_calls = 0
        self.synchronize_calls = 0
        self.moves = None
        self._agents = []
        self.projection = None

    def get_projection(self, name):
        assert name == "places_projection"
        return self.projection

    def agents(self, agent_type=None):
        return iter(self._agents)

    def move_agents(self, moves, restore_agent):
        self.move_agents_calls += 1
        self.moves = list(moves)

    def synchronize(self, restore_agent):
        self.synchronize_calls += 1


class _ProjectionContext:
    def __init__(self):
        self.projections = []

    def add_projection(self, projection):
        self.projections.append(projection)


class _TablesConn:
    def execute(self, query):
        return self

    def fetchall(self):
        return []


class _MovingPerson:
    def __init__(self, uid=(1, 0, 0), moved=True):
        self.uid = uid
        self.moved = moved
        self.move_calls = 0

    def move(self, cal, places_proj):
        self.move_calls += 1
        return self.moved


def _sim_time_at(minute_of_day: int) -> SimTime:
    cal = SimTime(datetime(2025, 1, 1))
    cal.set_datetime(datetime(2025, 1, 1, minute_of_day // 60, minute_of_day % 60))
    return cal


def test_build_primary_plan_falls_back_without_road_network():
    model = CasmPop.__new__(CasmPop)
    model.road_network = None
    model.params = {"roads.mode": "drive"}

    schedule = [
        Act(1, 0, 0, 0, 60, 100),
        Act(1, 1, 1, 120, 180, 200),
    ]

    plan = model._build_primary_plan(1, schedule, [100, 200])

    assert len(plan) == 3
    assert isinstance(plan[1], Leg)
    assert plan[1].mode == "travel"
    assert plan[1].travel_time_min is None


def test_build_primary_plan_uses_routing_when_available():
    model = CasmPop.__new__(CasmPop)
    model.road_network = FakeRoadNetwork()
    model.params = {"roads.mode": "drive"}

    schedule = [
        Act(1, 0, 0, 0, 60, 100),
        Act(1, 1, 1, 120, 180, 200),
    ]

    plan = model._build_primary_plan(1, schedule, [100, 200])

    assert len(plan) == 3
    assert isinstance(plan[1], Leg)
    assert plan[1].mode == "drive"
    assert plan[1].origin_place_id == 100
    assert plan[1].destination_place_id == 200
    assert plan[1].travel_time_min == 12


def test_build_primary_plan_logs_infeasible_routed_leg():
    model = CasmPop.__new__(CasmPop)
    model.road_network = SlowFakeRoadNetwork()
    model.params = {"roads.mode": "drive"}

    schedule = [
        Act(1, 0, 0, 0, 60, 100),
        Act(1, 1, 1, 90, 180, 200),
    ]

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        model._build_primary_plan(1, schedule, [100, 200])
    finally:
        logger.remove(sink_id)

    assert any("infeasible leg" in message for message in messages)


def test_load_records_from_duckdb_table(tmp_path):
    model = CasmPop.__new__(CasmPop)
    model.conn = duckdb.connect()
    model.data_path = pathlib.Path(tmp_path)

    model.conn.execute("CREATE TABLE road_nodes_test (node_id INTEGER, x DOUBLE, y DOUBLE)")
    model.conn.execute("INSERT INTO road_nodes_test VALUES (1, 10.0, 20.0), (2, 30.0, 40.0)")

    records = model._load_records("road_nodes_test", "road nodes")

    assert records == [
        {"node_id": 1, "x": 10.0, "y": 20.0},
        {"node_id": 2, "x": 30.0, "y": 40.0},
    ]


def test_load_records_from_parquet_file(tmp_path):
    model = CasmPop.__new__(CasmPop)
    model.conn = duckdb.connect()
    model.data_path = pathlib.Path(tmp_path)

    parquet_path = tmp_path / "road_edges.parquet"
    table = pa.table(
        {
            "edge_id": [1],
            "from_node_id": [10],
            "to_node_id": [20],
            "length_m": [123.4],
            "travel_time_min": [5.5],
        }
    )
    pq.write_table(table, parquet_path)

    records = model._load_records("road_edges.parquet", "road edges")

    assert records == [
        {
            "edge_id": 1,
            "from_node_id": 10,
            "to_node_id": 20,
            "length_m": 123.4,
            "travel_time_min": 5.5,
        }
    ]


def test_load_records_raises_for_missing_source(tmp_path):
    model = CasmPop.__new__(CasmPop)
    model.conn = duckdb.connect()
    model.data_path = pathlib.Path(tmp_path)

    with pytest.raises(MissingRequiredTableError):
        model._load_records("missing_nodes.parquet", "road nodes")


def test_load_records_raises_for_unsupported_file_type(tmp_path):
    model = CasmPop.__new__(CasmPop)
    model.conn = duckdb.connect()
    model.data_path = pathlib.Path(tmp_path)

    bad_path = tmp_path / "road_nodes.csv"
    bad_path.write_text("node_id,x,y\n1,10,20\n")

    with pytest.raises(ValueError, match="Unsupported road nodes source"):
        model._load_records("road_nodes.csv", "road nodes")


def _create_partitioned_input_tables(conn):
    conn.execute("CREATE SCHEMA input")
    conn.execute("CREATE SCHEMA partitions")
    conn.execute("""
        CREATE TABLE input.places (
            sp_id BIGINT,
            place_name VARCHAR
        )
        """)
    conn.execute("""
        CREATE TABLE input.persons (
            sp_id BIGINT,
            sp_hh_id BIGINT,
            sp_work_id BIGINT,
            sp_school_id BIGINT
        )
        """)
    conn.execute("""
        CREATE TABLE input.activities (
            sp_persons_id BIGINT,
            activity_id INTEGER,
            activity_sequence INTEGER,
            starttime_min INTEGER,
            endtime_min INTEGER,
            sp_act_id BIGINT
        )
        """)
    conn.execute("""
        CREATE TABLE partitions.metis_place_partitions (
            imputation INTEGER,
            n_ranks INTEGER,
            rank INTEGER,
            place_id BIGINT
        )
        """)
    conn.execute("INSERT INTO input.places VALUES (100, 'home-a'), (200, 'home-b'), (300, 'work')")
    conn.execute("INSERT INTO input.persons VALUES (1, 100, 300, NULL), (2, 200, 300, NULL)")
    conn.execute("""
        INSERT INTO input.activities VALUES
            (1, 0, 0, 0, 60, 100),
            (2, 0, 0, 0, 60, 200)
        """)


def _partitioned_model(rank=0, size=2, require_full_coverage=False):
    model = CasmPop.__new__(CasmPop)
    model.conn = duckdb.connect(":memory:")
    model.rank = rank
    model.size = size
    model.params = {
        "places.table": "input.places",
        "persons.table": "input.persons",
        "activities.table": "input.activities",
        "partition.table": "partitions.metis_place_partitions",
        "partition.default_rank": 0,
        "partition.require_full_coverage": require_full_coverage,
    }
    _create_partitioned_input_tables(model.conn)
    return model


def _duckdb_table_exists(conn, schema: str, table: str) -> bool:
    return conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM duckdb_tables
            WHERE schema_name = ?
              AND table_name = ?
        )
        """,
        [schema, table],
    ).fetchone()[0]


def test_default_parameters_disable_social_network_loading():
    assert CasmPop.get_default_parameters()["social_networks.enabled"] is False


def test_default_parameters_enable_communication_for_compatibility():
    assert CasmPop.get_default_parameters()["communication.enabled"] is True


def test_default_parameters_enable_agent_logging_for_compatibility():
    assert CasmPop.get_default_parameters()["observers.agent_log.enabled"] is True


def test_default_parameters_disable_phase_timing_profiling():
    assert CasmPop.get_default_parameters()["profiling.phase_timings.enabled"] is False
    assert CasmPop.get_default_parameters()["profiling.phase_timings.sample_stride"] == 512


def test_optional_params_fill_observer_partition_and_logging_defaults():
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "places.table": "input.places",
        "persons.table": "input.persons",
        "activities.table": "input.activities",
    }

    model._set_optional_params_with_defaults()

    assert model.params["random.seed"] == 42
    assert model.params["simulation.run_id"] == "seed_42"
    assert model.params["social_networks.enabled"] is False
    assert model.params["social_networks.remote_messages.enabled"] is False
    assert model.params["social_networks.remote_messages.interval_minutes"] == 60
    assert model.params["communication.enabled"] is True
    assert model.params["partition.table"] == ""
    assert model.params["partition.default_rank"] == 0
    assert model.params["partition.require_full_coverage"] is False
    assert model.params["observers.output_dir"] == "data/output"
    assert model.params["observers.agent_log.enabled"] is True
    assert model.params["observers.agent_log_file"] == "agent_log.parquet"
    assert model.params["observers.behavior_log.enabled"] is False
    assert model.params["observers.behavior_log_file"] == "behavior_log.parquet"
    assert model.params["logging.rank0_only"] is False
    assert model.params["logging.level"] == "INFO"
    assert model.params["profiling.phase_timings.enabled"] is False
    assert model.params["profiling.phase_timings.sample_stride"] == 512


@pytest.mark.parametrize(
    "value",
    [
        "output/agent_log.parquet",
        "/tmp/agent_log.parquet",
        r"output\agent_log.parquet",
    ],
)
def test_optional_params_reject_observer_output_files_with_directories(value):
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "places.table": "input.places",
        "persons.table": "input.persons",
        "activities.table": "input.activities",
        "observers.agent_log_file": value,
    }

    with pytest.raises(InvalidObserverOutputFileError, match="filename only"):
        model._set_optional_params_with_defaults()


def test_phase_timer_records_elapsed_time_when_enabled():
    model = CasmPop.__new__(CasmPop)
    model.params = {"profiling.phase_timings.enabled": True}
    model.phase_timings = {}

    result = model._time_phase("startup.test_phase", lambda value: value + 1, 41)

    assert result == 42
    assert model.phase_timings["startup.test_phase"] >= 0.0


def test_phase_timer_skips_recording_when_disabled():
    model = CasmPop.__new__(CasmPop)
    model.params = {"profiling.phase_timings.enabled": False}
    model.phase_timings = {}

    result = model._time_phase("startup.test_phase", lambda: "ok")

    assert result == "ok"
    assert model.phase_timings == {}


def test_phase_timing_summary_gathers_rank_timings():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.params = {"profiling.phase_timings.enabled": True}
    model.phase_timings = {"startup.create_persons": 1.0}
    model.comm = _GatherComm(
        [
            {"startup.create_persons": 1.0},
            {"startup.create_persons": 3.0, "tick.communication": 2.0},
        ]
    )

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        model._log_phase_timing_summary()
    finally:
        logger.remove(sink_id)

    assert model.comm.gather_calls == 1
    assert model.comm.last_value == {"startup.create_persons": 1.0}
    assert model.comm.last_root == 0
    assert any("startup.create_persons: min=1.000, avg=2.000, max=3.000" in message for message in messages)
    assert any("tick.communication: min=0.000, avg=1.000, max=2.000" in message for message in messages)


def test_fast_context_person_adder_uses_shared_context_indexes_when_only_places_projection():
    projection = object()
    model = CasmPop.__new__(CasmPop)
    model.places_proj = projection
    model.context = _FastContext(projection)
    person = Person.from_default_schedule_zero_location(1, 0, [], (100, None, None))

    add_person = model._fast_context_person_adder()
    add_person(person)

    assert model.context.added_via_public_api == []
    assert model.context._agent_manager.added == []
    assert list(model.context._agent_manager._local_agents.values()) == [person]
    assert list(model.context._agents_by_type[Person.TYPE].values()) == [person]


def test_fast_context_person_adder_falls_back_when_other_projections_are_registered():
    projection = object()
    model = CasmPop.__new__(CasmPop)
    model.places_proj = projection
    model.context = _FastContext(projection)
    model.context.projections["other_projection"] = object()
    person = Person.from_default_schedule_zero_location(1, 0, [], (100, None, None))

    add_person = model._fast_context_person_adder()
    add_person(person)

    assert model.context.added_via_public_api == [person]
    assert model.context._agent_manager.added == []
    assert model.context._agent_manager._local_agents == {}
    assert model.context._agents_by_type == {}


def test_create_input_tables_assigns_place_ranks_from_partition_table():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("ALTER TABLE input.places ADD COLUMN unused_place VARCHAR DEFAULT 'unused'")
        model.conn.execute("ALTER TABLE input.persons ADD COLUMN unused_person VARCHAR DEFAULT 'unused'")
        model.conn.execute("ALTER TABLE input.activities ADD COLUMN unused_activity VARCHAR DEFAULT 'unused'")
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 1, 300)
            """)

        model.create_input_tables()

        places = model.conn.execute("SELECT sp_id, rank FROM places ORDER BY sp_id").fetchall()
        place_ranks = model.conn.execute("SELECT sp_id, rank FROM place_ranks ORDER BY sp_id").fetchall()
        persons = model.conn.execute("SELECT sp_id, rank FROM persons ORDER BY sp_id").fetchall()
        activities = model.conn.execute("SELECT sp_persons_id FROM activities ORDER BY sp_persons_id").fetchall()
        assert places == [(200, 0)]
        assert place_ranks == [(100, 1), (200, 0), (300, 1)]
        assert persons == [(2, 0)]
        assert activities == [(2,)]
        assert "unused_place" not in [
            column[0] for column in model.conn.execute("SELECT * FROM places LIMIT 0").description
        ]
        assert "unused_person" not in [
            column[0] for column in model.conn.execute("SELECT * FROM persons LIMIT 0").description
        ]
        assert "unused_activity" not in [
            column[0] for column in model.conn.execute("SELECT * FROM activities LIMIT 0").description
        ]
    finally:
        model.conn.close()


def test_create_input_tables_materializes_local_households_from_configured_table():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            CREATE TABLE input.households (
                sp_id BIGINT,
                household_id BIGINT,
                household_size INTEGER,
                household_income DOUBLE,
                household_type VARCHAR,
                household_race VARCHAR,
                household_age INTEGER
            )
            """)
        model.conn.execute("""
            INSERT INTO input.households VALUES
                (200, 10, 1, 75000.0, 'single', 'unknown', 35),
                (100, 20, 2, 90000.0, 'family', 'unknown', 42)
            """)
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.params["households.table"] = "input.households"

        model.create_input_tables()

        rows = model.conn.execute("""
            SELECT household_id, place_id, rank
            FROM households
            ORDER BY household_id
            """).fetchall()
        rank_rows = model.conn.execute("""
            SELECT household_id, place_id, rank
            FROM household_ranks
            ORDER BY household_id
            """).fetchall()
        assert rows == [(10, 200, 0)]
        assert rank_rows == [(10, 200, 0), (20, 100, 1)]
    finally:
        model.conn.close()


def test_create_input_tables_uses_sp_home_id_as_household_place_alias():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            CREATE TABLE input.households (
                sp_id BIGINT,
                sp_home_id BIGINT,
                household_size INTEGER
            )
            """)
        model.conn.execute("""
            INSERT INTO input.households VALUES
                (10, 200, 1),
                (20, 100, 2)
            """)
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.params["households.table"] = "input.households"

        model.create_input_tables()

        rows = model.conn.execute("""
            SELECT household_id, place_id, rank
            FROM households
            ORDER BY household_id
            """).fetchall()
        rank_rows = model.conn.execute("""
            SELECT household_id, place_id, rank
            FROM household_ranks
            ORDER BY household_id
            """).fetchall()
        assert rows == [(10, 200, 0)]
        assert rank_rows == [(10, 200, 0), (20, 100, 1)]
    finally:
        model.conn.close()


def test_create_places_with_partition_instantiates_local_places_and_indexes_all_ranks():
    model = _partitioned_model(rank=0, size=2)
    model.places_proj = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    try:
        model.conn.execute("ALTER TABLE input.places ADD COLUMN unused_place VARCHAR DEFAULT 'unused'")
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 1, 300)
            """)

        model.create_input_tables()
        model.create_places()
        model._initialize_place_rank_index()

        local_place_ids = sorted(place.id for place in model.places_proj.get_all_places())
        assert local_place_ids == [200]
        assert model.places_proj.lookup_place(200).data.place_name == "home-b"
        assert model.place_to_rank == {100: 1, 200: 0, 300: 1}
        assert model.places_proj.rank_for_place(100) == 1
        assert model.places_proj.rank_for_place(200) == 0
    finally:
        model.conn.close()


def test_create_persons_materializes_local_people_from_column_batches():
    CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
    CasmPop.register_activity_names(["home", "work", "school"])
    model = _partitioned_model(rank=0, size=2)
    original_behavior_engine = Person.getBehaviorEngine()
    Person.registerBehaviorEngine(ScheduleBehaviorEngine)
    model.places_proj = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    model.context = _AgentContext([])
    model.road_network = None
    model.params["roads.mode"] = "drive"
    model.params["communication.enabled"] = False
    model.params["profiling.phase_timings.enabled"] = True
    model.phase_timings = {}
    try:
        model.conn.execute("ALTER TABLE input.persons ADD COLUMN unused_payload VARCHAR DEFAULT 'unused'")
        model.conn.execute("ALTER TABLE input.activities ADD COLUMN unused_activity VARCHAR DEFAULT 'unused'")
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.create_input_tables()
        model.create_places()
        model._initialize_place_rank_index()

        model.create_persons(None)

        assert [person.id for person in model.context._agents] == [2]
        person = model.context._agents[0]
        assert person.plans[0][0].person_id == 2
        assert type(person.state.places) is tuple
        assert person.state.places == (200, 300, None)
        assert person.state.place_id == 200
        assert "inbox" not in person.__dict__
        assert "pending_acks" not in person.__dict__
        assert model.places_proj.lookup_place(200).has_occupant(person)
        assert "startup.create_persons.construct_agents.sampled_total.build_payload" in model.phase_timings
        assert "startup.create_persons.construct_agents.sampled_total.resolve_places" in model.phase_timings
        assert "startup.create_persons.construct_agents.sampled_total.build_plans" in model.phase_timings
        assert "startup.create_persons.construct_agents.sampled_total.init_person" in model.phase_timings
        assert "startup.create_persons.construct_agents.sampled_total.attach_person" in model.phase_timings
        assert "startup.create_persons.construct_agents.build_payload" not in model.phase_timings
        assert "startup.create_persons.construct_agents.resolve_places" not in model.phase_timings
        assert "startup.create_persons.construct_agents.build_plans" not in model.phase_timings
        assert "startup.create_persons.construct_agents.init_person" not in model.phase_timings
        assert "startup.create_persons.construct_agents.attach_person" not in model.phase_timings
        assert (
            "startup.create_persons.construct_agents.build_plans.sampled_total.lookup_schedule" in model.phase_timings
        )
        assert "startup.create_persons.construct_agents.build_plans.sampled_total.primary_plan" in model.phase_timings
        assert "startup.create_persons.construct_agents.build_plans.sampled_total.empty_plans" in model.phase_timings
        assert "startup.create_persons.construct_agents.build_plans.sampled_total.places_tuple" in model.phase_timings
    finally:
        Person.registerBehaviorEngine(original_behavior_engine)
        model.conn.close()


def test_create_households_links_places_and_person_members():
    CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
    CasmPop.register_activity_names(["home", "work", "school"])
    model = _partitioned_model(rank=0, size=2)
    original_behavior_engine = Person.getBehaviorEngine()
    Person.registerBehaviorEngine(ScheduleBehaviorEngine)
    model.places_proj = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    model.context = _AgentContext([])
    model.road_network = None
    model.params["communication.enabled"] = False
    model.params["households.table"] = "input.households"
    model.params["partition.default_rank"] = 1
    try:
        model.conn.execute("""
            CREATE TABLE input.households (
                sp_id BIGINT,
                household_id BIGINT,
                household_size INTEGER,
                household_income DOUBLE,
                household_type VARCHAR,
                household_race VARCHAR,
                household_age INTEGER
            )
            """)
        model.conn.execute("""
            INSERT INTO input.households VALUES
                (200, 10, 1, 75000.0, 'single', 'unknown', 35)
            """)
        model.conn.execute("UPDATE input.persons SET sp_hh_id = 10 WHERE sp_id = 2")
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)

        model.create_input_tables()
        model.create_places()
        model._initialize_place_rank_index()
        model.create_households()
        model.create_persons(None)

        households = list(model.context.agents(agent_type=Household.TYPE))
        people = list(model.context.agents(agent_type=Person.TYPE))
        household = model.households_by_id[10]
        person = people[0]

        assert households == [household]
        assert household.getPlace() == model.places_proj.lookup_place(200)
        assert model.households_by_place_id[200] == household
        assert household.getHouseholdMembers() == [person]
        assert person.state.places == (200, 300, None)
        assert person.state.place_id == 200
        assert model.places_proj.lookup_place(200).has_occupant(person)
    finally:
        Person.registerBehaviorEngine(original_behavior_engine)
        model.conn.close()


def test_create_persons_fast_path_populates_direct_context_and_projection_indexes():
    CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
    CasmPop.register_activity_names(["home", "work", "school"])
    model = _partitioned_model(rank=0, size=2)
    original_behavior_engine = Person.getBehaviorEngine()
    Person.registerBehaviorEngine(ScheduleBehaviorEngine)
    model.places_proj = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    model.context = _FastContext(model.places_proj)
    model.road_network = None
    model.params["communication.enabled"] = False
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.create_input_tables()
        model.create_places()
        model._initialize_place_rank_index()

        model.create_persons(None)

        persons = list(model.context._agent_manager._local_agents.values())
        assert [person.id for person in persons] == [2]
        assert list(model.context._agents_by_type[Person.TYPE].values()) == persons
        assert model.context.added_via_public_api == []
        place = model.places_proj.lookup_place(200)
        assert model.places_proj.get_place_for_agent(persons[0]) == place
        assert place.has_occupant(persons[0])
    finally:
        Person.registerBehaviorEngine(original_behavior_engine)
        model.conn.close()


def test_create_activities_records_local_activity_map_timings():
    model = _partitioned_model(rank=0, size=2)
    model.params["profiling.phase_timings.enabled"] = True
    model.phase_timings = {}
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.conn.execute("INSERT INTO input.activities VALUES (2, 1, 1, 60, 120, 300)")
        model.create_input_tables()

        schedules = model.create_activities()

        assert list(schedules[0]) == [2]
        assert isinstance(schedules[0][2][0], Act)
        assert schedules[0][2][0].place_id == 200
        assert schedules[0][2][1] is DEFAULT_TRAVEL_LEG
        assert isinstance(schedules[0][2][2], Act)
        assert schedules[0][2][2].place_id == 300
        assert "startup.create_persons.create_activities.load_activity_rows" in model.phase_timings
        assert "startup.create_persons.create_activities.construct_activity_map" in model.phase_timings
    finally:
        model.conn.close()


def test_create_activities_uses_cast_activity_input_columns():
    model = _partitioned_model(rank=0, size=2)
    model.params["activities.table"] = "input.float_activities"
    model.road_network = None
    try:
        model.conn.execute("""
            CREATE TABLE input.float_activities (
                sp_persons_id BIGINT,
                activity_id DOUBLE,
                activity_sequence DOUBLE,
                starttime_min DOUBLE,
                endtime_min DOUBLE,
                sp_act_id DOUBLE
            )
            """)
        model.conn.execute("""
            INSERT INTO input.float_activities VALUES
                (2, 0.0, 0.0, 0.0, 60.0, 200.0),
                (2, 1.9, 1.9, 60.9, 120.9, 300.9)
            """)
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.create_input_tables()

        schedules = model.create_activities()

        assert schedules[0][2][0] == Act(2, 0, 0, 0, 60, 200)
        assert schedules[0][2][1] is DEFAULT_TRAVEL_LEG
        assert schedules[0][2][2] == Act(2, 1, 1, 60, 120, 300)
    finally:
        model.conn.close()


def test_create_activities_orders_rows_by_person_and_sequence():
    model = _partitioned_model(rank=0, size=2)
    model.road_network = None
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.conn.execute("""
            INSERT INTO input.activities VALUES
                (2, 2, 2, 120, 180, 100),
                (2, 1, 1, 60, 120, 300)
            """)
        model.create_input_tables()

        schedules = model.create_activities()

        assert schedules[0][2] == [
            Act(2, 0, 0, 0, 60, 200),
            DEFAULT_TRAVEL_LEG,
            Act(2, 1, 1, 60, 120, 300),
            DEFAULT_TRAVEL_LEG,
            Act(2, 2, 2, 120, 180, 100),
        ]
    finally:
        model.conn.close()


def test_create_activities_with_road_network_keeps_activity_only_schedule():
    model = _partitioned_model(rank=0, size=2)
    model.road_network = object()
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200),
                (1, 2, 0, 300)
            """)
        model.conn.execute("INSERT INTO input.activities VALUES (2, 1, 1, 60, 120, 300)")
        model.create_input_tables()

        schedules = model.create_activities()

        assert schedules[0][2] == [
            Act(2, 0, 0, 0, 60, 200),
            Act(2, 1, 1, 60, 120, 300),
        ]
    finally:
        model.conn.close()


def test_enhanced_places_projection_initializes_parallel_updater_lazily():
    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=True)

    assert projection.parallel_updates_enabled is True
    assert projection._parallel_updater is None


def test_enhanced_places_projection_assigns_new_agent_to_place_directly():
    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    place = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})

    projection.add_place(place)
    projection.assign_new_agent_to_place(person, place)

    assert projection.get_place_for_agent(person) == place
    assert place.has_occupant(person)


def test_enhanced_places_projection_bulk_add_reuses_registered_rank_map():
    projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
    projection.set_place_rank_map({100: 0, 200: 1}, copy=False)
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())

    projection.add_places([home], register_ranks=False)

    assert projection.lookup_place(100) is home
    assert projection.rank_for_place(100) == 0
    assert projection.rank_for_place(200) == 1
    assert projection.get_local_places() == [home]


def test_sync_person_ranks_with_places_returns_context_moves_for_remote_places():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.params = {"partition.table": "partitions.metis_place_partitions"}
    model.place_to_rank = {100: 0, 200: 1}
    model.places_proj = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
    model.places_proj.add_place(home)
    model.places_proj.add(person)
    model.places_proj.assign_agent_to_place(person, home)
    person.place_id = 200
    person.rank_place_id = 200
    model.context = _AgentContext([person])

    moves = model.sync_person_ranks_with_places()

    assert moves == [(person.uid, 1)]
    assert model.places_proj.get_place_for_agent(person) is None


def test_create_input_tables_skips_social_networks_when_disabled():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.params["social_networks.table"] = "input.missing_social_networks"
        model.params["social_networks.enabled"] = False

        model.create_input_tables()

        assert not _duckdb_table_exists(model.conn, "main", "social_networks")
    finally:
        model.conn.close()


def test_create_input_tables_materializes_social_networks_when_enabled():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            CREATE TABLE input.social_networks (
                person_id_a BIGINT,
                person_id_b BIGINT,
                network_kind VARCHAR,
                tie_strength DOUBLE
            )
            """)
        model.conn.execute("INSERT INTO input.social_networks VALUES (1, 2, 'household', 0.9)")
        model.params["social_networks.table"] = "input.social_networks"
        model.params["social_networks.enabled"] = True

        model.create_input_tables()

        assert _duckdb_table_exists(model.conn, "main", "social_networks")
        assert model.conn.execute("SELECT * FROM social_networks").fetchall() == [
            (1, 2, "household", 0.9),
        ]
    finally:
        model.conn.close()


def test_create_input_tables_requires_social_network_table_when_enabled():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.params["social_networks.table"] = "input.missing_social_networks"
        model.params["social_networks.enabled"] = True

        with pytest.raises(MissingRequiredTableError):
            model.create_input_tables()
    finally:
        model.conn.close()


def test_create_social_networks_loads_undirected_potential_ties():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            CREATE TABLE social_networks (
                person_id_a BIGINT,
                person_id_b BIGINT,
                network_kind VARCHAR,
                tie_strength DOUBLE
            )
        """)
        model.conn.execute("INSERT INTO social_networks VALUES (1, 2, 'household', 0.9)")

        assert model.create_social_networks() == {
            1: [(2, "household", 0.9)],
            2: [(1, "household", 0.9)],
        }
    finally:
        model.conn.close()


def test_create_social_networks_rejects_noncanonical_ties():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            CREATE TABLE social_networks (
                person_id_a BIGINT,
                person_id_b BIGINT,
                network_kind VARCHAR
            )
        """)
        model.conn.execute("INSERT INTO social_networks VALUES (2, 1, 'household')")

        with pytest.raises(InvalidSocialNetworkTableError, match="canonical endpoints"):
            model.create_social_networks()
    finally:
        model.conn.close()


class _ScheduledPerson:
    type = Person.TYPE

    def __init__(self, person_id: int, activity: Act, resolved_place_id: int):
        self.id = person_id
        self._activity = activity
        self._resolved_place_id = resolved_place_id
        self.uid = (person_id, Person.TYPE, 0)
        self.rank_place_id = resolved_place_id

    def get_plan_state(self, cal):
        return SimpleNamespace(element=self._activity)

    def _place_id_for_activity(self, activity_id):
        return self._resolved_place_id


def test_model_generates_physical_interactions_from_current_plans():
    model = CasmPop.__new__(CasmPop)
    model.context = _AgentContext(
        [
            _ScheduledPerson(1, Act(1, 0, 0, 480, 600, 10), 10),
            _ScheduledPerson(2, Act(2, 0, 0, 480, 600, 10), 10),
            _ScheduledPerson(3, Act(3, 0, 0, 480, 600, 11), 11),
        ]
    )
    model.cal = SimTime(datetime(2025, 6, 2, 9, 0, 0))
    model.time_step_minutes = 60
    model.social_ties = [SocialTie(1, 2, "work"), SocialTie(1, 3, "work")]

    events = model.generate_in_person_interactions()

    assert [(event.person_id_a, event.person_id_b, event.place_id) for event in events] == [(1, 2, 10)]
    assert (events[0].start_minute, events[0].end_minute) == (540, 600)


class _AllgatherComm:
    def allgather(self, value):
        return [value]


def test_model_generates_opt_in_remote_social_message_for_available_tie():
    model = CasmPop.__new__(CasmPop)
    model.context = _AgentContext(
        [
            _ScheduledPerson(1, Act(1, 0, 0, 480, 600, 10), 10),
            _ScheduledPerson(2, Act(2, 0, 0, 480, 600, 11), 11),
        ]
    )
    model.cal = SimTime(datetime(2025, 6, 2, 9, 0, 0))
    model.time_step_minutes = 60
    model.comm = _AllgatherComm()
    model.params = {"social_networks.remote_messages.interval_minutes": 60}
    model.social_ties = [SocialTie(1, 2, "work", 0.8)]

    intents = model.generate_remote_social_message_intents()

    assert len(intents) == 1
    assert intents[0].sender_uid == (1, Person.TYPE, 0)
    assert intents[0].receiver_uid == (2, Person.TYPE, 0)
    assert intents[0].receiver_place_id == 11
    assert intents[0].payload["kind"] == MessageKind.CHECK_IN
    assert intents[0].payload["metadata"] == {"network_kind": "work"}


def test_agent_logger_skips_when_disabled(tmp_path):
    log_path = tmp_path / "agent_log.parquet"
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "observers.agent_log.enabled": False,
        "observers.output_dir": str(tmp_path),
        "observers.agent_log_file": "agent_log.parquet",
    }

    logger_observer = AgentLogger("AgentLogger", model)
    logger_observer.on_step(model)

    assert not log_path.exists()


def test_social_interaction_logger_writes_only_aggregate_event_counts(tmp_path):
    log_path = tmp_path / "social_interactions.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.social_interaction_log.enabled": True,
        "observers.social_interaction_log_file": "social_interactions.parquet",
    }
    model.cal = SimpleNamespace(tick=12)
    model.interaction_events = [
        InteractionEvent(1, 2, "in_person", "work", 540, 600, place_id=10),
        InteractionEvent(2, 3, "in_person", "work", 540, 600, place_id=10),
    ]
    model.remote_social_message_intents = [
        SimpleNamespace(payload={"metadata": {"network_kind": "household"}})
    ]

    observer = SocialInteractionLogger("SocialInteractionLogger", model)
    observer.on_step(model)

    table = ds.dataset(log_path, format="parquet", partitioning="hive").to_table()
    rows = sorted(table.to_pylist(), key=lambda row: row["channel"])
    assert rows == [
        {
            "random_seed": 42,
            "channel": "in_person",
            "network_kind": "work",
            "event_count": 2,
            "run_id": "seed_42",
            "tick": 12,
            "rank": 0,
        },
        {
            "random_seed": 42,
            "channel": "remote",
            "network_kind": "household",
            "event_count": 1,
            "run_id": "seed_42",
            "tick": 12,
            "rank": 0,
        },
    ]
    assert {"person_id_a", "person_id_b", "place_id", "payload"}.isdisjoint(table.column_names)


def test_schedule_occupancy_logger_writes_only_aggregate_counts(tmp_path):
    log_path = tmp_path / "schedule_occupancy.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.schedule_occupancy_log.enabled": True,
        "observers.schedule_occupancy_log_file": "schedule_occupancy.parquet",
    }
    model.cal = SimpleNamespace(tick=12)
    model.active_planned_presences = lambda: [
        PresenceInterval(1, 10, 540, 600),
        PresenceInterval(2, 10, 540, 600),
        PresenceInterval(3, 11, 540, 600),
        PresenceInterval(4, 12, 540, 600),
        PresenceInterval(5, 12, 540, 600),
        PresenceInterval(6, 12, 540, 600),
    ]
    model.interaction_events = [object(), object()]
    model.remote_social_message_intents = [object()]

    observer = ScheduleOccupancyLogger("ScheduleOccupancyLogger", model)
    observer.on_step(model)

    table = ds.dataset(log_path, format="parquet", partitioning="hive").to_table()
    assert table.to_pylist() == [{
        "random_seed": 42,
        "active_person_count": 6,
        "active_place_count": 3,
        "co_located_person_count": 5,
        "max_place_occupancy": 3,
        "places_with_1_person": 1,
        "places_with_2_to_4_people": 2,
        "places_with_5_to_9_people": 0,
        "places_with_10_or_more_people": 0,
        "in_person_interaction_count": 2,
        "remote_message_count": 1,
        "run_id": "seed_42",
        "tick": 12,
        "rank": 0,
    }]
    assert {"person_id", "person_id_a", "person_id_b", "place_id", "latitude", "longitude"}.isdisjoint(table.column_names)


def test_empty_person_bucket_is_treated_as_no_local_people(tmp_path):
    agent_log_path = tmp_path / "agent_log.parquet"
    behavior_log_path = tmp_path / "behavior_log.parquet"
    model = CasmPop.__new__(CasmPop)
    model.context = _EmptyTypedContext()
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.agent_log_file": "agent_log.parquet",
        "observers.behavior_log_file": "behavior_log.parquet",
    }

    AgentLogger("AgentLogger", model).on_step(model)
    BehaviorLogger("BehaviorLogger", model).on_step(model)
    model.refresh_person_index()
    model.refresh_place_membership()

    assert not agent_log_path.exists()
    assert not behavior_log_path.exists()
    assert model.person_uid_map == {}
    assert model.place_members == {}


def test_run_communication_phases_skips_collectives_when_no_global_intents():
    model = CasmPop.__new__(CasmPop)
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    comm = _AllreduceComm(total=0)
    manager = _TrackingCommunicationManager()
    model.comm = comm
    model.context = _AgentContext([person])
    model.communication_manager = manager
    model.cal = _TickCal()

    model.run_communication_phases()

    assert comm.allreduce_calls == 1
    assert comm.last_value == 0
    assert manager.clear_calls == 1
    assert manager.route_calls == 0
    assert manager.exchange_remote_calls == 0
    assert manager.exchange_ack_calls == 0


def test_run_communication_phases_skips_all_work_when_disabled():
    model = CasmPop.__new__(CasmPop)
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    comm = _AllreduceComm(total=0)
    manager = _TrackingCommunicationManager()
    model.params = {"communication.enabled": False}
    model.comm = comm
    model.context = _AgentContext([person])
    model.communication_manager = manager
    model.cal = _TickCal()

    model.run_communication_phases()

    assert comm.allreduce_calls == 0
    assert manager.clear_calls == 0
    assert manager.route_calls == 0
    assert manager.exchange_remote_calls == 0
    assert manager.exchange_ack_calls == 0


def test_run_communication_phases_participates_when_other_rank_has_intents():
    model = CasmPop.__new__(CasmPop)
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    comm = _AllreduceComm(total=1)
    manager = _TrackingCommunicationManager()
    model.comm = comm
    model.context = _AgentContext([person])
    model.communication_manager = manager
    model.cal = _TickCal()

    model.run_communication_phases()

    assert comm.allreduce_calls == 1
    assert comm.last_value == 0
    assert manager.clear_calls == 1
    assert manager.route_calls == 1
    assert manager.routed_intents == []
    assert manager.exchange_remote_calls == 1
    assert manager.exchange_ack_calls == 1


def test_build_context_skips_startup_communication_indexes_when_disabled():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.comm = MPI.COMM_SELF
    model.context = _ProjectionContext()
    model.params = {
        "communication.enabled": False,
        "social_networks.enabled": False,
        "social_networks.table": "",
        "parallel.places.enabled": False,
        "parallel.places.min_threshold": 50,
        "parallel.places.max_workers": None,
    }
    model._observers = []
    model.queries = {"get_tables": "SHOW TABLES"}
    model.conn = _TablesConn()
    model.get_activity_names = lambda: ["home"]
    model._configure_behavior_engine = lambda: None
    model.create_input_tables = lambda: None
    model.create_places = lambda: None
    model._initialize_place_rank_index = lambda: None
    model.load_road_network = lambda: None
    model.create_persons = lambda rng: None
    model.places_proj = None
    called = []
    model.refresh_person_index = lambda: called.append("person")
    model.refresh_place_membership = lambda: called.append("places")

    def time_phase(phase, func, *args):
        return func(*args)

    model._time_phase = time_phase

    model.build_context()

    assert called == []


def test_build_context_refreshes_startup_communication_indexes_when_enabled():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.comm = MPI.COMM_SELF
    model.context = _ProjectionContext()
    model.params = {
        "communication.enabled": True,
        "social_networks.enabled": False,
        "social_networks.table": "",
        "parallel.places.enabled": False,
        "parallel.places.min_threshold": 50,
        "parallel.places.max_workers": None,
    }
    model._observers = []
    model.queries = {"get_tables": "SHOW TABLES"}
    model.conn = _TablesConn()
    model.get_activity_names = lambda: ["home"]
    model._configure_behavior_engine = lambda: None
    model.create_input_tables = lambda: None
    model.create_places = lambda: None
    model._initialize_place_rank_index = lambda: None
    model.load_road_network = lambda: None
    model.create_persons = lambda rng: None
    model.places_proj = None
    called = []
    model.refresh_person_index = lambda: called.append("person")
    model.refresh_place_membership = lambda: called.append("places")

    def time_phase(phase, func, *args):
        return func(*args)

    model._time_phase = time_phase

    model.build_context()

    assert called == ["person", "places"]


def test_partitioned_environment_step_skips_empty_rank_move_membership_sync():
    model = CasmPop.__new__(CasmPop)
    model.params = {"partition.table": "partitions.metis_place_partitions"}
    model.comm = _AllreduceComm(total=0)
    model.sync_person_ranks_with_places = lambda: []
    membership_syncs = []
    model.sync_place_projection_memberships = lambda: membership_syncs.append(True)
    context = _MigrationContext()
    previous_model = Model.get_model()
    Model.theModel = model
    try:
        SimEnvironment("sim_environment").step(context, _TickCal())
    finally:
        Model.theModel = previous_model

    assert model.comm.allreduce_calls == 1
    assert model.comm.last_value == 0
    assert context.move_agents_calls == 0
    assert context.synchronize_calls == 0
    assert membership_syncs == []


def test_partitioned_environment_step_enters_move_collective_when_any_rank_moves():
    model = CasmPop.__new__(CasmPop)
    model.params = {"partition.table": "partitions.metis_place_partitions"}
    model.comm = _AllreduceComm(total=1)
    model.sync_person_ranks_with_places = lambda: []
    membership_syncs = []
    model.sync_place_projection_memberships = lambda: membership_syncs.append(True)
    context = _MigrationContext()
    previous_model = Model.get_model()
    Model.theModel = model
    try:
        SimEnvironment("sim_environment").step(context, _TickCal())
    finally:
        Model.theModel = previous_model

    assert model.comm.allreduce_calls == 1
    assert model.comm.last_value == 0
    assert context.move_agents_calls == 1
    assert context.moves == []
    assert context.synchronize_calls == 0
    assert membership_syncs == [True]


def test_partitioned_environment_step_uses_movement_collected_rank_moves():
    model = CasmPop.__new__(CasmPop)
    model.params = {"partition.table": "partitions.metis_place_partitions"}
    model.comm = _AllreduceComm(total=1)
    model._rank_moves_from_moved_people_enabled = lambda: True
    model._rank_move_for_person = lambda person: (person.uid, 1)

    def fail_full_scan():
        raise AssertionError("rank moves should be collected during movement")

    model.sync_person_ranks_with_places = fail_full_scan
    membership_syncs = []
    model.sync_place_projection_memberships = lambda: membership_syncs.append(True)
    context = _MigrationContext()
    person = _MovingPerson()
    context._agents = [person]
    previous_model = Model.get_model()
    Model.theModel = model
    try:
        SimEnvironment("sim_environment").step(context, _TickCal())
    finally:
        Model.theModel = previous_model

    assert person.move_calls == 1
    assert model.comm.allreduce_calls == 1
    assert model.comm.last_value == 1
    assert context.move_agents_calls == 1
    assert context.moves == [(person.uid, 1)]
    assert context.synchronize_calls == 0
    assert membership_syncs == [True]


def test_create_input_tables_falls_back_for_missing_partition_assignments():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200)
            """)

        model.create_input_tables()

        assert model.conn.execute("SELECT rank FROM places WHERE sp_id = 300").fetchone() == (0,)
    finally:
        model.conn.close()


def test_create_input_tables_can_require_full_partition_coverage():
    model = _partitioned_model(rank=0, size=2, require_full_coverage=True)
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 1, 100),
                (1, 2, 0, 200)
            """)

        with pytest.raises(MissingPartitionAssignmentError):
            model.create_input_tables()
    finally:
        model.conn.close()


def test_partition_validation_rejects_invalid_rank_assignments():
    model = _partitioned_model(rank=0, size=2)
    try:
        model.conn.execute("""
            INSERT INTO partitions.metis_place_partitions VALUES
                (1, 2, 2, 100),
                (1, 2, 0, 200),
                (1, 2, 1, 300)
            """)

        with pytest.raises(InvalidPartitionRankError):
            model.create_input_tables()
    finally:
        model.conn.close()


def test_partition_validation_runs_locally_on_mpi_root_and_broadcasts_result():
    model = _partitioned_model(rank=0, size=2)
    comm = _BcastComm()
    model.comm = comm
    try:
        model._validate_partition_table("input.places", "partitions.missing_partition")

        raise AssertionError("Expected missing partition table validation to fail")
    except MissingRequiredTableError:
        assert comm.bcast_calls == 1
        assert isinstance(comm.last_value, MissingRequiredTableError)
        assert comm.last_root == 0
    finally:
        model.conn.close()


def test_partition_validation_uses_broadcast_result_on_non_root_rank():
    model = _partitioned_model(rank=1, size=2)
    comm = _BcastComm()
    model.comm = comm
    try:
        model._validate_partition_table("input.missing_places", "partitions.missing_partition")

        assert comm.bcast_calls == 1
        assert comm.last_value is None
        assert comm.last_root == 0
    finally:
        model.conn.close()


def test_configure_behavior_engine_uses_default_engine():
    model = CasmPop.__new__(CasmPop)
    model.params = {"behavior.engine": "default", "behavior.llm.enabled": False}
    original_engine = Person.getBehaviorEngine()
    try:
        model._configure_behavior_engine()
        assert Person.getBehaviorEngine() is BehaviorEngineV2
    finally:
        Person.registerBehaviorEngine(original_engine)


def test_configure_behavior_engine_uses_schedule_engine():
    model = CasmPop.__new__(CasmPop)
    model.params = {"behavior.engine": "schedule", "behavior.llm.enabled": False}
    original_engine = Person.getBehaviorEngine()
    try:
        model._configure_behavior_engine()
        assert Person.getBehaviorEngine() is ScheduleBehaviorEngine
    finally:
        Person.registerBehaviorEngine(original_engine)


def test_step_skips_redundant_person_loop_for_base_schedule_engine():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.params = {"communication.enabled": False, "profiling.phase_timings.enabled": True}
    model.phase_timings = {}
    model.cal = SimTime(datetime(2025, 1, 1))
    model.time_step_minutes = 60
    model._observers = []
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    step_calls = []
    person.step = lambda context, cal: step_calls.append((context, cal))
    model.context = _AgentContext([person])

    class _NoopEnvironment:
        def step(self, context, cal):
            pass

    original_engine = Person.getBehaviorEngine()
    original_environment = CasmPop.get_environment()
    try:
        Person.registerBehaviorEngine(ScheduleBehaviorEngine)
        CasmPop.register_environment(_NoopEnvironment())

        model.step()

        assert step_calls == []
        assert model.phase_timings["tick.person_step"] == 0.0
    finally:
        Person.registerBehaviorEngine(original_engine)
        CasmPop.register_environment(original_environment)


def test_step_runs_person_loop_for_cognitive_behavior_engine():
    model = CasmPop.__new__(CasmPop)
    model.rank = 0
    model.params = {"communication.enabled": False, "profiling.phase_timings.enabled": True}
    model.phase_timings = {}
    model.cal = SimTime(datetime(2025, 1, 1))
    model.time_step_minutes = 60
    model._observers = []
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    step_calls = []
    person.step = lambda context, cal: step_calls.append((context, cal))
    model.context = _AgentContext([person])

    class _NoopEnvironment:
        def step(self, context, cal):
            pass

    original_engine = Person.getBehaviorEngine()
    original_environment = CasmPop.get_environment()
    try:
        Person.registerBehaviorEngine(BehaviorEngineV2)
        CasmPop.register_environment(_NoopEnvironment())

        model.step()

        assert len(step_calls) == 1
        assert model.phase_timings["tick.person_step"] >= 0.0
    finally:
        Person.registerBehaviorEngine(original_engine)
        CasmPop.register_environment(original_environment)


def test_configure_behavior_engine_uses_local_llm_engine():
    model = CasmPop.__new__(CasmPop)
    model.params = {"behavior.engine": "llm_local", "behavior.llm.enabled": False}
    original_engine = Person.getBehaviorEngine()
    try:
        model._configure_behavior_engine()
        assert Person.getBehaviorEngine() is LLMBehaviorEngine
    finally:
        Person.registerBehaviorEngine(original_engine)


def test_configure_behavior_engine_applies_local_llm_parameters():
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "behavior.engine": "llm_local",
        "behavior.llm.enabled": False,
        "behavior.llm.deliberation_interval": 15,
        "behavior.llm.max_memory_events": 7,
        "behavior.llm.signal_cap": 2.25,
        "behavior.llm.memory_decay": 0.5,
        "behavior.activity_semantics.social_ids": [3],
        "behavior.activity_semantics.flexible_ids": [4],
        "behavior.activity_semantics.mandatory_ids": [5],
        "behavior.activity_semantics.travel_sensitive_ids": [6],
    }
    original_engine = Person.getBehaviorEngine()
    original_config = dict(LLMBehaviorEngine._config)
    try:
        model._configure_behavior_engine()
        person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
        person.behavior_engine = LLMBehaviorEngine(person)

        assert person.behavior_engine.deliberation_interval == 15
        assert person.behavior_engine.max_memory_events == 7
        assert person.behavior_engine.adapter.signal_cap == 2.25
        assert person.behavior_engine.adapter.memory_decay == 0.5
        assert person.behavior_engine.adapter.activity_semantics_overrides["social_ids"] == [3]
        assert person.behavior_engine.adapter.activity_semantics_overrides["flexible_ids"] == [4]
        assert person.behavior_engine.adapter.activity_semantics_overrides["mandatory_ids"] == [5]
        assert person.behavior_engine.adapter.activity_semantics_overrides["travel_sensitive_ids"] == [6]
    finally:
        LLMBehaviorEngine._config = original_config
        Person.registerBehaviorEngine(original_engine)


def test_configured_social_override_enables_cancel_social_activity():
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "behavior.engine": "llm_local",
        "behavior.llm.enabled": False,
        "behavior.activity_semantics.social_ids": [1],
        "behavior.activity_semantics.flexible_ids": [],
        "behavior.activity_semantics.mandatory_ids": [],
        "behavior.activity_semantics.travel_sensitive_ids": [],
    }
    original_engine = Person.getBehaviorEngine()
    original_config = dict(LLMBehaviorEngine._config)
    try:
        model._configure_behavior_engine()
        person = Person(
            1,
            0,
            [[Act(1, 0, 0, 0, 60, 100), Leg(mode="travel"), Act(1, 1, 1, 120, 180, 200)]],
            tuple([100, 200]),
            {"sp_id": 1},
        )
        person.behavior_engine = Person.getBehaviorEngine()(person)
        person.recent_messages = [
            CommMessage(
                msg_id="msg-1",
                sender_uid=(2, 0, 0),
                sender_place_id=100,
                receiver_uid=person.uid,
                receiver_place_id=100,
                mode="two_way",
                payload=build_message_payload(
                    MessageKind.COORDINATION,
                    topic="cancel_social",
                    text="Cancel lunch",
                    urgency=0.8,
                    trust_weight=0.8,
                    metadata={"sender_place_id": 100},
                ),
                tick=1,
            )
        ]

        projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
        home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
        work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
        projection.add_place(home)
        projection.add_place(work)
        projection.add(person)
        projection.assign_agent_to_place(person, home)

        person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

        memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
        assert memory_data["plan_adjustment_requested_kind"] == "cancel_social_activity"
        assert memory_data["plan_adjustment_applied"] is True
        assert memory_data["plan_adjustment_kind"] == "cancel_social_activity"
        assert len(person.plans[0]) == 1
        assert isinstance(person.plans[0][0], Act)
        assert person.plans[0][0].activity_id == 0
    finally:
        LLMBehaviorEngine._config = original_config
        Person.registerBehaviorEngine(original_engine)


def test_configured_social_override_absent_leaves_cancel_social_activity_inactive():
    model = CasmPop.__new__(CasmPop)
    model.params = {
        "behavior.engine": "llm_local",
        "behavior.llm.enabled": False,
        "behavior.activity_semantics.social_ids": [],
        "behavior.activity_semantics.flexible_ids": [],
        "behavior.activity_semantics.mandatory_ids": [],
        "behavior.activity_semantics.travel_sensitive_ids": [],
    }
    original_engine = Person.getBehaviorEngine()
    original_config = dict(LLMBehaviorEngine._config)
    try:
        model._configure_behavior_engine()
        original_plan = [Act(1, 0, 0, 0, 60, 100), Leg(mode="travel"), Act(1, 1, 1, 120, 180, 200)]
        person = Person(1, 0, [list(original_plan)], tuple([100, 200]), {"sp_id": 1})
        person.behavior_engine = Person.getBehaviorEngine()(person)
        person.recent_messages = [
            CommMessage(
                msg_id="msg-1",
                sender_uid=(2, 0, 0),
                sender_place_id=100,
                receiver_uid=person.uid,
                receiver_place_id=100,
                mode="two_way",
                payload=build_message_payload(
                    MessageKind.COORDINATION,
                    topic="cancel_social",
                    text="Cancel lunch",
                    urgency=0.8,
                    trust_weight=0.8,
                    metadata={"sender_place_id": 100},
                ),
                tick=1,
            )
        ]

        projection = EnhancedPlacesProjection("places_projection", MPI.COMM_SELF, enable_parallel_updates=False)
        home = Place({"sp_id": 100, "rank": 0}, Place.getPlaceDataClass())
        work = Place({"sp_id": 200, "rank": 0}, Place.getPlaceDataClass())
        projection.add_place(home)
        projection.add_place(work)
        projection.add(person)
        projection.assign_agent_to_place(person, home)

        person.behavior_engine.decide(_StubContext(projection), _sim_time_at(30))

        memory_data = person.behavior_engine.cognition.episodic_memory[-1]["data"]
        assert memory_data["plan_adjustment_requested_kind"] == ""
        assert memory_data["plan_adjustment_applied"] is False
        assert person.plans[0] == original_plan
    finally:
        LLMBehaviorEngine._config = original_config
        Person.registerBehaviorEngine(original_engine)


def test_default_parameters_include_behavior_observer_keys():
    params = CasmPop.get_default_parameters()

    assert "observers.output_dir" in params
    assert "observers.behavior_log.enabled" in params
    assert "observers.behavior_log_file" in params
    assert "observers.delta_agent_state.enabled" in params
    assert "observers.delta_agent_state_file" in params
    assert "observers.delta_agent_state_audit_file" in params
    assert "behavior.activity_semantics.social_ids" in params
    assert "behavior.activity_semantics.flexible_ids" in params


def test_delta_agent_state_logger_writes_first_observed_state(tmp_path):
    log_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.delta_agent_state_file": "agent_state_delta.parquet",
        "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
    }

    class StubCal:
        tick = 12

    model.cal = StubCal()
    person = Person(1, 0, [], tuple([100]), {"sp_id": 1, "x": 4, "y": 5})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.place_id = 100
    person.rank_place_id = 100
    person.behavior_engine.cognition.last_decision = "stay_home"
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 12,
            "event_type": "message_appraisal",
            "data": {
                "plan_adjustment_kind": "preserve_home_activity",
                "safety_signal": 1.2,
            },
        }
    )
    model.context = _AgentContext([person])

    logger_observer = DeltaAgentStateLogger("DeltaAgentStateLogger", model)
    logger_observer.on_step(model)

    table = ds.dataset(log_path, format="parquet", partitioning="hive").to_table()
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["run_id"] == "seed_42"
    assert rows[0]["random_seed"] == 42
    assert rows[0]["tick"] == 12
    assert rows[0]["rank"] == 0
    assert rows[0]["agent_id"] == 1
    assert rows[0]["change_mask"] == "__initial__"
    assert len(rows[0]["state_hash"]) == 64
    assert rows[0]["place_id"] == 100
    assert rows[0]["rank_place_id"] == 100
    assert rows[0]["last_decision"] == "stay_home"
    assert rows[0]["last_memory_event_type"] == "message_appraisal"
    assert rows[0]["last_plan_adjustment_kind"] == "preserve_home_activity"
    assert rows[0]["safety_signal"] == 1.2

    audit_rows = ds.dataset(audit_path, format="parquet", partitioning="hive").to_table().to_pylist()
    assert audit_rows == [
        {
            "run_id": "seed_42",
            "random_seed": 42,
            "agents_changed": 1,
            "agents_evaluated": 1,
            "rank": 0,
            "tick": 12,
        }
    ]


def test_delta_agent_state_logger_audits_unchanged_state_and_writes_changes(tmp_path):
    log_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.delta_agent_state_file": "agent_state_delta.parquet",
        "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
    }

    class StubCal:
        tick = 12

    model.cal = StubCal()
    person = Person(1, 0, [], tuple([100, 200]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.place_id = 100
    person.rank_place_id = 100
    model.context = _AgentContext([person])

    logger_observer = DeltaAgentStateLogger("DeltaAgentStateLogger", model)
    logger_observer.on_step(model)
    model.cal.tick = 13
    logger_observer.on_step(model)
    table = ds.dataset(log_path, format="parquet", partitioning="hive").to_table()
    assert table.num_rows == 1

    person.place_id = 200
    person.rank_place_id = 200
    model.cal.tick = 14
    logger_observer.on_step(model)

    rows = sorted(
        ds.dataset(log_path, format="parquet", partitioning="hive").to_table().to_pylist(),
        key=lambda row: row["tick"],
    )
    assert [row["tick"] for row in rows] == [12, 14]
    assert [row["run_id"] for row in rows] == ["seed_42", "seed_42"]
    assert [row["random_seed"] for row in rows] == [42, 42]
    assert rows[1]["place_id"] == 200
    assert rows[1]["rank_place_id"] == 200
    assert rows[1]["change_mask"] == "place_id,rank_place_id"
    assert rows[1]["state_hash"] != rows[0]["state_hash"]

    audit_rows = sorted(
        ds.dataset(audit_path, format="parquet", partitioning="hive").to_table().to_pylist(),
        key=lambda row: row["tick"],
    )
    assert [row["tick"] for row in audit_rows] == [12, 13, 14]
    assert [row["run_id"] for row in audit_rows] == ["seed_42", "seed_42", "seed_42"]
    assert [row["random_seed"] for row in audit_rows] == [42, 42, 42]
    assert [row["agents_evaluated"] for row in audit_rows] == [1, 1, 1]
    assert [row["agents_changed"] for row in audit_rows] == [1, 0, 1]


def test_delta_agent_state_logger_audits_empty_agent_ticks(tmp_path):
    log_path = tmp_path / "agent_state_delta.parquet"
    audit_path = tmp_path / "agent_state_delta_audit.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.delta_agent_state_file": "agent_state_delta.parquet",
        "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
    }

    class StubCal:
        tick = 12

    model.cal = StubCal()
    model.context = _EmptyTypedContext()

    logger_observer = DeltaAgentStateLogger("DeltaAgentStateLogger", model)
    logger_observer.on_step(model)

    assert not log_path.exists()
    audit_rows = ds.dataset(audit_path, format="parquet", partitioning="hive").to_table().to_pylist()
    assert audit_rows == [
        {
            "run_id": "seed_42",
            "random_seed": 42,
            "agents_changed": 0,
            "agents_evaluated": 0,
            "rank": 0,
            "tick": 12,
        }
    ]


def test_behavior_logger_extracts_behavior_state_snapshot():
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {"observers.behavior_log_file": "behavior_log.parquet"}

    class StubCal:
        tick = 12

    model.cal = StubCal()
    logger_observer = BehaviorLogger("BehaviorLogger", model)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.place_id = 100
    person.rank_place_id = 100
    person.behavior_engine.cognition.last_decision = "stay_home"
    person.behavior_engine.cognition.last_llm_summary = "Recent caution in memory continues to favor staying home."
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 12,
            "event_type": "message_appraisal",
            "summary": "Recent caution in memory continues to favor staying home.",
            "data": {
                "plan_adjustment_requested_kind": "preserve_home_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "preserve_home_activity",
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": 0,
                "plan_adjustment_target_place_id": 100,
                "safety_signal": 1.2,
                "social_signal": 0.0,
                "obligation_signal": 0.0,
                "schedule_signal": 0.0,
                "reply_signal": 0.0,
            },
        }
    )

    row = logger_observer.get_person_log_data(model, person)

    assert row.run_id == "seed_42"
    assert row.random_seed == 42
    assert row.tick == 12
    assert row.rank == 0
    assert row.agent_id == 1
    assert row.place_id == 100
    assert row.last_decision == "stay_home"
    assert row.last_memory_event_type == "message_appraisal"
    assert row.last_plan_adjustment_requested_kind == "preserve_home_activity"
    assert row.last_plan_adjustment_applied is True
    assert row.last_plan_adjustment_skip_reason == ""
    assert row.last_plan_adjustment_kind == "preserve_home_activity"
    assert row.last_plan_adjustment_delay_minutes == 0
    assert row.last_plan_adjustment_target_activity_id == 0
    assert row.last_plan_adjustment_target_place_id == 100
    assert row.safety_signal == 1.2


def test_behavior_logger_writes_parquet_dataset(tmp_path):
    log_path = tmp_path / "behavior_log.parquet"
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {
        "observers.output_dir": str(tmp_path),
        "observers.behavior_log_file": "behavior_log.parquet",
    }

    class StubCal:
        tick = 12

    class StubContext:
        def __init__(self, people):
            self._people = people

        def agents(self, agent_type=0):
            return list(self._people)

    model.cal = StubCal()
    person_home = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person_home.behavior_engine = LLMBehaviorEngine(person_home)
    person_home.place_id = 100
    person_home.rank_place_id = 100
    person_home.behavior_engine.cognition.last_decision = "stay_home"
    person_home.behavior_engine.cognition.last_llm_summary = "Recent caution in memory continues to favor staying home."
    person_home.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 12,
            "event_type": "message_appraisal",
            "summary": "Recent caution in memory continues to favor staying home.",
            "data": {
                "plan_adjustment_requested_kind": "preserve_home_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "preserve_home_activity",
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": 0,
                "plan_adjustment_target_place_id": 100,
                "safety_signal": 1.2,
                "social_signal": 0.0,
                "obligation_signal": 0.0,
                "schedule_signal": 0.0,
                "reply_signal": 0.0,
            },
        }
    )
    person_defer = Person(2, 0, [], tuple([100]), {"sp_id": 2})
    person_defer.behavior_engine = LLMBehaviorEngine(person_defer)
    person_defer.place_id = 100
    person_defer.rank_place_id = 100
    person_defer.behavior_engine.cognition.last_decision = "follow_schedule"
    person_defer.behavior_engine.cognition.last_llm_summary = (
        "A coordination request benefits from a prompt acknowledgment."
    )
    person_defer.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 12,
            "event_type": "message_appraisal",
            "summary": "A coordination request benefits from a prompt acknowledgment.",
            "data": {
                "plan_adjustment_requested_kind": "defer_next_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "defer_next_activity",
                "plan_adjustment_delay_minutes": 30,
                "plan_adjustment_target_activity_id": 1,
                "plan_adjustment_target_place_id": 100,
                "safety_signal": 0.0,
                "social_signal": 0.0,
                "obligation_signal": 0.0,
                "schedule_signal": 0.0,
                "reply_signal": 1.0,
            },
        }
    )
    model.context = StubContext([person_home, person_defer])

    logger_observer = BehaviorLogger("BehaviorLogger", model)
    logger_observer.on_step(model)

    parquet_files = sorted(log_path.rglob("*.parquet"))
    assert parquet_files

    table = pq.read_table(parquet_files[0])
    rows = table.to_pylist()
    assert len(rows) == 2
    rows_by_agent = {row["agent_id"]: row for row in rows}
    assert {row["random_seed"] for row in rows} == {42}
    dataset_rows = ds.dataset(log_path, format="parquet", partitioning="hive").to_table().to_pylist()
    assert {row["run_id"] for row in dataset_rows} == {"seed_42"}
    assert rows_by_agent[1]["last_decision"] == "stay_home"
    assert rows_by_agent[1]["last_plan_adjustment_requested_kind"] == "preserve_home_activity"
    assert rows_by_agent[1]["last_plan_adjustment_applied"] is True
    assert rows_by_agent[1]["last_plan_adjustment_skip_reason"] == ""
    assert rows_by_agent[1]["last_plan_adjustment_kind"] == "preserve_home_activity"
    assert rows_by_agent[1]["last_plan_adjustment_delay_minutes"] == 0
    assert rows_by_agent[1]["last_plan_adjustment_target_activity_id"] == 0
    assert rows_by_agent[1]["last_plan_adjustment_target_place_id"] == 100
    assert rows_by_agent[1]["safety_signal"] == 1.2
    assert rows_by_agent[2]["last_decision"] == "follow_schedule"
    assert rows_by_agent[2]["last_plan_adjustment_requested_kind"] == "defer_next_activity"
    assert rows_by_agent[2]["last_plan_adjustment_applied"] is True
    assert rows_by_agent[2]["last_plan_adjustment_skip_reason"] == ""
    assert rows_by_agent[2]["last_plan_adjustment_kind"] == "defer_next_activity"
    assert rows_by_agent[2]["last_plan_adjustment_delay_minutes"] == 30
    assert rows_by_agent[2]["last_plan_adjustment_target_activity_id"] == 1
    assert rows_by_agent[2]["last_plan_adjustment_target_place_id"] == 100
    assert rows_by_agent[2]["reply_signal"] == 1.0


def test_behavior_logger_extracts_defer_next_activity_kind():
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {"observers.behavior_log_file": "behavior_log.parquet"}

    class StubCal:
        tick = 18

    model.cal = StubCal()
    logger_observer = BehaviorLogger("BehaviorLogger", model)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.behavior_engine.cognition.last_decision = "follow_schedule"
    person.behavior_engine.cognition.last_llm_summary = "A coordination request benefits from a prompt acknowledgment."
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 18,
            "event_type": "message_appraisal",
            "summary": "A coordination request benefits from a prompt acknowledgment.",
            "data": {
                "plan_adjustment_requested_kind": "defer_next_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "defer_next_activity",
                "plan_adjustment_delay_minutes": 30,
                "plan_adjustment_target_activity_id": 1,
                "plan_adjustment_target_place_id": 100,
                "reply_signal": 1.0,
            },
        }
    )

    row = logger_observer.get_person_log_data(model, person)

    assert row.last_plan_adjustment_requested_kind == "defer_next_activity"
    assert row.last_plan_adjustment_applied is True
    assert row.last_plan_adjustment_skip_reason == ""
    assert row.last_plan_adjustment_kind == "defer_next_activity"
    assert row.last_plan_adjustment_delay_minutes == 30
    assert row.last_plan_adjustment_target_activity_id == 1
    assert row.last_plan_adjustment_target_place_id == 100


def test_behavior_logger_extracts_skipped_adjustment_request():
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {"observers.behavior_log_file": "behavior_log.parquet"}

    class StubCal:
        tick = 20

    model.cal = StubCal()
    logger_observer = BehaviorLogger("BehaviorLogger", model)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.behavior_engine.cognition.last_decision = "follow_schedule"
    person.behavior_engine.cognition.last_llm_summary = "A coordination request could not be applied."
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 20,
            "event_type": "message_appraisal",
            "summary": "A coordination request could not be applied.",
            "data": {
                "plan_adjustment_requested_kind": "defer_next_activity",
                "plan_adjustment_applied": False,
                "plan_adjustment_skip_reason": "no_slack",
                "plan_adjustment_kind": "",
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": 1,
                "plan_adjustment_target_place_id": 200,
            },
        }
    )

    row = logger_observer.get_person_log_data(model, person)

    assert row.last_plan_adjustment_requested_kind == "defer_next_activity"
    assert row.last_plan_adjustment_applied is False
    assert row.last_plan_adjustment_skip_reason == "no_slack"
    assert row.last_plan_adjustment_kind == ""
    assert row.last_plan_adjustment_delay_minutes == 0
    assert row.last_plan_adjustment_target_activity_id == 1
    assert row.last_plan_adjustment_target_place_id == 200


def test_behavior_logger_extracts_skip_flexible_activity_kind():
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {"observers.behavior_log_file": "behavior_log.parquet"}

    class StubCal:
        tick = 22

    model.cal = StubCal()
    logger_observer = BehaviorLogger("BehaviorLogger", model)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.behavior_engine.cognition.last_decision = "follow_schedule"
    person.behavior_engine.cognition.last_llm_summary = (
        "A cancellation request favors skipping the next flexible activity."
    )
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 22,
            "event_type": "message_appraisal",
            "summary": "A cancellation request favors skipping the next flexible activity.",
            "data": {
                "plan_adjustment_requested_kind": "skip_flexible_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "skip_flexible_activity",
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": 1,
                "plan_adjustment_target_place_id": 200,
            },
        }
    )

    row = logger_observer.get_person_log_data(model, person)

    assert row.last_plan_adjustment_requested_kind == "skip_flexible_activity"
    assert row.last_plan_adjustment_applied is True
    assert row.last_plan_adjustment_skip_reason == ""
    assert row.last_plan_adjustment_kind == "skip_flexible_activity"
    assert row.last_plan_adjustment_delay_minutes == 0
    assert row.last_plan_adjustment_target_activity_id == 1
    assert row.last_plan_adjustment_target_place_id == 200


def test_behavior_logger_extracts_cancel_social_activity_kind():
    model = CasmPop.__new__(CasmPop)
    model.comm = MPI.COMM_SELF
    model.params = {"observers.behavior_log_file": "behavior_log.parquet"}

    class StubCal:
        tick = 24

    model.cal = StubCal()
    logger_observer = BehaviorLogger("BehaviorLogger", model)

    person = Person(1, 0, [], tuple([100]), {"sp_id": 1})
    person.behavior_engine = LLMBehaviorEngine(person)
    person.behavior_engine.cognition.last_decision = "follow_schedule"
    person.behavior_engine.cognition.last_llm_summary = (
        "A social cancellation message favors canceling the next social activity."
    )
    person.behavior_engine.cognition.episodic_memory.append(
        {
            "tick": 24,
            "event_type": "message_appraisal",
            "summary": "A social cancellation message favors canceling the next social activity.",
            "data": {
                "plan_adjustment_requested_kind": "cancel_social_activity",
                "plan_adjustment_applied": True,
                "plan_adjustment_skip_reason": "",
                "plan_adjustment_kind": "cancel_social_activity",
                "plan_adjustment_delay_minutes": 0,
                "plan_adjustment_target_activity_id": 1,
                "plan_adjustment_target_place_id": 200,
            },
        }
    )

    row = logger_observer.get_person_log_data(model, person)

    assert row.last_plan_adjustment_requested_kind == "cancel_social_activity"
    assert row.last_plan_adjustment_applied is True
    assert row.last_plan_adjustment_skip_reason == ""
    assert row.last_plan_adjustment_kind == "cancel_social_activity"
    assert row.last_plan_adjustment_delay_minutes == 0
    assert row.last_plan_adjustment_target_activity_id == 1
    assert row.last_plan_adjustment_target_place_id == 200
