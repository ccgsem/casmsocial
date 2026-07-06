"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the CasmPop
"""

import hashlib
import json
import os
import pathlib
import re
import time
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import repast4py
import repast4py.random
from dotenv import load_dotenv
from loguru import logger
from mpi4py import MPI
from numpy.random import Generator
from pyarrow.dataset import HivePartitioning
from repast4py import context as ctx, schedule

from casmsocial.activities import (
    DEFAULT_TRAVEL_LEG,
    Act,
    Leg,
    Plan,
    make_plan,
    make_routed_plan,
    validate_leg_against_schedule,
)
from casmsocial.communication import CommunicationManager, MessageIntent
from casmsocial.data_utilities import (
    check_if_table_exists,
    convert_to_int,
    quote_table_identifier,
)
from casmsocial.date_utilities import get_closest_monday, get_midnight
from casmsocial.ducklake_utils import get_ducklake_connection
from casmsocial.environment import Environment
from casmsocial.factory import Models
from casmsocial.household import Household
from casmsocial.model import Model
from casmsocial.observer import Observer
from casmsocial.person import BehaviorEngineV2, LLMBehaviorEngine, Person, PersonData, ScheduleBehaviorEngine
from casmsocial.place import Place, PlaceData, PlacesProjectionV2
from casmsocial.road_network import RoadNetwork
from casmsocial.sim_time import SimTime


# Custom exceptions for CasmPop
class MissingRequiredParameterError(Exception):
    def __init__(self, keys):
        keys_str = ", ".join(str(k) for k in keys) if isinstance(keys, list | tuple) else str(keys)
        super().__init__(f"Missing required parameter(s): {keys_str}")


class MissingRequiredTableError(Exception):
    def __init__(self, keys):
        keys_str = ", ".join(str(k) for k in keys) if isinstance(keys, list | tuple) else str(keys)
        super().__init__(f"Missing required table(s): {keys_str}")


class InvalidTimeStepError(Exception):
    def __init__(self, value):
        super().__init__(f"Invalid time step value: {value}. Time step must be " "an integer.")


class InvalidTableNameError(Exception):
    def __init__(self, table_name):
        super().__init__(f"Invalid table name: {table_name}")


class MissingDataPathError(Exception):
    def __init__(self, data_path):
        super().__init__(f"Missing or invalid data path: {data_path}")


class MissingPartitionAssignmentError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InvalidPartitionRankError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InvalidObserverOutputFileError(ValueError):
    def __init__(self, key: str, value: Any):
        super().__init__(
            f"Invalid observer output file for {key}: {value!r}. "
            "Set observers.output_dir to the directory and use a filename only."
        )


class InvalidAgentLogColumnsError(ValueError):
    def __init__(self, value: Any):
        super().__init__(f"Invalid observers.agent_log_columns: {value!r}. Expected a list of column names.")


class InvalidAgentLogColumnError(ValueError):
    def __init__(self, column: str, reason: str):
        super().__init__(f"Invalid observers.agent_log_columns entry {column!r}: {reason}")


# Note: the CasmPop class is designed to be subclassed for specific models. The
# CasmPop class provides the core functionality for initializing the
# simulation, creating agents and places, and managing the simulation time and
# schedule. Subclasses can override the build_context method to create the
# specific agents, places, and activities for their model, and can also add
# observers to log data or perform other actions at each time step or at the
# end of the simulation.
# For an example of how to subclass CasmPop, see the CitySocialModel class in
# casmsocial/citysim/citysocialmodel.py.
class SimEnvironment(Environment):
    """Sim (social interaction model) environment class.

    Creates a basic physical and social environment for the simulation.
    """

    def __init__(self, name: str):
        """Constructor for the SimEnvironment class.

        Args:
            name: The name of the environment.
        """
        super().__init__(name)

    def setup(self) -> None:
        """Set up the environment."""
        pass

    def teardown(self) -> None:
        """Tear down the environment."""
        pass

    def movePersons(
        self,
        context: ctx.SharedContext,
        cal: SimTime,
        model=None,
    ) -> list[tuple[tuple[int, int, int], int]] | None:
        """Move all persons"""
        # to_move = []
        # next_place = Place()
        countOfBadMoves = 0

        places_proj = context.get_projection("places_projection")
        minute_of_day = getattr(cal, "minute_of_day", None)
        is_weekday = getattr(cal, "is_weekday", None)
        can_move_at = minute_of_day is not None and is_weekday is not None
        moves_to_apply = None
        rank_move_for_person = None
        if model is not None:
            rank_moves_from_moved_people_enabled = getattr(
                model,
                "_rank_moves_from_moved_people_enabled",
                None,
            )
            rank_move_for_person = getattr(model, "_rank_move_for_person", None)
            if (
                rank_moves_from_moved_people_enabled is not None
                and rank_move_for_person is not None
                and rank_moves_from_moved_people_enabled()
            ):
                moves_to_apply = []

        for person in _person_agents(context):
            move_at = getattr(person, "move_at", None)
            result = (
                move_at(minute_of_day, is_weekday, places_proj)
                if can_move_at and move_at is not None
                else person.move(cal, places_proj)
            )
            if not result:
                countOfBadMoves += 1
                continue
            if rank_move_for_person is not None and moves_to_apply is not None:
                rank_move = rank_move_for_person(person)
                if rank_move is not None:
                    moves_to_apply.append(rank_move)

        logger.debug("number of bad moves = {}", countOfBadMoves)
        return moves_to_apply

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        # theModel = Model.get_model()
        # tick = self.runner.schedule.tick
        model = Model.get_model()

        # move persons
        moves_to_apply = self._time_model_phase(
            model,
            "tick.environment.move_persons",
            self.movePersons,
            context,
            cal,
            model,
        )
        if model is not None and hasattr(model, "sync_person_ranks_with_places"):
            if moves_to_apply is None:
                moves_to_apply = self._time_model_phase(
                    model,
                    "tick.environment.rank_moves",
                    model.sync_person_ranks_with_places,
                )
            else:
                self._time_model_phase(model, "tick.environment.rank_moves", lambda: None)
        if moves_to_apply is None:
            moves_to_apply = []
        sync_place_memberships = True
        if model is not None and getattr(model, "_partition_table_name", lambda: None)() is not None:
            global_move_count = (
                self._time_model_phase(
                    model,
                    "tick.environment.rank_move_allreduce",
                    model._global_rank_move_count,
                    len(moves_to_apply),
                )
                if hasattr(model, "_global_rank_move_count")
                else len(moves_to_apply)
            )
            if global_move_count:
                self._time_model_phase(
                    model,
                    "tick.environment.move_agents",
                    context.move_agents,
                    moves_to_apply,
                    Person.restore,
                )
            else:
                sync_place_memberships = False
        else:
            self._time_model_phase(
                model,
                "tick.environment.context_synchronize",
                context.synchronize,
                Person.restore,
            )
        if sync_place_memberships and model is not None and hasattr(model, "sync_place_projection_memberships"):
            self._time_model_phase(
                model,
                "tick.environment.place_memberships",
                model.sync_place_projection_memberships,
            )

        # theModel.make_contacts(tick)

    def at_end(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Actions to perform at the end of the simulation."""
        pass

    def _time_model_phase(self, model, phase: str, func, *args, **kwargs):
        if model is not None and hasattr(model, "_time_phase"):
            return model._time_phase(phase, func, *args, **kwargs)
        return func(*args, **kwargs)


@dataclass
class PersonBehaviorLogData:
    run_id: str
    random_seed: int
    tick: int
    rank: int
    agent_id: int
    place_id: int
    rank_place_id: int
    last_decision: str
    last_llm_summary: str
    last_memory_event_type: str
    last_plan_adjustment_requested_kind: str
    last_plan_adjustment_applied: bool
    last_plan_adjustment_skip_reason: str
    last_plan_adjustment_kind: str
    last_plan_adjustment_delay_minutes: int
    last_plan_adjustment_target_activity_id: int
    last_plan_adjustment_target_place_id: int
    safety_signal: float
    social_signal: float
    obligation_signal: float
    schedule_signal: float
    reply_signal: float


@dataclass
class AgentStateDeltaLogData:
    run_id: str
    random_seed: int
    tick: int
    rank: int
    agent_id: int
    state_hash: str
    change_mask: str
    x: float
    y: float
    place_id: int
    rank_place_id: int
    last_decision: str
    last_memory_event_type: str
    last_plan_adjustment_kind: str
    safety_signal: float
    social_signal: float
    obligation_signal: float
    schedule_signal: float
    reply_signal: float


@dataclass
class AgentStateDeltaAuditLogData:
    run_id: str
    random_seed: int
    tick: int
    rank: int
    agents_evaluated: int
    agents_changed: int


def _person_agents(context):
    """Return local person agents, tolerating ranks with no local people."""
    try:
        return context.agents(agent_type=Person.TYPE)
    except KeyError:
        return iter(())


def _behavior_memory_state(person: Person) -> tuple[Any, dict[str, Any], Any]:
    behavior_engine = getattr(person, "behavior_engine", None)
    cognition = getattr(behavior_engine, "cognition", None)
    last_event = cognition.episodic_memory[-1] if cognition and cognition.episodic_memory else {}
    last_data = dict(last_event.get("data", {}))
    return last_event, last_data, cognition


def _safe_run_id(raw_value: Any) -> str:
    run_id = str(raw_value).strip()
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("_.-")
    return run_id or "run"


def _model_random_seed(model: Model) -> int:
    value = getattr(model, "params", {}).get("random.seed", 42)
    return int(value) if value is not None else 42


def _model_run_id(model: Model) -> str:
    params = getattr(model, "params", {})
    configured_run_id = params.get("simulation.run_id")
    if configured_run_id is not None and str(configured_run_id).strip():
        return _safe_run_id(configured_run_id)
    return _safe_run_id(f"seed_{_model_random_seed(model)}")


def _output_partition_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("tick", pa.int32()),
            pa.field("rank", pa.int32()),
        ]
    )


OBSERVER_OUTPUT_DIR_PARAM = "observers.output_dir"
OBSERVER_OUTPUT_DIR_DEFAULT = "data/output"
OBSERVER_OUTPUT_FILE_DEFAULTS = {
    "observers.agent_log_file": "agent_log.parquet",
    "observers.behavior_log_file": "behavior_log.parquet",
    "observers.delta_agent_state_file": "agent_state_delta.parquet",
    "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
}


def _observer_output_filename(key: str, value: Any, default_filename: str) -> str:
    if value is None:
        return default_filename

    filename = str(value).strip()
    if not filename:
        return default_filename

    posix_path = pathlib.PurePosixPath(filename)
    windows_path = pathlib.PureWindowsPath(filename)
    if (
        filename in {".", ".."}
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or posix_path.name != filename
        or windows_path.name != filename
    ):
        raise InvalidObserverOutputFileError(key, value)
    return filename


def _observer_output_dir(model: Model | None) -> str:
    params = getattr(model, "params", {}) if model is not None else {}
    value = params.get(OBSERVER_OUTPUT_DIR_PARAM, "output")
    if value is None or not str(value).strip():
        value = "output"
    return str(value)


def _join_output_path(base: str, filename: str) -> str:
    """Join a filename onto an output base that may be a local path or a
    URI (e.g. ``s3://bucket/prefix``, per casmdb's output-directory
    contract). ``pathlib.Path`` collapses the "//" in a URI's scheme
    separator -- ``s3://bucket/x`` silently becomes ``s3:/bucket/x``,
    which pyarrow's filesystem resolver then rejects as a malformed URI.
    """
    if "://" in base:
        return f"{base.rstrip('/')}/{filename.lstrip('/')}"
    return str(pathlib.Path(base) / filename)


def _observer_output_path(model: Model | None, key: str) -> str:
    default_filename = OBSERVER_OUTPUT_FILE_DEFAULTS[key]
    params = getattr(model, "params", {}) if model is not None else {}
    filename = _observer_output_filename(
        key,
        params.get(key),
        default_filename,
    )
    return _join_output_path(_observer_output_dir(model), filename)


_AGENT_LOG_SCALAR_TYPES = (int, float, str, bool)


# Define a basic Observer class for logging agent data
class AgentLogger(Observer):
    """Observer class for logging agent data.

    ``observers.agent_log_columns`` selects which per-person columns are
    logged alongside the fixed run/tick/rank/agent_id identity columns.
    ``"x"`` and ``"y"`` are read from the person's location; every other
    name is read from ``person.state`` (the model's registered
    ``PersonData`` class), so a model that registers a derived ``PersonData``
    subclass gets its extra fields logged just by naming them here -- no new
    Observer subclass required. Defaults to ``("x", "y", "place_id")``,
    matching the historical fixed schema.
    """

    DEFAULT_STATE_COLUMNS: ClassVar[tuple[str, ...]] = ("x", "y", "place_id")

    def __init__(self, name, model: Model = None):
        super().__init__(name, model)
        self.log_file = _observer_output_path(model, "observers.agent_log_file")
        self._last_table: pa.Table | None = None
        params = getattr(model, "params", {}) if model is not None else {}
        requested_columns = params.get("observers.agent_log_columns")
        self.state_columns: tuple[str, ...] = (
            tuple(requested_columns) if requested_columns else self.DEFAULT_STATE_COLUMNS
        )

    def _person_state_value(self, person: Person, column: str) -> Any:
        """Resolve one configured agent-log column for a person."""
        if column == "x":
            return person.pt.x
        if column == "y":
            return person.pt.y
        try:
            value = getattr(person.state, column)
        except AttributeError as err:
            raise InvalidAgentLogColumnError(column, f"not an attribute of {type(person.state).__name__}") from err
        if value is not None and not isinstance(value, _AGENT_LOG_SCALAR_TYPES):
            raise InvalidAgentLogColumnError(
                column, f"value of type {type(value).__name__} is not Arrow-scalar-compatible"
            )
        return value

    def get_person_log_data(self, model: Model, person: Person) -> dict[str, Any]:
        """Get the agent data for logging."""
        row: dict[str, Any] = {
            "run_id": _model_run_id(model),
            "random_seed": _model_random_seed(model),
            "tick": model.cal.tick,
            "rank": model.comm.Get_rank(),
            "agent_id": person.id,
        }
        for column in self.state_columns:
            row[column] = self._person_state_value(person, column)
        return row

    def on_step(self, model: Model) -> None:
        if hasattr(model, "_agent_log_enabled") and not model._agent_log_enabled():
            return

        people = list(_person_agents(model.context))
        if not people:
            return

        # create a DataFrame for the agent logs
        logger.info("Logging agents' data...")

        agent_log_df = pl.DataFrame([self.get_person_log_data(model, person) for person in people])

        # convert the DataFrame to an Arrow Table
        agent_log_table = agent_log_df.to_arrow()
        self._last_table = agent_log_table

        # Set Hive-style partitioning
        partitioning = HivePartitioning(_output_partition_schema())

        # Write dataset
        ds.write_dataset(
            data=agent_log_table,
            base_dir=self.log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

    def get_output_tables(self, model: Model) -> dict[str, pa.Table]:
        """Return the most recently logged agent table, keyed by channel name."""
        if self._last_table is None:
            return {}
        return {"agent_log": self._last_table}


class BehaviorLogger(Observer):
    """Observer class for logging behavior-engine state."""

    def __init__(self, name, model: Model = None):
        super().__init__(name, model)
        self.log_file = _observer_output_path(model, "observers.behavior_log_file")
        self._last_table: pa.Table | None = None

    def get_person_log_data(self, model: Model, person: Person) -> PersonBehaviorLogData:
        """Get a compact snapshot of the person's behavior state."""
        last_event, last_data, cognition = _behavior_memory_state(person)
        return PersonBehaviorLogData(
            run_id=_model_run_id(model),
            random_seed=_model_random_seed(model),
            tick=model.cal.tick,
            rank=model.comm.Get_rank(),
            agent_id=person.id,
            place_id=person.place_id,
            rank_place_id=person.rank_place_id,
            last_decision=getattr(cognition, "last_decision", ""),
            last_llm_summary=getattr(cognition, "last_llm_summary", ""),
            last_memory_event_type=str(last_event.get("event_type", "")),
            last_plan_adjustment_requested_kind=str(last_data.get("plan_adjustment_requested_kind", "")),
            last_plan_adjustment_applied=bool(last_data.get("plan_adjustment_applied", False)),
            last_plan_adjustment_skip_reason=str(last_data.get("plan_adjustment_skip_reason", "")),
            last_plan_adjustment_kind=str(last_data.get("plan_adjustment_kind", "")),
            last_plan_adjustment_delay_minutes=int(last_data.get("plan_adjustment_delay_minutes", 0)),
            last_plan_adjustment_target_activity_id=int(last_data.get("plan_adjustment_target_activity_id", -1)),
            last_plan_adjustment_target_place_id=int(last_data.get("plan_adjustment_target_place_id", 0)),
            safety_signal=float(last_data.get("safety_signal", 0.0)),
            social_signal=float(last_data.get("social_signal", 0.0)),
            obligation_signal=float(last_data.get("obligation_signal", 0.0)),
            schedule_signal=float(last_data.get("schedule_signal", 0.0)),
            reply_signal=float(last_data.get("reply_signal", 0.0)),
        )

    def on_step(self, model: Model) -> None:
        """Write per-agent behavior state for the current tick."""
        people = list(_person_agents(model.context))
        if not people:
            return

        logger.info("Logging agents' behavior state...")
        behavior_log_df = pl.DataFrame([self.get_person_log_data(model, person) for person in people])
        behavior_log_table = behavior_log_df.to_arrow()
        self._last_table = behavior_log_table
        partitioning = HivePartitioning(_output_partition_schema())
        ds.write_dataset(
            data=behavior_log_table,
            base_dir=self.log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

    def get_output_tables(self, model: Model) -> dict[str, pa.Table]:
        """Return the most recently logged behavior table, keyed by channel name."""
        if self._last_table is None:
            return {}
        return {"behavior_log": self._last_table}


class DeltaAgentStateLogger(Observer):
    """Observer that writes full agent state rows only when state changes."""

    def __init__(self, name, model: Model = None):
        super().__init__(name, model)
        self.log_file = _observer_output_path(model, "observers.delta_agent_state_file")
        self.audit_file = _observer_output_path(model, "observers.delta_agent_state_audit_file")
        self._previous_state_by_agent_id: dict[int, dict[str, int | float | str]] = {}
        self._previous_hash_by_agent_id: dict[int, str] = {}
        self._last_delta_table: pa.Table | None = None
        self._last_audit_table: pa.Table | None = None

    def _normalized_person_state(self, person: Person) -> dict[str, int | float | str]:
        last_event, last_data, cognition = _behavior_memory_state(person)
        return {
            "x": round(float(person.pt.x), 6),
            "y": round(float(person.pt.y), 6),
            "place_id": int(person.place_id),
            "rank_place_id": int(person.rank_place_id),
            "last_decision": str(getattr(cognition, "last_decision", "")),
            "last_memory_event_type": str(last_event.get("event_type", "")),
            "last_plan_adjustment_kind": str(last_data.get("plan_adjustment_kind", "")),
            "safety_signal": round(float(last_data.get("safety_signal", 0.0)), 6),
            "social_signal": round(float(last_data.get("social_signal", 0.0)), 6),
            "obligation_signal": round(float(last_data.get("obligation_signal", 0.0)), 6),
            "schedule_signal": round(float(last_data.get("schedule_signal", 0.0)), 6),
            "reply_signal": round(float(last_data.get("reply_signal", 0.0)), 6),
        }

    def _state_hash(self, state: dict[str, int | float | str]) -> str:
        state_json = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(state_json.encode("utf-8")).hexdigest()

    def _change_mask(self, agent_id: int, state: dict[str, int | float | str]) -> str:
        previous_state = self._previous_state_by_agent_id.get(agent_id)
        if previous_state is None:
            return "__initial__"
        return ",".join(key for key, value in state.items() if previous_state.get(key) != value)

    def _delta_row(self, model: Model, person: Person) -> AgentStateDeltaLogData | None:
        state = self._normalized_person_state(person)
        state_hash = self._state_hash(state)
        agent_id = int(person.id)
        if self._previous_hash_by_agent_id.get(agent_id) == state_hash:
            return None

        change_mask = self._change_mask(agent_id, state)
        self._previous_state_by_agent_id[agent_id] = state
        self._previous_hash_by_agent_id[agent_id] = state_hash
        return AgentStateDeltaLogData(
            run_id=_model_run_id(model),
            random_seed=_model_random_seed(model),
            tick=model.cal.tick,
            rank=model.comm.Get_rank(),
            agent_id=agent_id,
            state_hash=state_hash,
            change_mask=change_mask,
            x=float(state["x"]),
            y=float(state["y"]),
            place_id=int(state["place_id"]),
            rank_place_id=int(state["rank_place_id"]),
            last_decision=str(state["last_decision"]),
            last_memory_event_type=str(state["last_memory_event_type"]),
            last_plan_adjustment_kind=str(state["last_plan_adjustment_kind"]),
            safety_signal=float(state["safety_signal"]),
            social_signal=float(state["social_signal"]),
            obligation_signal=float(state["obligation_signal"]),
            schedule_signal=float(state["schedule_signal"]),
            reply_signal=float(state["reply_signal"]),
        )

    def _write_partitioned_rows(self, rows: list[Any], base_dir: str) -> pa.Table:
        table = pl.DataFrame(rows).to_arrow()
        partitioning = HivePartitioning(_output_partition_schema())
        ds.write_dataset(
            data=table,
            base_dir=base_dir,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )
        return table

    def _write_audit_row(self, model: Model, *, agents_evaluated: int, agents_changed: int) -> pa.Table:
        return self._write_partitioned_rows(
            [
                AgentStateDeltaAuditLogData(
                    run_id=_model_run_id(model),
                    random_seed=_model_random_seed(model),
                    tick=model.cal.tick,
                    rank=model.comm.Get_rank(),
                    agents_evaluated=agents_evaluated,
                    agents_changed=agents_changed,
                )
            ],
            self.audit_file,
        )

    def on_step(self, model: Model) -> None:
        """Write changed full-state rows for local people at the current tick."""
        people = list(_person_agents(model.context))
        delta_rows = [row for person in people if (row := self._delta_row(model, person)) is not None]

        if delta_rows:
            logger.info("Logging changed agents' state...")
            self._last_delta_table = self._write_partitioned_rows(delta_rows, self.log_file)
        self._last_audit_table = self._write_audit_row(
            model, agents_evaluated=len(people), agents_changed=len(delta_rows)
        )

    def get_output_tables(self, model: Model) -> dict[str, pa.Table]:
        """Return the most recent delta/audit tables, keyed by channel name."""
        tables: dict[str, pa.Table] = {}
        if self._last_delta_table is not None:
            tables["agent_state_delta"] = self._last_delta_table
        if self._last_audit_table is not None:
            tables["agent_state_delta_audit"] = self._last_audit_table
        return tables


class CasmPop(Model):
    """
    The CasmPop class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    The CasmPop class is a subclass of the Model class, which is an abstract
    base class that defines the interface for all models in the casmsocial.
    The CasmPop class implements the start and step methods, which are called
    by the run function in the casmsocial module to start and run the model.

    The CasmPop class adds the following functionality to the Model class:

    - The CasmPop class initializes geographic places and agents.
    - The CasmPop class updates the  environment for the current time step.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    # class variables
    __personClass: type[Person] = Person
    __placeClass: type[Place] = Place
    __householdClass: type[Household] = Household

    # list of planned activities (column names in person file for
    # activities)
    __planned_activity_names: ClassVar[list[str]] = []

    # list of activities
    __activity_names: ClassVar[list[str]] = []

    # activities data type: namedtuple
    __activities_data_type: namedtuple = None

    # environment
    __environment: Environment = None

    # class methods
    @classmethod
    def get_default_parameters(cls) -> dict:
        """Get the default parameters for the CasmPop model."""
        return {
            "model.name": cls.__module__ + "." + cls.__name__,
            "random.seed": 42,
            "simulation.run_id": None,
            "start.datetime": "2025-06-02 00:00:00",
            "duration.hours": 168,
            "timezone": "America/New_York",
            "time.step.minutes": 60,
            "places.table": "rti_synth_pop_v2_dmv_100.places",
            "households.table": "",
            "activities.table": "rti_synth_pop_v2_dmv_100.activities",
            "contacts.table": "rti_synth_pop_v2_dmv_100.contacts",
            "contacts.enabled": False,
            "communication.enabled": True,
            "persons.table": "rti_synth_pop_v2_dmv_100.persons",
            "behavior.engine": "default",
            "behavior.llm.enabled": False,
            "behavior.llm.deliberation_interval": 60,
            "behavior.llm.max_memory_events": 20,
            "behavior.llm.signal_cap": 1.5,
            "behavior.llm.memory_decay": 0.65,
            "behavior.activity_semantics.social_ids": [],
            "behavior.activity_semantics.flexible_ids": [],
            "behavior.activity_semantics.mandatory_ids": [],
            "behavior.activity_semantics.travel_sensitive_ids": [],
            "roads.enabled": False,
            "roads.nodes.file": "",
            "roads.edges.file": "",
            "roads.place_snap.file": "",
            "roads.mode": "drive",
            "roads.time_model": "validated_gap",
            "partition.table": "",
            "partition.default_rank": 0,
            "partition.require_full_coverage": False,
            "observers.output_dir": OBSERVER_OUTPUT_DIR_DEFAULT,
            "observers.agent_log.enabled": True,
            "observers.agent_log_file": "agent_log.parquet",
            "observers.agent_log_columns": [],
            "observers.behavior_log.enabled": False,
            "observers.behavior_log_file": "behavior_log.parquet",
            "observers.delta_agent_state.enabled": False,
            "observers.delta_agent_state_file": "agent_state_delta.parquet",
            "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
            "observers.arrow_server.enabled": False,
            "observers.arrow_server.host": "127.0.0.1",
            "logging.rank0_only": False,
            "logging.level": "INFO",
            "profiling.phase_timings.enabled": False,
            "profiling.phase_timings.sample_stride": 512,
        }

    @classmethod
    def get_default_observer_parameters(cls) -> dict:
        """Get the default observer parameters for the CasmPop model."""
        return {
            "observers.output_dir": OBSERVER_OUTPUT_DIR_DEFAULT,
            "observers.agent_log.enabled": True,
            "observers.agent_log_file": "agent_log.parquet",
            "observers.agent_log_columns": [],
            "observers.behavior_log.enabled": False,
            "observers.behavior_log_file": "behavior_log.parquet",
            "observers.delta_agent_state.enabled": False,
            "observers.delta_agent_state_file": "agent_state_delta.parquet",
            "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
            "observers.arrow_server.enabled": False,
            "observers.arrow_server.host": "127.0.0.1",
        }

    @classmethod
    def get_default_performance_parameters(cls) -> dict:
        """Get the default performance parameters for the CasmPop model.

        These parameters are used to configure the parallel processing
        settings for places and agents. The default settings are based on
        performance testing and may be adjusted based on the specific model
        and hardware configuration.
        """
        return {
            "parallel.places.enabled": True,
            "parallel.places.min_threshold": 50,
            "parallel.places.max_workers": None,  # Use CPU count
            "parallel.places.auto_update": False,
            "parallel.agents.enabled": False,
            "parallel.agents.min_threshold": 1000000,
        }

    @classmethod
    def getPersonClass(cls) -> type[Person]:
        """Get the person class."""
        return cls.__personClass

    @classmethod
    def setPersonClass(cls, person_class: type[Person], person_data) -> None:
        """Set the person class."""
        person_class.setPersonDataClass(person_data)
        cls.__personClass = person_class

    @classmethod
    def getPlaceClass(cls) -> type[Place]:
        """Get the place class."""
        return cls.__placeClass

    @classmethod
    def setPlaceClass(cls, place_class: type[Place], place_data) -> None:
        """Set the place class."""
        place_class.setPlaceDataClass(place_data)
        cls.__placeClass = place_class

    @classmethod
    def getHouseholdClass(cls) -> type[Household]:
        """Get the household class."""
        return cls.__householdClass

    @classmethod
    def setHouseholdClass(cls, household_class: type[Household], household_data) -> None:
        """Set the household class."""
        household_class.setHouseholdDataClass(household_data)
        cls.__householdClass = household_class

    @classmethod
    def register_planned_activity_names(cls, planned_activity_names: list[str]) -> None:
        """Register planned activities."""
        cls.__planned_activity_names = planned_activity_names

    @classmethod
    def get_planned_activity_names(cls) -> list[str]:
        """Get the planned activities."""
        return cls.__planned_activity_names

    @classmethod
    def register_activity_names(cls, activity_names: list[str]) -> None:
        """Register alternate activities."""
        cls.__activity_names = activity_names

    @classmethod
    def get_activity_names(cls) -> list[str]:
        """Get all activities."""
        return cls.__activity_names

    @classmethod
    def get_activities_data_type(cls) -> namedtuple:
        """Get the activities data type."""
        if not cls.__activities_data_type:
            cls.__activities_data_type = namedtuple("ActivitiesDataclass", cls.get_activity_names())
        return cls.__activities_data_type

    @classmethod
    def register_environment(cls, environment: Environment) -> None:
        """Register the environment."""
        cls.__environment = environment

    @classmethod
    def get_environment(cls) -> Environment:
        """Get the environment."""
        environment = cls.__environment
        if not environment:
            cls.__environment = SimEnvironment("sim_environment")
        return cls.__environment

    # instance variables
    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the CasmPop class

        Args:
            comm: the mpi communicator over which the model is distributed.
            params: the simulation input parameters
        """
        Model.set_model(self)

        logger.info("Creating CasmPop...")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.params = params

        # start timer
        self.start_time = time.time()
        self.phase_timings: dict[str, float] = {}

        self._validate_and_set_required_params()
        self._set_optional_params_with_defaults()
        self._compute_ticks()
        self._configure_parallel_processing()

        logger.info(f"Rank {self.rank} starting CasmPop with params: " f"{self.params}")

        # create the schedule
        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_event(0, self.build_context)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(self.params["ticks"])
        self.runner.schedule_end_event(self.at_end)

        # set the start datetime and timezone
        start_datetime = datetime.strptime(self.params["start.datetime"], "%Y-%m-%d %H:%M:%S")
        tz = ZoneInfo(self.params["timezone"])
        start_datetime = start_datetime.replace(tzinfo=tz)

        # initialize the simulation time
        self.cal = SimTime(start_datetime=start_datetime)

        # set the time step in minutes
        self.time_step_minutes = self.params["time.step.minutes"]

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(self.comm)

        # set the data resources (e.g. data paths, DuckLake connection, etc.)
        self._set_data_resources()

        self.queries = {}

        self.contact_map = {}
        self.place_to_rank: dict[int, int] = {}
        self.place_members: dict[int, list[tuple[int, int, int]]] = {}
        self.person_uid_map: dict[tuple[int, int, int], Person] = {}
        self.households_by_id: dict[int, Household] = {}
        self.households_by_place_id: dict[int, Household] = {}
        self.road_network: RoadNetwork | None = None
        self.communication_manager = CommunicationManager(self.comm)

        # **note**
        #
        # If you are using the repast4py.parameters module, you can just
        # include a 'random.seed' key in your YAML or JSON configuration file.
        # The framework will automatically call init() for you during parameter
        # initialization.

        # instance-level observers
        self._observers: list[Observer] = []

        # live Arrow/Ice observation server (optional, rank 0 only)
        self._arrow_server_handle = None

    def add_observer(self, observer: Observer) -> None:
        """Register an observer for this CasmPop instance."""
        self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer from this CasmPop instance."""
        self._observers = [obs for obs in self._observers if obs is not observer]

    def _notify_step_observers(self) -> None:
        """Notify all observers that a simulation step completed."""
        for observer in self._observers:
            observer.on_step(self)

    def _notify_end_observers(self) -> None:
        """Notify all observers that simulation ended."""
        for observer in self._observers:
            observer.on_end(self)

    def get_observer_output_tables(self) -> dict[str, pa.Table]:
        """Collect Arrow tables from all registered observers, keyed by channel name.

        Lets callers such as casmservice consume the latest observer outputs
        directly from memory instead of reading back the parquet files.
        """
        tables: dict[str, pa.Table] = {}
        for observer in self._observers:
            tables.update(observer.get_output_tables(self))
        return tables

    def _start_arrow_server_if_enabled(self) -> None:
        """Start the embedded live Arrow/Ice server on rank 0, if configured.

        Each MPI rank is a separate process with its own local agents, so
        only rank 0's local observer tables are exposed by the live server
        in this first iteration; full multi-rank aggregation is a follow-up.
        """
        if getattr(self, "_arrow_server_handle", None) is not None:
            return
        if self.rank != 0:
            return
        if not self._arrow_server_enabled():
            return

        from casmsocial.arrow_serialization import ArrowServerUnavailableError

        host = self.params.get("observers.arrow_server.host", "127.0.0.1")
        try:
            from casmsocial.arrow_server import start_arrow_server

            self._arrow_server_handle = start_arrow_server(self, host=host, endpoint_dir=pathlib.Path.cwd())
        except ArrowServerUnavailableError as exc:
            self._arrow_server_handle = None
            logger.warning(f"Arrow/Ice live observation server not started: {exc}")
            return
        logger.info(
            f"Arrow/Ice live observation server listening on "
            f"{self._arrow_server_handle.host}:{self._arrow_server_handle.port}"
        )

    def _stop_arrow_server(self) -> None:
        """Tear down the embedded live Arrow/Ice server, if it was started."""
        handle = getattr(self, "_arrow_server_handle", None)
        if handle is None:
            return
        handle.shutdown()
        self._arrow_server_handle = None

    def _set_data_resources(self) -> None:
        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        load_dotenv()  # load environment variables from .env file if it exists

        # check if the data path is set and valid
        data_path = os.environ.get("CASMSOCIAL_DATA_PATH")
        if not data_path or not pathlib.Path(data_path).exists():
            raise MissingDataPathError(data_path)
        self.data_path = pathlib.Path(data_path)

        ducklake_path = os.environ.get("CASMSOCIAL_DUCKLAKE_PATH")
        if ducklake_path:
            self.conn = get_ducklake_connection(pathlib.Path(ducklake_path))
        else:
            raise MissingDataPathError("CASMSOCIAL_DUCKLAKE_PATH")

    def _validate_and_set_required_params(self):
        """Validate and set required parameters."""
        required_keys = ["places.table", "persons.table", "activities.table"]
        for key in required_keys:
            if key not in self.params:
                logger.error(f"Missing required parameter: {key}")
                raise MissingRequiredParameterError(key)

    def _set_optional_params_with_defaults(self):
        """Set optional parameters with default values if not provided."""
        optional_keys = [
            "random.seed",
            "start.datetime",
            "duration.hours",
            "timezone",
            "time.step.minutes",
            "households.table",
            "contacts.table",
            "contacts.enabled",
            "communication.enabled",
            "partition.table",
            "partition.default_rank",
            "partition.require_full_coverage",
            "observers.output_dir",
            "observers.agent_log.enabled",
            "observers.agent_log_file",
            "observers.agent_log_columns",
            "observers.behavior_log.enabled",
            "observers.behavior_log_file",
            "observers.delta_agent_state.enabled",
            "observers.delta_agent_state_file",
            "observers.delta_agent_state_audit_file",
            "observers.arrow_server.enabled",
            "observers.arrow_server.host",
            "logging.rank0_only",
            "logging.level",
            "profiling.phase_timings.enabled",
            "profiling.phase_timings.sample_stride",
        ]
        for key in optional_keys:
            if key not in self.params:
                logger.warning(f"Optional parameter {key} not found, using " f"default value.")
                self.params[key] = None

        self._set_default_start_datetime()
        self._set_default_run_metadata()
        self._set_default_duration_hours()
        self._set_default_timezone()
        self._parse_time_step_minutes()
        self._set_default_contacts_enabled()
        self._set_default_communication_enabled()
        self._set_default_partition_params()
        self._set_default_agent_log_enabled()
        self._set_default_agent_log_columns()
        self._set_default_observer_outputs()
        self._set_default_arrow_server_params()
        self._set_default_logging_params()
        self._set_default_profiling_params()

    def _set_default_start_datetime(self):
        if self.params["start.datetime"] is None:
            midnight = get_midnight(get_closest_monday(datetime.now()))
            self.params["start.datetime"] = midnight.strftime("%Y-%m-%d %H:%M:%S")

    def _set_default_run_metadata(self) -> None:
        if self.params["random.seed"] is None:
            self.params["random.seed"] = 42
        self.params["random.seed"] = int(self.params["random.seed"])
        self.params["simulation.run_id"] = _model_run_id(self)

    def _set_default_duration_hours(self):
        if self.params["duration.hours"] is None:
            self.params["duration.hours"] = 24

    def _set_default_timezone(self):
        if self.params["timezone"] is None:
            self.params["timezone"] = "America/New_York"

    def _parse_time_step_minutes(self):
        if self.params["time.step.minutes"] is None:
            self.params["time.step.minutes"] = 60
            return
        if isinstance(self.params["time.step.minutes"], str):
            try:
                self.params["time.step.minutes"] = int(self.params["time.step.minutes"])
            except ValueError as err:
                logger.error(f"Invalid time step value: " f"{self.params['time.step.minutes']}")
                raise InvalidTimeStepError(self.params["time.step.minutes"]) from err
        if self.params["time.step.minutes"] <= 0:
            logger.error(
                f"Invalid time step value: "
                f"{self.params['time.step.minutes']}. "
                f"Time step must be a positive integer."
            )
            raise InvalidTimeStepError(self.params["time.step.minutes"])
        if 1440 % self.params["time.step.minutes"] != 0:
            logger.error(
                f"Invalid time step value: "
                f"{self.params['time.step.minutes']}. Time step must be "
                f"a divisor of 1440 (minutes in a day)."
            )
            raise InvalidTimeStepError(self.params["time.step.minutes"])

    def _set_default_contacts_enabled(self) -> None:
        if self.params["contacts.enabled"] is None:
            self.params["contacts.enabled"] = False

    def _set_default_communication_enabled(self) -> None:
        if self.params["communication.enabled"] is None:
            self.params["communication.enabled"] = True

    def _set_default_agent_log_enabled(self) -> None:
        if self.params["observers.agent_log.enabled"] is None:
            self.params["observers.agent_log.enabled"] = True

    def _set_default_agent_log_columns(self) -> None:
        value = self.params["observers.agent_log_columns"]
        if value is None:
            self.params["observers.agent_log_columns"] = []
            return
        if isinstance(value, str) or not isinstance(value, list | tuple):
            raise InvalidAgentLogColumnsError(value)
        self.params["observers.agent_log_columns"] = [str(column) for column in value]

    def _set_default_arrow_server_params(self) -> None:
        if self.params["observers.arrow_server.enabled"] is None:
            self.params["observers.arrow_server.enabled"] = False
        host = self.params["observers.arrow_server.host"]
        if host is None or not str(host).strip():
            self.params["observers.arrow_server.host"] = "127.0.0.1"
        else:
            self.params["observers.arrow_server.host"] = str(host)

    def _set_default_observer_outputs(self) -> None:
        if self.params["observers.output_dir"] is None or not str(self.params["observers.output_dir"]).strip():
            self.params["observers.output_dir"] = OBSERVER_OUTPUT_DIR_DEFAULT
        else:
            self.params["observers.output_dir"] = str(self.params["observers.output_dir"])
        if self.params["observers.behavior_log.enabled"] is None:
            self.params["observers.behavior_log.enabled"] = False
        if self.params["observers.delta_agent_state.enabled"] is None:
            self.params["observers.delta_agent_state.enabled"] = False

        for key, default_filename in OBSERVER_OUTPUT_FILE_DEFAULTS.items():
            self.params[key] = _observer_output_filename(
                key,
                self.params[key],
                default_filename,
            )

    def _set_default_partition_params(self) -> None:
        if self.params["partition.table"] is None:
            self.params["partition.table"] = ""
        if self.params["partition.default_rank"] is None:
            self.params["partition.default_rank"] = 0
        if self.params["partition.require_full_coverage"] is None:
            self.params["partition.require_full_coverage"] = False

    def _set_default_logging_params(self) -> None:
        if self.params["logging.rank0_only"] is None:
            self.params["logging.rank0_only"] = False
        if self.params["logging.level"] is None:
            self.params["logging.level"] = "INFO"

    def _set_default_profiling_params(self) -> None:
        if self.params["profiling.phase_timings.enabled"] is None:
            self.params["profiling.phase_timings.enabled"] = False
        if self.params["profiling.phase_timings.sample_stride"] is None:
            self.params["profiling.phase_timings.sample_stride"] = 512

    def _param_enabled(self, key: str, default: bool = False) -> bool:
        value = getattr(self, "params", {}).get(key, default)
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _contacts_enabled(self) -> bool:
        return self._param_enabled("contacts.enabled", False)

    def _communication_enabled(self) -> bool:
        return self._param_enabled("communication.enabled", True)

    def _person_step_enabled(self) -> bool:
        """Return whether the per-person behavior step should run after environment movement."""
        return not (self.getPersonClass() is Person and Person.getBehaviorEngine() is ScheduleBehaviorEngine)

    def _agent_log_enabled(self) -> bool:
        return self._param_enabled("observers.agent_log.enabled", True)

    def _arrow_server_enabled(self) -> bool:
        return self._param_enabled("observers.arrow_server.enabled", False)

    def _phase_timings_enabled(self) -> bool:
        return self._param_enabled("profiling.phase_timings.enabled", False)

    def _phase_timing_sample_stride(self) -> int:
        """Return the row stride for sampled detailed phase timings."""
        value = self.params.get("profiling.phase_timings.sample_stride", 512)
        if value is None:
            return 512
        return max(0, int(value))

    def _record_phase_timing(self, phase: str, duration: float) -> None:
        if self._phase_timings_enabled():
            if not hasattr(self, "phase_timings"):
                self.phase_timings = {}
            self.phase_timings[phase] = self.phase_timings.get(phase, 0.0) + duration

    def _time_phase(self, phase: str, func, *args, **kwargs):
        if not self._phase_timings_enabled():
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._record_phase_timing(phase, time.perf_counter() - start)

    def _log_phase_timing_summary(self) -> None:
        if not self._phase_timings_enabled():
            return

        local_timings = dict(self.phase_timings)
        gather = getattr(self.comm, "gather", None)
        gathered = gather(local_timings, root=0) if gather is not None else [local_timings]

        if self.rank != 0:
            return
        gathered = gathered or []
        phases = sorted({phase for rank_timings in gathered for phase in rank_timings})
        if not phases:
            logger.info("Phase timing summary: no timings recorded.")
            return

        logger.info("Phase timing summary (seconds, min/avg/max across ranks):")
        for phase in phases:
            values = [rank_timings.get(phase, 0.0) for rank_timings in gathered]
            avg_value = sum(values) / len(values)
            logger.info(f"  {phase}: " f"min={min(values):.3f}, " f"avg={avg_value:.3f}, " f"max={max(values):.3f}")

    def _contacts_table_name(self) -> str | None:
        contacts_table = self.params.get("contacts.table")
        if contacts_table is None:
            return None
        contacts_table = str(contacts_table).strip()
        return contacts_table or None

    def _households_table_name(self) -> str | None:
        households_table = self.params.get("households.table")
        if households_table is None:
            return None
        households_table = str(households_table).strip()
        return households_table or None

    def _compute_ticks(self):
        if "time.step.minutes" not in self.params or "duration.hours" not in self.params:
            logger.error("Missing required parameters: time.step.minutes or " "duration.hours")
            raise MissingRequiredParameterError(["time.step.minutes", "duration.hours"])
        self.params["ticks"] = int(self.params["duration.hours"] * 60 / self.params["time.step.minutes"])

    def _configure_parallel_processing(self):
        """Configure parallel processing settings."""
        # Default parallel processing settings
        if "parallel.places.enabled" not in self.params:
            self.params["parallel.places.enabled"] = True
        if "parallel.places.min_threshold" not in self.params:
            self.params["parallel.places.min_threshold"] = 50
        if "parallel.places.max_workers" not in self.params:
            self.params["parallel.places.max_workers"] = None

        # Disable automatic place updates during simulation steps
        if "parallel.places.auto_update" not in self.params:
            self.params["parallel.places.auto_update"] = False

        # Agent processing: parallel disabled due to performance degradation
        # Thread overhead exceeded benefits for lightweight operations
        if "parallel.agents.enabled" not in self.params:
            self.params["parallel.agents.enabled"] = False
        if "parallel.agents.min_threshold" not in self.params:
            self.params["parallel.agents.min_threshold"] = 1000000

    def build_context(self) -> None:
        """
        Initialize population.

        This method initializes the population by creating the places and
        agents from the input data files.
        """
        # derived classes can extend this method to register their own agent
        # types and activities and create the specific agents, places, and
        # activities for their model.
        # Note: if the activity names are already registered (e.g. by a derived
        # class), we skip registering activities here to avoid overwriting the
        # activity names registered by the derived class.
        if len(self.get_activity_names()) == 0:
            logger.info("Registering activities for base class CasmPop...")

            # register the activities (default is just home, work, school -
            #  can be extended by derived classes)
            CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
            CasmPop.register_activity_names(["home", "work", "school"])

            logger.info("Now running initialize population for CasmPop...")

            # add a basic agent logger observer if no observers have been
            # registered yet (e.g. by a derived class)
            if len(self._observers) == 0 and self._agent_log_enabled():
                self.add_observer(AgentLogger("AgentLogger", self))
            if self.params.get("observers.behavior_log.enabled", False):
                self.add_observer(BehaviorLogger("BehaviorLogger", self))
            if self._param_enabled("observers.delta_agent_state.enabled", False):
                self.add_observer(DeltaAgentStateLogger("DeltaAgentStateLogger", self))
        else:
            logger.info("Activity names already registered by derived class...")

        self._start_arrow_server_if_enabled()

        # create SharedContext consisting of all places in this model
        # Use enhanced projection with configurable parallel processing
        self._time_phase("startup.configure_behavior", self._configure_behavior_engine)
        parallel_enabled = self.params.get("parallel.places.enabled", True)
        parallel_min_threshold = self.params.get("parallel.places.min_threshold", 50)
        parallel_max_workers = self.params.get("parallel.places.max_workers", None)

        self.places_proj = PlacesProjectionV2(
            "places_projection",
            self.comm,
            enable_parallel_updates=parallel_enabled,
            parallel_min_threshold=parallel_min_threshold,
            parallel_max_workers=parallel_max_workers,
        )
        self.context.add_projection(self.places_proj)

        # create the input tables
        self._time_phase("startup.create_input_tables", self.create_input_tables)

        # initialize the places
        # (note: already checked if "places.file" is in params)
        self._time_phase("startup.create_places", self.create_places)
        self._time_phase("startup.initialize_place_rank_index", self._initialize_place_rank_index)
        self._time_phase("startup.create_households", self.create_households)
        self._time_phase("startup.load_road_network", self.load_road_network)

        local_places = self.places_proj.get_local_places()
        logger.info(f"rank {self.rank}: number of local " f"places={len(local_places)}")
        # add geometry to the places table
        # self.conn.execute(self.queries["add_geometries"])

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons
        # at each place
        contacts_table = self._contacts_table_name()
        if self._contacts_enabled():
            if contacts_table is None:
                raise MissingRequiredParameterError("contacts.table")
            logger.info(f"Loading contact table {contacts_table}...")
            self.contact_map = self._time_phase("startup.create_contacts", self.create_contacts)
        else:
            self.contact_map = {}
            if contacts_table:
                logger.info("contacts.enabled is false; skipping contacts table load.")
            else:
                logger.info("contacts table not specified; skipping contacts table load.")

        logger.debug("rank {}: contacts size={}", self.rank, len(self.contact_map))

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        # self.person_id_map = {}
        self._time_phase("startup.create_persons", self.create_persons, self.rng)
        if self._communication_enabled():
            self._time_phase("startup.refresh_person_index", self.refresh_person_index)
            self._time_phase("startup.refresh_place_membership", self.refresh_place_membership)

        result = self.conn.execute(self.queries["get_tables"]).fetchall()
        logger.info(f"rank {self.rank}: DuckDB tables after " f"initialization: {result}")

    def _configure_behavior_engine(self) -> None:
        """Register the configured person behavior engine.

        ``behavior.engine`` selects the engine class. Recognized values:

        * ``"default"`` (or anything unrecognized) — :class:`BehaviorEngineV2`,
          the deterministic cognitive baseline.
        * ``"schedule"`` — :class:`ScheduleBehaviorEngine`, a minimal engine
          that only follows the existing activity schedule.
        * ``"llm"`` — :class:`LLMBehaviorEngine`. The adapter behind the engine
          is selected separately by ``behavior.llm.adapter``.
        * ``"llm_local"`` — back-compat alias for ``"llm"`` from before the
          adapter became configurable.

        ``behavior.llm.enabled: true`` also enables the LLM engine, kept for
        back-compat. New configs should prefer ``behavior.engine: 'llm'``.

        For ``behavior.engine: 'llm'``, the adapter is chosen by
        ``behavior.llm.adapter`` (default ``"local"``). Recognized values:

        * ``"local"`` — :class:`LocalBehaviorLLMAdapter`. No network. Default.
        * ``"anthropic"`` — :class:`AnthropicBehaviorLLMAdapter`. Calls
          Anthropic Claude via the optional ``anthropic`` package
          (``uv sync --extra remote``). Configured by the
          ``behavior.llm.anthropic.*`` keys.
        """
        engine_name = str(self.params.get("behavior.engine", "default"))
        llm_enabled = bool(self.params.get("behavior.llm.enabled", False))
        if engine_name == "schedule":
            Person.registerBehaviorEngine(ScheduleBehaviorEngine)
            return
        if engine_name in ("llm", "llm_local") or llm_enabled:
            LLMBehaviorEngine.configure(
                # Adapter selection.
                adapter=str(self.params.get("behavior.llm.adapter", "local")),
                # Local-adapter parameters (also consumed when adapter == "local").
                deliberation_interval=int(self.params.get("behavior.llm.deliberation_interval", 60)),
                max_memory_events=int(self.params.get("behavior.llm.max_memory_events", 20)),
                signal_cap=float(self.params.get("behavior.llm.signal_cap", 1.5)),
                memory_decay=float(self.params.get("behavior.llm.memory_decay", 0.65)),
                activity_semantics_overrides={
                    "social_ids": list(self.params.get("behavior.activity_semantics.social_ids", [])),
                    "flexible_ids": list(self.params.get("behavior.activity_semantics.flexible_ids", [])),
                    "mandatory_ids": list(self.params.get("behavior.activity_semantics.mandatory_ids", [])),
                    "travel_sensitive_ids": list(
                        self.params.get("behavior.activity_semantics.travel_sensitive_ids", [])
                    ),
                },
                # Anthropic-adapter parameters (consumed only when adapter == "anthropic").
                # api_key=None lets the SDK read ANTHROPIC_API_KEY from the environment.
                anthropic_model=str(self.params.get("behavior.llm.anthropic.model", "claude-haiku-4-5-20251001")),
                anthropic_api_key=self.params.get("behavior.llm.anthropic.api_key"),
                anthropic_max_tokens=int(self.params.get("behavior.llm.anthropic.max_tokens", 1024)),
            )
            Person.registerBehaviorEngine(LLMBehaviorEngine)
            return
        Person.registerBehaviorEngine(BehaviorEngineV2)

    def _initialize_place_rank_index(self) -> None:
        """Cache the fixed rank owner for each place."""
        get_place_rank_map = getattr(self.places_proj, "get_place_rank_map", None)
        if get_place_rank_map is not None:
            place_rank_map = get_place_rank_map()
            if place_rank_map:
                self.place_to_rank = place_rank_map
                return
        self.place_to_rank = {place.id: place.rank for place in self.places_proj.get_all_places()}

    def load_road_network(self) -> None:
        """Load optional road-network artifacts for routed travel legs."""
        if not self.params.get("roads.enabled"):
            self.road_network = None
            return

        nodes = self._load_road_nodes(self.params["roads.nodes.file"])
        edges = self._load_road_edges(self.params["roads.edges.file"])
        place_snaps = self._load_place_snaps(self.params["roads.place_snap.file"])
        self.road_network = RoadNetwork.from_tables(nodes, edges, place_snaps)

    def _load_road_nodes(self, table_or_file: str) -> list[dict]:
        """Load road-node records from the configured source."""
        return self._load_records(table_or_file, "road nodes")

    def _load_road_edges(self, table_or_file: str) -> list[dict]:
        """Load road-edge records from the configured source."""
        return self._load_records(table_or_file, "road edges")

    def _load_place_snaps(self, table_or_file: str) -> list[dict]:
        """Load place-to-road-node snap records from the configured source."""
        return self._load_records(table_or_file, "place snaps")

    def _load_records(self, table_or_file: str, label: str) -> list[dict]:
        """Load records from a DuckDB table or parquet file."""
        if not table_or_file:
            raise MissingRequiredParameterError(f"roads {label} source")

        if check_if_table_exists(self.conn, table_or_file):
            identifier = quote_table_identifier(table_or_file)
            logger.info(f"Loading {label} from table {table_or_file}...")
            table = self.conn.execute(f"SELECT * FROM {identifier}").arrow().read_all()  # noqa: S608
            return table.to_pylist()

        source_path = pathlib.Path(table_or_file)
        if not source_path.is_absolute():
            source_path = self.data_path / source_path

        if not source_path.exists():
            raise MissingRequiredTableError(table_or_file)

        if source_path.suffix != ".parquet":
            raise ValueError(f"Unsupported {label} source: {source_path}")

        logger.info(f"Loading {label} from parquet file {source_path}...")
        table = self.conn.execute("SELECT * FROM read_parquet(?)", [str(source_path)]).arrow().read_all()
        return table.to_pylist()

    def refresh_person_index(self) -> None:
        """Rebuild the local uid-to-person lookup after migration."""
        self.person_uid_map = {person.uid: person for person in _person_agents(self.context)}

    def refresh_place_membership(self) -> None:
        """Rebuild the model-level place membership index for locally owned people."""
        members: dict[int, list[tuple[int, int, int]]] = {}
        for person in _person_agents(self.context):
            members.setdefault(person.rank_place_id, []).append(person.uid)
        self.place_members = members

    def sync_person_ranks_with_places(self) -> list[tuple[tuple[int, int, int], int]]:
        """Return context moves needed to align person ownership with current place ownership."""
        moves_to_apply: list[tuple[tuple[int, int, int], int]] = []
        if self._partition_table_name() is None:
            return moves_to_apply

        get_place_for_agent = getattr(self.places_proj, "get_place_for_agent", None)
        remove_agent_from_place = getattr(self.places_proj, "remove_agent_from_place", None)
        for person in _person_agents(self.context):
            target_rank = self.place_to_rank.get(person.rank_place_id)
            if target_rank is None:
                logger.debug("No owner rank found for place {}", person.rank_place_id)
                continue
            if int(target_rank) == self.rank:
                continue

            current_place = get_place_for_agent(person) if get_place_for_agent is not None else None
            if current_place is not None and remove_agent_from_place is not None:
                remove_agent_from_place(person)
            moves_to_apply.append((person.uid, int(target_rank)))
        return moves_to_apply

    def _rank_moves_from_moved_people_enabled(self) -> bool:
        """Return whether movement can collect rank migrations without a second person scan."""
        return (
            self._partition_table_name() is not None
            and self.getPersonClass() is Person
            and Person.getBehaviorEngine() is ScheduleBehaviorEngine
        )

    def _rank_move_for_person(self, person: Person) -> tuple[tuple[int, int, int], int] | None:
        """Return a context move for one moved person, if their destination is remote."""
        target_rank = self.place_to_rank.get(person.rank_place_id)
        if target_rank is None:
            logger.debug("No owner rank found for place {}", person.rank_place_id)
            return None
        target_rank = int(target_rank)
        if target_rank == self.rank:
            return None

        remove_agent_from_place = getattr(self.places_proj, "remove_agent_from_place", None)
        get_place_for_agent = getattr(self.places_proj, "get_place_for_agent", None)
        current_place = get_place_for_agent(person) if get_place_for_agent is not None else None
        if current_place is not None and remove_agent_from_place is not None:
            remove_agent_from_place(person)
        return (person.uid, target_rank)

    def sync_place_projection_memberships(self) -> None:
        """Ensure local people are attached to local place objects after migration."""
        get_place_for_agent = getattr(self.places_proj, "get_place_for_agent", None)
        lookup_place = getattr(self.places_proj, "lookup_place", None)
        assign_agent_to_place = getattr(self.places_proj, "assign_agent_to_place", None)
        move_agent_to_place = getattr(self.places_proj, "move_agent_to_place", None)
        remove_agent_from_place = getattr(self.places_proj, "remove_agent_from_place", None)
        if lookup_place is None or assign_agent_to_place is None:
            return

        for person in _person_agents(self.context):
            current_place = get_place_for_agent(person) if get_place_for_agent is not None else None
            if person.place_id == 0 or person.place_id != person.rank_place_id:
                if current_place is not None and remove_agent_from_place is not None:
                    remove_agent_from_place(person)
                continue

            place = lookup_place(person.place_id)
            if place is None:
                continue
            if current_place is None:
                assign_agent_to_place(person, place)
            elif current_place.id != place.id and move_agent_to_place is not None:
                move_agent_to_place(person, place)

    def get_person_by_uid(self, uid: tuple[int, int, int]) -> Person | None:
        """Lookup a locally owned person by uid."""
        return self.person_uid_map.get(uid)

    def collect_message_intents(self) -> list[MessageIntent]:
        """Gather communication intents from locally owned people."""
        intents: list[MessageIntent] = []
        for person in _person_agents(self.context):
            intents.extend(person.decide_messages(self))
        return intents

    def process_person_inboxes(self) -> None:
        """Run the inbox-processing phase for locally owned people."""
        for person in _person_agents(self.context):
            person.process_inbox(self)

    def _global_message_intent_count(self, local_count: int) -> int:
        """Return the total number of message intents across all ranks."""
        return int(self.comm.allreduce(int(local_count), op=MPI.SUM))

    def _global_rank_move_count(self, local_count: int) -> int:
        """Return the total number of pending person rank moves across all ranks."""
        return int(self.comm.allreduce(int(local_count), op=MPI.SUM))

    def run_communication_phases(self) -> None:
        """Execute post-movement communication phases for the current tick."""
        if not self._communication_enabled():
            return

        self.refresh_person_index()
        self.refresh_place_membership()

        tick = int(self.cal.tick)
        intents = self.collect_message_intents()

        self.communication_manager.clear_buffers()
        if self._global_message_intent_count(len(intents)) == 0:
            return

        self.communication_manager.route(intents, self, tick)
        self.communication_manager.exchange_remote(self)
        self.communication_manager.exchange_acks(self)
        self.process_person_inboxes()

    @staticmethod
    def _quote_column_identifier(column_name: str) -> str:
        """Quote a DuckDB column identifier."""
        escaped = column_name.replace('"', '""')
        return f'"{escaped}"'

    def _select_columns_expr(self, table_name: str, alias: str, exclude: set[str] | None = None) -> str:
        """Build a qualified select-list for a table, excluding named columns."""
        exclude_lower = {column.lower() for column in (exclude or set())}
        table_identifier = quote_table_identifier(table_name)
        columns = [
            column[0]
            for column in self.conn.execute(f"SELECT * FROM {table_identifier} LIMIT 0").description
            if column[0].lower() not in exclude_lower
        ]
        if not columns:
            raise MissingRequiredTableError(table_name)
        return ", ".join(f"{alias}.{self._quote_column_identifier(column)}" for column in columns)

    def _select_named_columns_expr(
        self,
        table_name: str,
        alias: str,
        required: list[str],
        optional: list[str] | None = None,
        exclude: set[str] | None = None,
    ) -> str:
        """Build a qualified select-list from required plus available optional columns."""
        table_identifier = quote_table_identifier(table_name)
        available_columns = [
            column[0] for column in self.conn.execute(f"SELECT * FROM {table_identifier} LIMIT 0").description
        ]
        available_column_set = set(available_columns)
        missing_columns = [column for column in required if column not in available_column_set]
        if missing_columns:
            raise MissingRequiredTableError([f"{table_name}.{column}" for column in missing_columns])

        exclude_lower = {column.lower() for column in (exclude or set())}
        selected_columns: list[str] = []
        for column in [*required, *(optional or [])]:
            if column not in available_column_set:
                continue
            if column.lower() in exclude_lower:
                continue
            if column in selected_columns:
                continue
            selected_columns.append(column)

        return ", ".join(f"{alias}.{self._quote_column_identifier(column)}" for column in selected_columns)

    def _use_minimal_place_input_columns(self) -> bool:
        """Return whether default place input can be narrowed before materialization."""
        place_class = self.getPlaceClass()
        return place_class is Place and place_class.getPlaceDataClass() is PlaceData

    def _use_minimal_person_input_columns(self) -> bool:
        """Return whether default person input can be narrowed before materialization."""
        person_class = self.getPersonClass()
        return person_class is Person and person_class.getPersonDataClass() is PersonData

    def _minimal_place_columns_expr(self, table_name: str, alias: str, exclude: set[str] | None = None) -> str:
        """Build the source columns needed by the default PlaceData path."""
        return self._select_named_columns_expr(
            table_name,
            alias,
            required=["sp_id"],
            optional=["rank", "place_type", "place_name", "latitude", "longitude"],
            exclude=exclude,
        )

    def _minimal_person_columns_expr(self, table_name: str, alias: str, exclude: set[str] | None = None) -> str:
        """Build the source columns needed by the default PersonData path."""
        planned_activity_names = self.get_planned_activity_names()
        return self._select_named_columns_expr(
            table_name,
            alias,
            required=list(dict.fromkeys(["sp_id", "sp_hh_id", *planned_activity_names])),
            optional=["rank", "x", "y", "network", "minute_last_moved"],
            exclude=exclude,
        )

    def _activity_columns_expr(self, table_name: str, alias: str) -> str:
        """Build the activity columns consumed by create_activities."""
        required = [
            "sp_persons_id",
            "activity_id",
            "activity_sequence",
            "starttime_min",
            "endtime_min",
            "sp_act_id",
        ]
        table_identifier = quote_table_identifier(table_name)
        available_columns = [
            column[0] for column in self.conn.execute(f"SELECT * FROM {table_identifier} LIMIT 0").description
        ]
        available_column_set = set(available_columns)
        missing_columns = [column for column in required if column not in available_column_set]
        if missing_columns:
            raise MissingRequiredTableError([f"{table_name}.{column}" for column in missing_columns])

        return ", ".join(
            f"CAST(trunc({alias}.{self._quote_column_identifier(column)}) AS BIGINT) AS "
            f"{self._quote_column_identifier(column)}"
            for column in required
        )

    def _partition_table_name(self) -> str | None:
        """Return the configured partition table name, if enabled."""
        partition_table = self.params.get("partition.table", "")
        if partition_table is None:
            return None
        partition_table = str(partition_table).strip()
        return partition_table or None

    def _partition_imputation(self) -> int:
        """Resolve the imputation value used by the partition table."""
        if "partition.imputation" in self.params:
            return int(self.params["partition.imputation"])
        if "Imputation" in self.params and self.params["Imputation"] is not None:
            return int(self.params["Imputation"])
        return 1

    def _partition_default_rank(self) -> int:
        """Resolve fallback rank for rows absent from the partition table."""
        return int(self.params.get("partition.default_rank", 0))

    def _require_full_partition_coverage(self) -> bool:
        """Return whether missing partition assignments should fail startup."""
        return bool(self.params.get("partition.require_full_coverage", False))

    def _validate_partition_table_local(self, places_table: str, partition_table: str) -> None:
        """Validate partition rows using this process' DuckDB connection."""
        if not check_if_table_exists(self.conn, partition_table):
            raise MissingRequiredTableError(partition_table)

        partition_identifier = quote_table_identifier(partition_table)
        places_identifier = quote_table_identifier(places_table)
        partition_imputation = self._partition_imputation()
        require_full_coverage = self._require_full_partition_coverage()

        partition_count, invalid_rank_count = self.conn.execute(
            f"""
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN rank < 0 OR rank >= ? THEN 1 ELSE 0 END), 0)
            FROM {partition_identifier}
            WHERE imputation = ? AND n_ranks = ?
            """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
            [self.size, partition_imputation, self.size],
        ).fetchone()
        if partition_count == 0:
            message = (
                f"No partition rows found in {partition_table} for "
                f"(imputation={partition_imputation}, n_ranks={self.size})"
            )
            if require_full_coverage:
                raise MissingPartitionAssignmentError(message)
            logger.warning(f"{message}; all places will use default rank {self._partition_default_rank()}.")
            return

        if invalid_rank_count:
            raise InvalidPartitionRankError(
                f"Partition table {partition_table} has {invalid_rank_count} rank assignments outside [0, {self.size})."
            )

        missing_assignment_count = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {places_identifier} p
            LEFT JOIN {partition_identifier} part
                   ON part.place_id = p.sp_id
                  AND part.imputation = ?
                  AND part.n_ranks = ?
            WHERE part.place_id IS NULL
            """,  # noqa: S608 - table identifiers are validated by quote_table_identifier.
            [partition_imputation, self.size],
        ).fetchone()[0]
        if not missing_assignment_count:
            return

        message = (
            f"Partition table {partition_table} is missing {missing_assignment_count} place assignments "
            f"for (imputation={partition_imputation}, n_ranks={self.size})"
        )
        if require_full_coverage:
            raise MissingPartitionAssignmentError(message)
        logger.warning(f"{message}; missing places will use default rank {self._partition_default_rank()}.")

    def _validate_partition_table(self, places_table: str, partition_table: str) -> None:
        """Validate partition rows once per MPI run before materializing temp tables."""
        comm = getattr(self, "comm", None)
        bcast = getattr(comm, "bcast", None)
        if bcast is None or getattr(self, "size", 1) <= 1:
            self._validate_partition_table_local(places_table, partition_table)
            return

        validation_error: Exception | None = None
        if getattr(self, "rank", 0) == 0:
            try:
                self._validate_partition_table_local(places_table, partition_table)
            except Exception as exc:
                validation_error = exc

        validation_error = bcast(validation_error, root=0)
        if validation_error is not None:
            raise validation_error

    def _validate_partition_table_exists(self, partition_table: str) -> None:
        """Validate partition table presence once per MPI run."""
        comm = getattr(self, "comm", None)
        bcast = getattr(comm, "bcast", None)
        if bcast is None or getattr(self, "size", 1) <= 1:
            if not check_if_table_exists(self.conn, partition_table):
                raise MissingRequiredTableError(partition_table)
            return

        validation_error: Exception | None = None
        if getattr(self, "rank", 0) == 0:
            try:
                if not check_if_table_exists(self.conn, partition_table):
                    raise MissingRequiredTableError(partition_table)
            except Exception as exc:
                validation_error = exc

        validation_error = bcast(validation_error, root=0)
        if validation_error is not None:
            raise validation_error

    def _validate_materialized_full_partition_table_local(self, places_table: str, partition_table: str) -> None:
        """Validate the full-coverage partition after place_ranks is materialized."""
        places_identifier = quote_table_identifier(places_table)
        partition_imputation = self._partition_imputation()

        partition_count, invalid_rank_count = self.conn.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN rank < 0 OR rank >= ? THEN 1 ELSE 0 END), 0)
            FROM place_ranks
            """,
            [self.size],
        ).fetchone()
        if partition_count == 0:
            raise MissingPartitionAssignmentError(
                f"No partition rows found in {partition_table} for "
                f"(imputation={partition_imputation}, n_ranks={self.size})"
            )

        if invalid_rank_count:
            raise InvalidPartitionRankError(
                f"Partition table {partition_table} has {invalid_rank_count} rank assignments outside [0, {self.size})."
            )

        missing_assignment_count = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {places_identifier} p
            ANTI JOIN place_ranks
                    ON place_ranks.sp_id = p.sp_id
            """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
        ).fetchone()[0]
        if missing_assignment_count:
            raise MissingPartitionAssignmentError(
                f"Partition table {partition_table} is missing {missing_assignment_count} place assignments "
                f"for (imputation={partition_imputation}, n_ranks={self.size})"
            )

    def _validate_materialized_full_partition_table(self, places_table: str, partition_table: str) -> None:
        """Validate materialized full-coverage partition rows once per MPI run."""
        comm = getattr(self, "comm", None)
        bcast = getattr(comm, "bcast", None)
        if bcast is None or getattr(self, "size", 1) <= 1:
            self._validate_materialized_full_partition_table_local(places_table, partition_table)
            return

        validation_error: Exception | None = None
        if getattr(self, "rank", 0) == 0:
            try:
                self._validate_materialized_full_partition_table_local(places_table, partition_table)
            except Exception as exc:
                validation_error = exc

        validation_error = bcast(validation_error, root=0)
        if validation_error is not None:
            raise validation_error

    def _create_places_input_table(
        self,
        places_table: str,
        partition_table: str | None,
        partition_exists_validated: bool = False,
    ) -> None:
        """Create the temporary places table, optionally overlaying partition ranks."""
        places_identifier = quote_table_identifier(places_table)
        if partition_table is None:
            if self._use_minimal_place_input_columns():
                place_columns = self._minimal_place_columns_expr(places_table, "p")
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TEMPORARY TABLE places AS
                    SELECT {place_columns}
                    FROM {places_identifier} p
                    """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                )
                return
            self.conn.execute(
                "CREATE OR REPLACE TEMPORARY TABLE places AS "  # noqa: S608
                "SELECT * FROM "
                f"{places_identifier}"
            )
            return

        partition_identifier = quote_table_identifier(partition_table)
        if self._use_minimal_place_input_columns():
            place_columns = self._minimal_place_columns_expr(places_table, "p", exclude={"rank"})
        else:
            place_columns = self._select_columns_expr(places_table, "p", exclude={"rank"})

        if self._require_full_partition_coverage():
            if not partition_exists_validated:
                self._time_phase(
                    "startup.create_input_tables.create_places_table.validate_partition_exists",
                    self._validate_partition_table_exists,
                    partition_table,
                )
            self._time_phase(
                "startup.create_input_tables.create_places_table.create_place_ranks",
                self.conn.execute,
                f"""
                    CREATE OR REPLACE TEMPORARY TABLE place_ranks AS
                    SELECT
                        CAST(part.place_id AS BIGINT) AS sp_id,
                        CAST(part.rank AS INTEGER) AS rank
                    FROM {partition_identifier} part
                    WHERE part.imputation = ?
                      AND part.n_ranks = ?
                    """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                [self._partition_imputation(), self.size],
            )
            self._time_phase(
                "startup.create_input_tables.create_places_table.validate_place_ranks",
                self._validate_materialized_full_partition_table,
                places_table,
                partition_table,
            )
        else:
            self._time_phase(
                "startup.create_input_tables.create_places_table.validate_partition",
                self._validate_partition_table,
                places_table,
                partition_table,
            )
            self._time_phase(
                "startup.create_input_tables.create_places_table.create_place_ranks",
                self.conn.execute,
                f"""
                    CREATE OR REPLACE TEMPORARY TABLE place_ranks AS
                    SELECT
                        p.sp_id,
                        CAST(COALESCE(part.rank, ?) AS INTEGER) AS rank
                    FROM {places_identifier} p
                    LEFT JOIN {partition_identifier} part
                           ON part.place_id = p.sp_id
                          AND part.imputation = ?
                          AND part.n_ranks = ?
                    """,  # noqa: S608 - table identifiers are validated by quote_table_identifier.
                [self._partition_default_rank(), self._partition_imputation(), self.size],
            )

        if self._require_full_partition_coverage():
            self._time_phase(
                "startup.create_input_tables.create_places_table.create_local_places",
                self.conn.execute,
                f"""
                    CREATE OR REPLACE TEMPORARY TABLE places AS
                    SELECT
                        {place_columns},
                        CAST(? AS INTEGER) AS rank
                    FROM {places_identifier} p
                    WHERE EXISTS (
                        SELECT 1
                        FROM place_ranks
                        WHERE place_ranks.sp_id = p.sp_id
                          AND place_ranks.rank = ?
                    )
                    """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                [self.rank, self.rank],
            )
        else:
            self._time_phase(
                "startup.create_input_tables.create_places_table.create_local_places",
                self.conn.execute,
                f"""
                    CREATE OR REPLACE TEMPORARY TABLE places AS
                    SELECT
                        {place_columns},
                        place_ranks.rank
                    FROM {places_identifier} p
                    INNER JOIN place_ranks
                            ON place_ranks.sp_id = p.sp_id
                    WHERE place_ranks.rank = ?
                    """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                [self.rank],
            )

    def _create_persons_input_table(
        self, persons_table: str, imputation: int | None, partition_table: str | None
    ) -> None:
        """Create the temporary persons table, optionally limited to this rank."""
        persons_identifier = quote_table_identifier(persons_table)
        if partition_table is None:
            if imputation is not None:
                logger.info(f"Using imputation {imputation} for <persons> " f"table <{persons_table}>...")
                logger.info(f"persons_identifier: {persons_identifier}")
                if self._use_minimal_person_input_columns():
                    person_columns = self._minimal_person_columns_expr(persons_table, "pe")
                    self.conn.execute(
                        f"""
                        CREATE OR REPLACE TEMPORARY TABLE persons AS
                        SELECT {person_columns}
                        FROM {persons_identifier} pe
                        WHERE pe.Imputation = ?;
                        """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                        [imputation],
                    )
                    logger.info("Created persons view with imputation filter.")
                    return
                self.conn.execute(
                    f"""CREATE OR REPLACE TEMPORARY TABLE persons AS
                    SELECT * FROM {persons_identifier}
                    WHERE Imputation = ?;
                    """,  # noqa: S608
                    [imputation],
                )
                logger.info("Created persons view with imputation filter.")
                return

            logger.info(f"Using <persons> table {persons_table}...")
            if self._use_minimal_person_input_columns():
                person_columns = self._minimal_person_columns_expr(persons_table, "pe")
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TEMPORARY TABLE persons AS
                    SELECT {person_columns}
                    FROM {persons_identifier} pe
                    """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                )
                return
            self.conn.execute(
                f"CREATE OR REPLACE TEMPORARY TABLE persons AS "  # noqa: S608
                f"SELECT * FROM {persons_identifier}"
            )
            return

        if self._use_minimal_person_input_columns():
            person_columns = self._minimal_person_columns_expr(persons_table, "pe", exclude={"rank"})
        else:
            person_columns = self._select_columns_expr(persons_table, "pe", exclude={"rank"})
        where_clause = ""
        params: list[int] = [self._partition_default_rank()]
        if imputation is not None:
            where_clause = "WHERE pe.Imputation = ?"
            params.append(int(imputation))
        params.append(self.rank)

        household_join = ""
        rank_place_expr = "pe.sp_hh_id"
        if self._households_table_name() is not None:
            household_join = "LEFT JOIN household_ranks hh ON hh.household_id = pe.sp_hh_id"
            rank_place_expr = "COALESCE(hh.place_id, pe.sp_hh_id)"

        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE persons AS
            SELECT *
            FROM (
                SELECT
                    {person_columns},
                    CAST(COALESCE(place_ranks.rank, ?) AS INTEGER) AS rank
                FROM {persons_identifier} pe
                {household_join}
                LEFT JOIN place_ranks
                       ON place_ranks.sp_id = {rank_place_expr}
                {where_clause}
            ) ranked_persons
            WHERE rank = ?
            """,  # noqa: S608 - table identifiers are validated by quote_table_identifier.
            params,
        )

    def _household_columns_expr(
        self, households_table: str, alias: str, exclude: set[str] | None = None
    ) -> tuple[str, str]:
        """Build household input columns plus the place-id expression used for rank joins."""
        households_identifier = quote_table_identifier(households_table)
        available_columns = [
            column[0] for column in self.conn.execute(f"SELECT * FROM {households_identifier} LIMIT 0").description
        ]
        available_column_set = set(available_columns)
        if "household_id" not in available_column_set and "sp_id" not in available_column_set:
            raise MissingRequiredTableError([f"{households_table}.household_id", f"{households_table}.sp_id"])
        if not {"place_id", "sp_home_id", "sp_id"} & available_column_set:
            raise MissingRequiredTableError(
                [f"{households_table}.place_id", f"{households_table}.sp_home_id", f"{households_table}.sp_id"]
            )

        household_columns = self._select_columns_expr(households_table, alias, exclude=exclude)
        derived_columns: list[str] = []
        sp_id_expr = f"{alias}.{self._quote_column_identifier('sp_id')}" if "sp_id" in available_column_set else None
        if "household_id" not in available_column_set:
            if sp_id_expr is None:
                raise MissingRequiredTableError([f"{households_table}.household_id", f"{households_table}.sp_id"])
            derived_columns.append(f"{sp_id_expr} AS {self._quote_column_identifier('household_id')}")
        if "place_id" in available_column_set:
            place_expr = f"{alias}.{self._quote_column_identifier('place_id')}"
        elif "sp_home_id" in available_column_set:
            place_expr = f"{alias}.{self._quote_column_identifier('sp_home_id')}"
            derived_columns.append(f"{place_expr} AS {self._quote_column_identifier('place_id')}")
        else:
            if sp_id_expr is None:
                raise MissingRequiredTableError(
                    [f"{households_table}.place_id", f"{households_table}.sp_home_id", f"{households_table}.sp_id"]
                )
            place_expr = sp_id_expr
            derived_columns.append(f"{sp_id_expr} AS {self._quote_column_identifier('place_id')}")
        if derived_columns:
            household_columns = f"{household_columns}, {', '.join(derived_columns)}"
        return household_columns, place_expr

    def _create_households_input_table(
        self,
        households_table: str | None,
        imputation: int | None,
        partition_table: str | None,
    ) -> None:
        """Create the optional temporary households table."""
        if households_table is None:
            return
        if not check_if_table_exists(self.conn, households_table):
            raise MissingRequiredTableError(households_table)

        households_identifier = quote_table_identifier(households_table)
        household_columns, place_expr = self._household_columns_expr(households_table, "h", exclude={"rank"})
        if partition_table is not None:
            params: list[int] = []
            if imputation is not None:
                imputation_clause = "WHERE h.Imputation = ?"
                params.append(int(imputation))
            else:
                imputation_clause = ""
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMPORARY TABLE household_ranks AS
                SELECT
                    {household_columns},
                    place_ranks.rank
                    FROM {households_identifier} h
                    INNER JOIN place_ranks
                            ON place_ranks.sp_id = {place_expr}
                {imputation_clause}
                """,  # noqa: S608 - table identifiers are validated by quote_table_identifier.
                params,
            )
            self.conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE households AS
                SELECT *
                FROM household_ranks
                WHERE rank = ?
                """,
                [self.rank],
            )
            return

        if imputation is not None:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMPORARY TABLE households AS
                SELECT {household_columns}
                FROM {households_identifier} h
                WHERE Imputation = ?
                """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                [int(imputation)],
            )
            return

        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE households AS
            SELECT {household_columns}
            FROM {households_identifier} h
            """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
        )

    def _create_activities_input_table(
        self, activities_table: str, imputation: int | None, partition_table: str | None
    ) -> None:
        """Create the temporary activities table, optionally limited to local people."""
        activities_identifier = quote_table_identifier(activities_table)
        if partition_table is not None:
            logger.info(f"Using activities table {activities_table} for local rank persons...")
            activity_columns = self._activity_columns_expr(activities_table, "a")
            params: list[int] = []
            if imputation is not None:
                imputation_clause = "a.Imputation = ? AND"
                params.append(int(imputation))
            else:
                imputation_clause = ""
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMPORARY TABLE activities AS
                SELECT {activity_columns}
                FROM {activities_identifier} a
                WHERE {imputation_clause} EXISTS (
                    SELECT 1
                    FROM persons pe
                    WHERE pe.sp_id = a.sp_persons_id
                )
                """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                params,
            )
            return

        if imputation is not None:
            logger.info(f"Using imputation {imputation} for activities " f"table {activities_table}...")
            activity_columns = self._activity_columns_expr(activities_table, "a")
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMPORARY TABLE activities AS
                SELECT {activity_columns}
                FROM {activities_identifier} a
                WHERE a.Imputation = ?;
                """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
                [int(imputation)],
            )
            return

        logger.info(f"Using activities table {activities_table}...")
        activity_columns = self._activity_columns_expr(activities_table, "a")
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE activities AS
            SELECT {activity_columns}
            FROM {activities_identifier} a
            """,  # noqa: S608 - table identifier is validated by quote_table_identifier.
        )

    def _resolve_required_input_tables(self, extra_table_names: tuple[str, ...] = ()) -> dict[str, str]:
        """Resolve and validate required source tables with one DuckDB catalog query."""
        table_params = ("places.table", "persons.table", "activities.table")
        table_specs: list[tuple[str, str, str, str]] = []

        def add_table_spec(param_name: str, table_name: str | None) -> None:
            if table_name is None:
                raise MissingRequiredTableError(param_name)
            table_name = str(table_name)
            table_name_parts = table_name.split(".")
            if len(table_name_parts) > 2:
                raise MissingRequiredTableError(table_name)

            schema_name = "main"
            unqualified_table_name = table_name
            if len(table_name_parts) == 2:
                schema_name, unqualified_table_name = table_name_parts
            table_specs.append((param_name, table_name, schema_name, unqualified_table_name))

        for param_name in table_params:
            add_table_spec(param_name, self.params.get(param_name))
        for table_name in extra_table_names:
            add_table_spec(table_name, table_name)

        values_sql = ", ".join(["(?, ?)"] * len(table_specs))
        params = [
            value
            for _, _, schema_name, unqualified_table_name in table_specs
            for value in (schema_name, unqualified_table_name)
        ]
        existing_tables = set(
            self.conn.execute(
                f"""
                WITH required(schema_name, table_name) AS (
                    VALUES {values_sql}
                )
                SELECT required.schema_name, required.table_name
                FROM required
                INNER JOIN duckdb_tables
                        ON duckdb_tables.schema_name = required.schema_name
                       AND duckdb_tables.table_name = required.table_name
                """,  # noqa: S608 - values_sql is built from placeholders only.
                params,
            ).fetchall()
        )
        missing_tables = [
            table_name
            for _, table_name, schema_name, unqualified_table_name in table_specs
            if (schema_name, unqualified_table_name) not in existing_tables
        ]
        if missing_tables:
            raise MissingRequiredTableError(missing_tables)

        return {param_name: table_name for param_name, table_name, _, _ in table_specs}

    def create_input_tables(self) -> None:
        """Load tables from the database."""

        imputation = self.params.get("Imputation", None) if "Imputation" in self.params else None
        partition_table = self._partition_table_name()
        input_tables = self._time_phase(
            "startup.create_input_tables.check_required_tables",
            self._resolve_required_input_tables,
            (partition_table,) if partition_table is not None else (),
        )

        #  create the places table as a view from the ducklake table
        places_table = input_tables["places.table"]
        logger.info(f"creating <places> view from <{places_table}>...")
        self._time_phase(
            "startup.create_input_tables.create_places_table",
            self._create_places_input_table,
            places_table,
            partition_table,
            partition_table is not None,
        )

        households_table = self._households_table_name()
        self._time_phase(
            "startup.create_input_tables.create_households_table",
            self._create_households_input_table,
            households_table,
            imputation,
            partition_table,
        )

        # create the persons table
        persons_table = input_tables["persons.table"]
        logger.info(f"creating <persons> view from <{persons_table}>...")
        self._time_phase(
            "startup.create_input_tables.create_persons_table",
            self._create_persons_input_table,
            persons_table,
            imputation,
            partition_table,
        )

        # create the activities table
        activities_table = input_tables["activities.table"]
        self._time_phase(
            "startup.create_input_tables.create_activities_table",
            self._create_activities_input_table,
            activities_table,
            imputation,
            partition_table,
        )

        self.queries = {
            "get_tables": "SHOW TABLES",
            "add_geometries": """
                -- 1. Load the spatial extension
                -- This is necessary to use ST_Point and other geospatial
                -- functions.
                INSTALL spatial;
                LOAD spatial;
                -- 2. Add the 'location' column of type GEOMETRY
                -- GEOMETRY is a generic spatial type that can store points,
                --lines, polygons, etc.
                ALTER TABLE places ADD COLUMN location GEOMETRY;
                -- 3. Populate the 'location' column
                -- ST_Point expects (X, Y) which translates to (longitude,
                -- latitude) for geographic points.
                UPDATE places
                -- Ensure that longitude and latitude are in the correct order
                -- for ST_Point
                SET location = ST_Point(longitude, latitude);
                """,
        }

        # create the contacts table only when contact behavior is enabled.
        self._time_phase("startup.create_input_tables.contacts", self._create_contacts_input_table, imputation)

    def _create_contacts_input_table(self, imputation: int | None) -> None:
        """Create the optional contacts table used by contact-aware behavior."""
        if self._contacts_enabled():
            contacts_table = self._contacts_table_name()
            if contacts_table is None:
                raise MissingRequiredParameterError("contacts.table")
            if not check_if_table_exists(self.conn, contacts_table):
                logger.error(f"Error: contacts table {contacts_table} " "does not exist in the database.")
                raise MissingRequiredTableError(contacts_table)
            if imputation is not None:
                logger.info(f"Using imputation {imputation} for " f"contacts table {contacts_table}...")
                contacts_identifier = quote_table_identifier(contacts_table)
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMPORARY TABLE contacts AS
                    SELECT * FROM {contacts_identifier}
                    WHERE Imputation = {imputation};
                    """)  # noqa: S608
            else:
                logger.info(f"Using contacts table {contacts_table}...")
                contacts_identifier = quote_table_identifier(contacts_table)
                self.conn.execute(
                    f"CREATE OR REPLACE TEMPORARY TABLE contacts AS "  # noqa: S608
                    f"SELECT * FROM {contacts_identifier}"
                )
        elif self._contacts_table_name():
            logger.info("contacts.enabled is false; skipping contacts table materialization.")

    def _household_row_place_id(self, row: dict) -> int | None:
        """Resolve the physical place id from a household row."""
        place_id = row.get("place_id")
        if place_id is None:
            place_id = row.get("sp_home_id")
        if place_id is None:
            place_id = row.get("sp_id")
        if place_id is None:
            place_id = row.get("sp_hh_id", row.get("home_place_id"))
        if place_id is None:
            place_id = row.get("household_id", row.get("sp_id", row.get("hh_id")))
        return int(place_id) if place_id is not None else None

    def create_households(self) -> None:
        """Create optional household social-unit agents and link them to places."""
        self.households_by_id = {}
        self.households_by_place_id = {}
        if self._households_table_name() is None:
            return

        try:
            table = self.conn.execute("SELECT * FROM households").arrow().read_all()
        except Exception as exc:
            raise MissingRequiredTableError("households") from exc

        household_type = self.getHouseholdClass()
        household_data_type = household_type.getHouseholdDataClass()
        context_add = self.context.add
        lookup_place = self.places_proj.lookup_place

        for row in table.to_pylist():
            place_id = self._household_row_place_id(row)
            if place_id is None:
                logger.error("Skipping household row without a place id: {}", row)
                continue

            place = lookup_place(place_id)
            if place is None:
                logger.error("Skipping household row for missing place {}", place_id)
                continue
            if int(place.rank) != self.rank:
                continue

            row.setdefault("rank", place.rank)
            household = household_type(row, household_data_type)
            household.setPlace(place)
            context_add(household)
            self.households_by_id[int(household.id)] = household
            self.households_by_place_id[int(household.place_id)] = household

    def _fast_context_person_adder(self):
        """Return a person-only context adder when SharedContext internals are safe to use."""
        storage = self._fast_context_person_storage()
        if storage is None:
            return self.context.add

        local_agents, person_agents, context_rank = storage

        def add_person(person):
            person.local_rank = context_rank
            uid = person.uid
            local_agents[uid] = person
            person_agents[uid] = person

        return add_person

    def _fast_context_person_storage(self):
        """Return direct person indexes when the SharedContext shape matches the startup fast path."""
        projections = getattr(self.context, "projections", None)
        if not isinstance(projections, dict):
            return None
        if len(projections) != 1 or next(iter(projections.values())) is not self.places_proj:
            return None

        agent_manager = getattr(self.context, "_agent_manager", None)
        local_agents = getattr(agent_manager, "_local_agents", None)
        agents_by_type = getattr(self.context, "_agents_by_type", None)
        if not isinstance(local_agents, dict) or not isinstance(agents_by_type, dict):
            return None

        person_agents = agents_by_type.setdefault(Person.TYPE, OrderedDict())
        context_rank = getattr(agent_manager, "rank", None)
        if context_rank is None:
            context_rank = getattr(self.context, "rank", getattr(self, "rank", 0))
        return local_agents, person_agents, context_rank

    def _resolve_person_household_place(self, household_id, places):
        """Resolve a person's household id to a household agent and physical place id."""
        if household_id is None:
            return None, household_id, places
        household_key = int(household_id)
        household = getattr(self, "households_by_id", {}).get(household_key)
        if household is None:
            household = getattr(self, "households_by_place_id", {}).get(household_key)
        if household is None:
            return None, household_id, places

        place_id = household.place_id
        if place_id == household_id:
            return household, place_id, places
        if isinstance(places, tuple):
            return household, place_id, (place_id, *places[1:])
        if isinstance(places, list):
            return household, place_id, [place_id, *places[1:]]
        return household, place_id, places

    def create_persons(
        self,
        rng: Generator,
    ) -> None:
        """Create persons from the given file.

        Args:
            rng: The random number generator.
        """
        # get the person type
        personType = self.getPersonClass()

        # Create the activities
        #  - schedulesList is a list of dict of personID->list[Act]
        schedulesList = self._time_phase("startup.create_persons.create_activities", self.create_activities)
        if not schedulesList or len(schedulesList) == 0:
            logger.error("Error: no activities found.")
            raise MissingRequiredParameterError("activities.file")
        logger.info(f"rank {self.rank}: weekday activitiesMap " f"size={len(schedulesList[0])}")

        # get the weekday activity map. Currently, we assume there is only
        # one source activity set in the list and derive additional plans
        # from it locally.
        activitiesMap = schedulesList[0] if schedulesList else {}

        # get the activities data type: namedtuple to store places
        # for activities
        activitiesDataType = self.get_activities_data_type()

        # get the planned_activity_names, which are the fields in
        # the person file that contain the place ids
        # (e.g. 'sp_work_id', 'sp_school_id', etc.)
        planned_activity_names = self.get_planned_activity_names()

        # get the activity names
        # (list should be as long as planned_activity_names)
        activity_names = self.get_activity_names()

        # get alternate activity names
        # (activities not in planned activities)
        alternate_activities_names = activity_names[len(planned_activity_names) :]
        use_minimal_person_payload = personType is Person and personType.getPersonDataClass() is PersonData

        # load the persons from the file
        # table = pq.read_table(personsFile)
        person_query = "SELECT * FROM persons"
        if use_minimal_person_payload:
            required_person_columns = list(dict.fromkeys(["sp_id", *planned_activity_names]))
            optional_person_columns = ["x", "y", "network", "minute_last_moved"]
            available_person_columns = {
                column[0] for column in self.conn.execute("SELECT * FROM persons LIMIT 0").description
            }
            selected_person_columns = required_person_columns + [
                column
                for column in optional_person_columns
                if column in available_person_columns and column not in required_person_columns
            ]
            selected_person_columns_sql = ", ".join(
                quote_table_identifier(column) for column in selected_person_columns
            )
            person_query = f"SELECT {selected_person_columns_sql} FROM persons"  # noqa: S608

        def load_person_rows():
            cursor = self.conn.execute(person_query)
            description = cursor.description
            return (
                [column[0] for column in description],
                {column[0]: str(column[1]).upper() for column in description},
                cursor.fetchall(),
            )

        person_column_names, person_column_types, person_rows = self._time_phase(
            "startup.create_persons.load_person_rows",
            load_person_rows,
        )

        construct_start = time.perf_counter()
        try:
            person_column_indexes = {column_name: index for index, column_name in enumerate(person_column_names)}
            person_id_index = person_column_indexes["sp_id"]
            planned_place_indexes = [person_column_indexes[name] for name in planned_activity_names]
            use_native_place_columns = all(
                "INT" in person_column_types.get(name, "") or person_column_types.get(name, "") == "NULL"
                for name in planned_activity_names
            )
            use_tuple_place_values = use_minimal_person_payload and not alternate_activities_names
            planned_place_index_count = len(planned_place_indexes)
            if planned_place_index_count == 3:
                place_index_0, place_index_1, place_index_2 = planned_place_indexes
            else:
                place_index_0 = place_index_1 = place_index_2 = None
            x_index = person_column_indexes.get("x")
            y_index = person_column_indexes.get("y")
            network_index = person_column_indexes.get("network")
            minute_last_moved_index = person_column_indexes.get("minute_last_moved")
            needs_custom_payload = (
                x_index is not None
                or y_index is not None
                or network_index is not None
                or minute_last_moved_index is not None
            )
            place_lookup_map = getattr(self.places_proj, "_places", None)
            lookup_place = place_lookup_map.get if isinstance(place_lookup_map, dict) else self.places_proj.lookup_place
            projection_add = self.places_proj.add
            assign_agent_to_place = self.places_proj.assign_agent_to_place
            assign_new_agent_to_place = getattr(self.places_proj, "assign_new_agent_to_place", None)
            context_add = self.context.add
            build_primary_plan = self._build_primary_plan
            make_default_person = Person.from_default_fields if use_minimal_person_payload else None
            make_zero_location_person = (
                Person.from_default_zero_location if use_minimal_person_payload and not needs_custom_payload else None
            )
            make_schedule_person = (
                Person.from_default_schedule_zero_location
                if use_minimal_person_payload and not needs_custom_payload
                else None
            )
            behavior_engine_type = Person.getBehaviorEngine() if use_minimal_person_payload else None
            initialize_person_communication_state = (
                self._communication_enabled() or behavior_engine_type is not ScheduleBehaviorEngine
            )
            use_unrouted_plan = self.road_network is None
            convert_place_id = convert_to_int
            alternate_places = [None] * len(alternate_activities_names)
            empty_plan_count = 1 + len(alternate_activities_names)
            timing_enabled = self._phase_timings_enabled()
            construct_detail_sample_stride = self._phase_timing_sample_stride()
            collect_construct_detail_timing = timing_enabled and construct_detail_sample_stride > 0
            sampled_payload_seconds = 0.0
            sampled_place_resolution_seconds = 0.0
            sampled_plan_seconds = 0.0
            plan_lookup_seconds = 0.0
            primary_plan_seconds = 0.0
            empty_plan_seconds = 0.0
            places_tuple_seconds = 0.0
            construct_detail_sample_count = 0
            construct_detail_candidate_count = 0
            sampled_person_init_seconds = 0.0
            sampled_projection_seconds = 0.0

            use_default_schedule_fast_path = (
                use_minimal_person_payload
                and make_zero_location_person is not None
                and make_schedule_person is not None
                and behavior_engine_type is ScheduleBehaviorEngine
                and not initialize_person_communication_state
                and use_unrouted_plan
                and empty_plan_count == 1
                and use_native_place_columns
                and use_tuple_place_values
                and planned_place_index_count == 3
                and not needs_custom_payload
                and alternate_places == []
                and assign_new_agent_to_place is not None
                and isinstance(place_lookup_map, dict)
            )
            if use_default_schedule_fast_path:
                fast_context_storage = self._fast_context_person_storage()
                if fast_context_storage is None:
                    fast_context_add = self.context.add
                    fast_local_agents = fast_person_agents = None
                    fast_context_rank = None
                else:
                    fast_local_agents, fast_person_agents, fast_context_rank = fast_context_storage
                    fast_context_add = None
                fast_agent_locations = getattr(self.places_proj, "_agent_locations", None)
                use_fast_projection_assignment = isinstance(fast_agent_locations, dict)
                if collect_construct_detail_timing:
                    construct_detail_sample_count = 1
                for row in person_rows:
                    personID = row[person_id_index]
                    place_0 = row[place_index_0]
                    place_1 = row[place_index_1]
                    place_2 = row[place_index_2]
                    places = (
                        place_0 if place_0 else None,
                        place_1 if place_1 else None,
                        place_2 if place_2 else None,
                    )
                    household_agent, home_place_id, places = self._resolve_person_household_place(place_0, places)

                    household_place = lookup_place(home_place_id)
                    if not household_place:
                        logger.error(f"Error: No household place found for person {personID} at place {home_place_id}.")
                        continue

                    rank = household_place.rank
                    if rank != self.rank:
                        logger.error(f"Error: Person {personID} tagged on " f"rank={rank} is not on this rank.")
                        continue

                    schedule = activitiesMap.get(personID)
                    if schedule is None:
                        logger.error(f"Error: No activities found for " f"person {personID}.")
                        continue

                    person = make_schedule_person(
                        personID,
                        rank,
                        schedule,
                        places,
                    )
                    if fast_context_add is None:
                        person.local_rank = fast_context_rank
                        uid = person.uid
                        fast_local_agents[uid] = person
                        fast_person_agents[uid] = person
                    else:
                        fast_context_add(person)

                    if use_fast_projection_assignment:
                        fast_agent_locations[person.id] = household_place.id
                        household_place.occupants.add(person)
                    else:
                        assign_new_agent_to_place(person, household_place)
                    if household_agent is not None:
                        household_agent.addMember(person)
                return

            for row in person_rows:
                personID = row[person_id_index]
                collect_this_construct_detail = False
                if collect_construct_detail_timing:
                    collect_this_construct_detail = (
                        construct_detail_candidate_count % construct_detail_sample_stride == 0
                    )
                    construct_detail_candidate_count += 1
                    if collect_this_construct_detail:
                        construct_detail_sample_count += 1

                phase_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if use_minimal_person_payload:
                    if needs_custom_payload:
                        p_x = row[x_index] if x_index is not None else 0
                        p_y = row[y_index] if y_index is not None else 0
                        p_network = row[network_index] if network_index is not None else None
                        p_minute_last_moved = row[minute_last_moved_index] if minute_last_moved_index is not None else 0
                    else:
                        p_x = 0
                        p_y = 0
                        p_network = None
                        p_minute_last_moved = 0
                else:
                    p = {column_name: row[column_index] for column_name, column_index in person_column_indexes.items()}
                if collect_this_construct_detail:
                    sampled_payload_seconds += time.perf_counter() - phase_start

                # TODO: add tests for this
                #  - activities_data = [p[x] for x in
                #    planned_activity_names]
                #  - all places should be in placeMap
                #  - the first place is a household
                #  - handle person not on this rank?
                #  - handle person not in activitiesMap?
                phase_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if use_native_place_columns:
                    if use_tuple_place_values and planned_place_index_count == 3:
                        place_0 = row[place_index_0]
                        place_1 = row[place_index_1]
                        place_2 = row[place_index_2]
                        places = (
                            place_0 if place_0 else None,
                            place_1 if place_1 else None,
                            place_2 if place_2 else None,
                        )
                    elif use_tuple_place_values:
                        places = tuple(
                            row[place_index] if row[place_index] else None for place_index in planned_place_indexes
                        )
                    else:
                        places = [
                            row[place_index] if row[place_index] else None for place_index in planned_place_indexes
                        ]
                else:
                    places = [convert_place_id(row[place_index]) for place_index in planned_place_indexes]

                    for place in places:
                        if isinstance(place, str):
                            logger.error(f"Error: Place {place} not found.")
                            return

                hhId = places[0]  # p['sp_hh_id']
                household_agent, hhId, places = self._resolve_person_household_place(hhId, places)

                household_place = lookup_place(hhId)
                if not household_place:
                    logger.error(f"Error: No household place found for person {personID} at place {hhId}.")
                    continue

                rank = household_place.rank

                if rank != self.rank:
                    message = f"Person {personID} tagged on rank={rank} is not on this rank."
                    if self._partition_table_name() is None and self.size > 1:
                        logger.debug(message)
                    else:
                        logger.error(f"Error: {message}")
                    continue

                if collect_this_construct_detail:
                    sampled_place_resolution_seconds += time.perf_counter() - phase_start

                # Person
                #  - activitiesMap: plansList[0] is a dict of
                #    personID->list[Act]

                phase_start = time.perf_counter() if collect_this_construct_detail else 0.0

                detail_start = time.perf_counter() if collect_this_construct_detail else 0.0
                schedule = activitiesMap.get(personID)
                if collect_this_construct_detail:
                    plan_lookup_seconds += time.perf_counter() - detail_start
                if schedule is None:
                    logger.error(f"Error: No activities found for " f"person {personID}.")
                    continue

                # get the schedule for the person
                detail_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if use_unrouted_plan and empty_plan_count == 1:
                    plans: list[Plan] = [schedule, []]
                elif use_unrouted_plan:
                    plans: list[Plan] = [schedule]
                else:
                    plans: list[Plan] = [build_primary_plan(personID, schedule, places)]
                if collect_this_construct_detail:
                    primary_plan_seconds += time.perf_counter() - detail_start

                # create an empty plan for weekend activities (if not already
                # present), plus one empty plan per alternate activity.
                detail_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if empty_plan_count == 1:
                    if not use_unrouted_plan:
                        plans.append([])
                else:
                    plans.extend([] for _ in range(empty_plan_count))
                if collect_this_construct_detail:
                    empty_plan_seconds += time.perf_counter() - detail_start

                # add alternate activities to the person's places
                if alternate_places:
                    places = places + alternate_places

                # Default PersonData only needs an indexable sequence; custom
                # data classes retain the namedtuple representation.
                detail_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if use_minimal_person_payload:
                    if not use_tuple_place_values:
                        places = tuple(places)
                else:
                    places = activitiesDataType(*places)
                if collect_this_construct_detail:
                    places_tuple_seconds += time.perf_counter() - detail_start
                if collect_this_construct_detail:
                    sampled_plan_seconds += time.perf_counter() - phase_start

                phase_start = time.perf_counter() if collect_this_construct_detail else 0.0
                if make_zero_location_person is not None:
                    person = make_zero_location_person(
                        personID,
                        rank,
                        plans,
                        places,
                        behavior_engine_type,
                        initialize_person_communication_state,
                    )
                elif make_default_person is not None:
                    person = make_default_person(
                        personID,
                        rank,
                        plans,
                        places,
                        x=p_x,
                        y=p_y,
                        minute_last_moved=p_minute_last_moved,
                        network=p_network,
                        behavior_engine=behavior_engine_type,
                        initialize_communication_state=initialize_person_communication_state,
                    )
                else:
                    person = personType(
                        personID,
                        rank,
                        plans,
                        places,
                        p,  # initDict for additional data
                    )
                if collect_this_construct_detail:
                    sampled_person_init_seconds += time.perf_counter() - phase_start

                phase_start = time.perf_counter() if collect_this_construct_detail else 0.0
                context_add(person)
                if assign_new_agent_to_place is not None:
                    assign_new_agent_to_place(person, household_place)
                else:
                    projection_add(person)
                    assign_agent_to_place(person, household_place)
                if household_agent is not None:
                    household_agent.addMember(person)
                if collect_this_construct_detail:
                    sampled_projection_seconds += time.perf_counter() - phase_start
        finally:
            self._record_phase_timing("startup.create_persons.construct_agents", time.perf_counter() - construct_start)
            if self._phase_timings_enabled() and construct_detail_sample_count:
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.sampled_total.build_payload",
                    sampled_payload_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.sampled_total.resolve_places",
                    sampled_place_resolution_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.sampled_total.build_plans",
                    sampled_plan_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.build_plans.sampled_total.lookup_schedule",
                    plan_lookup_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.build_plans.sampled_total.primary_plan",
                    primary_plan_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.build_plans.sampled_total.empty_plans",
                    empty_plan_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.build_plans.sampled_total.places_tuple",
                    places_tuple_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.sampled_total.init_person",
                    sampled_person_init_seconds,
                )
                self._record_phase_timing(
                    "startup.create_persons.construct_agents.sampled_total.attach_person",
                    sampled_projection_seconds,
                )

    def _iter_plan_triples(self, plan: Plan):
        """Yield (activity, leg, activity) triples from a plan."""
        for index, element in enumerate(plan):
            if not isinstance(element, Leg):
                continue

            prev_act = plan[index - 1] if index > 0 and isinstance(plan[index - 1], Act) else None
            next_act = plan[index + 1] if index + 1 < len(plan) and isinstance(plan[index + 1], Act) else None
            if prev_act is not None and next_act is not None:
                yield prev_act, element, next_act

    def _log_invalid_legs(self, person_id: int, plan: Plan) -> None:
        """Log any routed leg whose travel time exceeds the scheduled gap."""
        for prev_act, leg, next_act in self._iter_plan_triples(plan):
            if validate_leg_against_schedule(prev_act, leg, next_act):
                continue

            gap_min = next_act.starttime_min - prev_act.endtime_min
            logger.warning(
                f"Person {person_id} has infeasible leg from place "
                f"{leg.origin_place_id} to {leg.destination_place_id}: "
                f"travel_time_min={leg.travel_time_min}, gap_min={gap_min}"
            )

    def _build_primary_plan(self, person_id: int, schedule: list[Act], places: list[int]) -> Plan:
        """Build the main weekday plan for a person, using routing when available."""
        if self.road_network is None:
            return make_plan(schedule)

        routed_plan = make_routed_plan(
            schedule,
            places,
            self.road_network,
            self.params.get("roads.mode", "drive"),
        )
        self._log_invalid_legs(person_id, routed_plan)
        return routed_plan

    def _load_place_rank_map(self) -> dict[int, int]:
        """Load place rank ownership from DuckDB into a Python lookup map."""
        rank_columns = self.conn.execute("SELECT sp_id, rank FROM place_ranks").fetchnumpy()
        return dict(zip(rank_columns["sp_id"].tolist(), rank_columns["rank"].tolist()))

    def create_places(self) -> None:
        """Create places in the project."""

        logger.info("Creating places...")

        # Get the place type and data type
        placeType = self.getPlaceClass()
        placeDataType = placeType.getPlaceDataClass()
        use_minimal_place_payload = placeType is Place and placeDataType is PlaceData
        place_ranks_loaded_into_projection = False

        if self._partition_table_name() is not None:
            self.place_to_rank = self._time_phase(
                "startup.create_places.load_place_rank_map",
                self._load_place_rank_map,
            )
            set_place_rank_map = getattr(self.places_proj, "set_place_rank_map", None)
            if set_place_rank_map is not None:
                try:
                    set_place_rank_map(self.place_to_rank, copy=False)
                except TypeError:
                    set_place_rank_map(self.place_to_rank)
                place_ranks_loaded_into_projection = True

        place_query = "SELECT * FROM places"
        selected_place_columns: list[str] = []
        if use_minimal_place_payload:
            required_place_columns = ["sp_id"]
            optional_place_columns = ["rank", "place_type", "place_name", "latitude", "longitude"]
            available_place_columns = {
                column[0] for column in self.conn.execute("SELECT * FROM places LIMIT 0").description
            }
            selected_place_columns = required_place_columns + [
                column
                for column in optional_place_columns
                if column in available_place_columns and column not in required_place_columns
            ]
            selected_place_columns_sql = ", ".join(quote_table_identifier(column) for column in selected_place_columns)
            place_query = f"SELECT {selected_place_columns_sql} FROM places"  # noqa: S608

        if use_minimal_place_payload:

            def load_place_columns():
                columns = self.conn.execute(place_query).fetchnumpy()
                return [columns[column].tolist() for column in selected_place_columns]

            place_value_lists = self._time_phase(
                "startup.create_places.load_place_rows",
                load_place_columns,
            )
        else:
            table = self._time_phase(
                "startup.create_places.load_place_rows",
                lambda: self.conn.execute(place_query).arrow().read_all(),
            )

        construct_start = time.perf_counter()
        try:
            add_place = self.places_proj.add_place
            add_places = getattr(self.places_proj, "add_places", None)
            if use_minimal_place_payload:
                make_place = placeType.from_default_fields
                place_column_indexes = {column_name: index for index, column_name in enumerate(selected_place_columns)}
                place_id_index = place_column_indexes["sp_id"]
                rank_index = place_column_indexes.get("rank")
                place_type_index = place_column_indexes.get("place_type")
                place_name_index = place_column_indexes.get("place_name")
                latitude_index = place_column_indexes.get("latitude")
                longitude_index = place_column_indexes.get("longitude")
                place_ids = place_value_lists[place_id_index]
                ranks = place_value_lists[rank_index] if rank_index is not None else None
                place_types = place_value_lists[place_type_index] if place_type_index is not None else None
                place_names = place_value_lists[place_name_index] if place_name_index is not None else None
                latitudes = place_value_lists[latitude_index] if latitude_index is not None else None
                longitudes = place_value_lists[longitude_index] if longitude_index is not None else None

                places = []
                append_place = places.append
                if (
                    rank_index is not None
                    and place_type_index is not None
                    and place_name_index is None
                    and latitude_index is not None
                    and longitude_index is not None
                ):
                    for place_id, rank, place_type, latitude, longitude in zip(
                        place_ids,
                        ranks,
                        place_types,
                        latitudes,
                        longitudes,
                    ):
                        append_place(
                            make_place(
                                place_id,
                                rank,
                                place_type,
                                "",
                                latitude,
                                longitude,
                            )
                        )
                else:
                    for row_index, place_id in enumerate(place_ids):
                        append_place(
                            make_place(
                                place_id,
                                ranks[row_index] if ranks is not None else 0,
                                place_types[row_index] if place_types is not None else "Household",
                                place_names[row_index] if place_names is not None else "",
                                latitudes[row_index] if latitudes is not None else float("nan"),
                                longitudes[row_index] if longitudes is not None else float("nan"),
                            )
                        )

                if add_places is not None:
                    add_places(places, register_ranks=not place_ranks_loaded_into_projection)
                else:
                    for place in places:
                        add_place(place)
                return

            for batch in table.to_batches():
                data = batch.to_pydict()
                column_names = list(data.keys())
                column_values = [data[column_name] for column_name in column_names]
                row_count = batch.num_rows

                for row_index in range(row_count):
                    place_record = {
                        column_name: values[row_index] for column_name, values in zip(column_names, column_values)
                    }
                    if "rank" not in place_record:
                        place_record["rank"] = 0
                    place = placeType(place_record, placeDataType)
                    add_place(place)
        finally:
            self._record_phase_timing("startup.create_places.construct_places", time.perf_counter() - construct_start)

    def create_activities(self) -> list[dict[int, Plan]]:
        """Create activities from the given file.

        This method reads the activities file and creates a mapping of
        person IDs to their weekday activities. When no road network is active,
        shared default travel legs are inserted while rows are read. Routed runs
        keep activity-only schedules so plan construction can populate route
        details later. Additional empty plans can be added later for weekends
        or unplanned behavior.

        Returns:
            list[dict[int, Plan]]: A list containing a single
            dictionary mapping person IDs to their weekday activities.
        """
        activity_columns = [
            "sp_persons_id",
            "activity_id",
            "activity_sequence",
            "starttime_min",
            "endtime_min",
            "sp_act_id",
        ]
        activity_column_sql = ", ".join(quote_table_identifier(column) for column in activity_columns)
        activity_order_sql = ", ".join(
            quote_table_identifier(column)
            for column in [
                "sp_persons_id",
                "activity_sequence",
            ]
        )

        def load_activity_rows():
            columns = self.conn.execute(
                f"SELECT {activity_column_sql} FROM activities ORDER BY {activity_order_sql}"  # noqa: S608
            ).fetchnumpy()
            return list(zip(*(columns[column].tolist() for column in activity_columns)))

        rows = self._time_phase(
            "startup.create_persons.create_activities.load_activity_rows",
            load_activity_rows,
        )

        # activitiesMap looks like:
        # personID -> list[Act]
        act_map: dict[int, Plan] = {}
        construct_start = time.perf_counter()
        try:
            act_new = Act.__new__
            default_travel_leg = DEFAULT_TRAVEL_LEG
            build_unrouted_plans = getattr(self, "road_network", None) is None
            missing_person_id = object()
            current_person_id = missing_person_id
            current_plan = None

            cast_values = bool(rows) and any(not isinstance(value, int) for value in rows[0])
            if cast_values:
                to_int = int
                if build_unrouted_plans:
                    for sp_persons_id, activity_id, activity_seq, start, end, act_place_id in rows:
                        person_id = to_int(sp_persons_id)
                        if person_id == current_person_id:
                            person_plan = current_plan
                            person_plan.append(default_travel_leg)
                        else:
                            current_person_id = person_id
                            person_plan = []
                            act_map[person_id] = person_plan
                            current_plan = person_plan
                        activity = act_new(Act)
                        activity.person_id = person_id
                        activity.activity_id = to_int(activity_id)
                        activity.activity_sequence = to_int(activity_seq)
                        activity.starttime_min = to_int(start)
                        activity.endtime_min = to_int(end)
                        activity.place_id = to_int(act_place_id)
                        person_plan.append(activity)
                else:
                    for sp_persons_id, activity_id, activity_seq, start, end, act_place_id in rows:
                        person_id = to_int(sp_persons_id)
                        if person_id == current_person_id:
                            person_plan = current_plan
                        else:
                            current_person_id = person_id
                            person_plan = []
                            act_map[person_id] = person_plan
                            current_plan = person_plan
                        activity = act_new(Act)
                        activity.person_id = person_id
                        activity.activity_id = to_int(activity_id)
                        activity.activity_sequence = to_int(activity_seq)
                        activity.starttime_min = to_int(start)
                        activity.endtime_min = to_int(end)
                        activity.place_id = to_int(act_place_id)
                        person_plan.append(activity)
            else:
                if build_unrouted_plans:
                    for sp_persons_id, activity_id, activity_seq, start, end, act_place_id in rows:
                        if sp_persons_id == current_person_id:
                            person_plan = current_plan
                            person_plan.append(default_travel_leg)
                        else:
                            current_person_id = sp_persons_id
                            person_plan = []
                            act_map[sp_persons_id] = person_plan
                            current_plan = person_plan
                        activity = act_new(Act)
                        activity.person_id = sp_persons_id
                        activity.activity_id = activity_id
                        activity.activity_sequence = activity_seq
                        activity.starttime_min = start
                        activity.endtime_min = end
                        activity.place_id = act_place_id
                        person_plan.append(activity)
                else:
                    for sp_persons_id, activity_id, activity_seq, start, end, act_place_id in rows:
                        if sp_persons_id == current_person_id:
                            person_plan = current_plan
                        else:
                            current_person_id = sp_persons_id
                            person_plan = []
                            act_map[sp_persons_id] = person_plan
                            current_plan = person_plan
                        activity = act_new(Act)
                        activity.person_id = sp_persons_id
                        activity.activity_id = activity_id
                        activity.activity_sequence = activity_seq
                        activity.starttime_min = start
                        activity.endtime_min = end
                        activity.place_id = act_place_id
                        person_plan.append(activity)
        finally:
            self._record_phase_timing(
                "startup.create_persons.create_activities.construct_activity_map",
                time.perf_counter() - construct_start,
            )

        return [act_map]

    def create_contacts(self) -> dict[int, dict[int, int]]:
        # contactMap looks like:
        # personID -> { hour_of_day -> [ otherPersonIDs ] }
        contactMap = {}

        # with open(contactFile, 'r', newline='') as f:
        #     contacts = DictReader(f)
        # table = pq.read_table(contactFile)
        table = self.conn.execute("SELECT * FROM contacts").arrow().read_all()

        for batch in table.to_batches():
            d = batch.to_pydict()
            for source, target, hour_of_the_day in zip(d["from_person"], d["to_person"], d["hour"]):
                if source not in contactMap:
                    contactMap[source] = {}

                if hour_of_the_day not in contactMap[source]:
                    contactMap[source][hour_of_the_day] = []

                contactMap[source][hour_of_the_day].append(target)

        return contactMap

    def step(self) -> None:
        """Step the model forward one time step."""

        self.cal.increment(self.time_step_minutes)

        # log the current step
        logger.info(
            "Step on "
            f"day {self.cal.day_of_year}, "
            f"hour {self.cal.hour_of_day}, "
            f"minute {self.cal.minute_of_day}"
        )

        # Automatic place updates are disabled - they caused
        # performance degradation

        # 2025-02-26 jcline: this is a hack to get the person_id_map
        # self.get_local_ids()

        self._time_phase("tick.environment_step", self.get_environment().step, self.context, self.cal)

        # sequence of actions
        # 1. sense physical environment
        # 2. sense social environment
        # 3. update state
        # 4. update beliefs
        # 5. communicate
        # 6. make decisions
        # 7. act on decisions

        self._time_phase("tick.communication", self.run_communication_phases)

        if self._person_step_enabled():
            # Process person agents (TYPE=0) with optimized sequential
            # processing. Parallel processing caused performance degradation
            # due to thread overhead exceeding benefits for lightweight
            # agent operations
            person_agents = list(_person_agents(self.context))

            if len(person_agents) > 0:
                agent_start_time = time.time()

                # Optimized sequential processing with minimal overhead
                for person in person_agents:
                    person.step(self.context, self.cal)

                agent_processing_time = time.time() - agent_start_time
                self._record_phase_timing("tick.person_step", agent_processing_time)

                # Log performance for large datasets
                if self.rank == 0 and len(person_agents) >= 1000:
                    agents_per_second = len(person_agents) / agent_processing_time if agent_processing_time > 0 else 0
                    logger.info(
                        f"Person agent processing: "
                        f"{len(person_agents):,} agents, "
                        f"{agent_processing_time:.2f}s, "
                        f"rate: {agents_per_second:,.0f} agents/sec"
                    )
        else:
            self._record_phase_timing("tick.person_step", 0.0)

        self._time_phase("tick.log_agents", self.log_agents)

        # for person in self.context.agents():
        #     person.count_colocations(self.cspace)

        # self.data_set.log(tick)
        # clear the meet log counts for the next tick
        # self.meet_log.max_meets = \
        #     self.meet_log.min_meets = self.meet_log.total_meets = 0

    def reset(self) -> None:
        for place in self.local_places:
            place.reset()

    def get_local_ids(self) -> None:
        for person in self.context.agents():
            if person.id not in self.person_id_map:
                self.person_id_map[person.id] = person.uid

    def add_people_to_places(self) -> None:
        for person in self.context.agents():
            logger.debug("Adding person {} to place {}", person.id, person.state.place_id)
            # if person.state.place_id not in self.place_map:
            #     logger.error(f"Person {person.id} has no place.")
            #     return
            # self.place_map[person.state.place_id].addPerson(person)

    def make_contacts(self, tick) -> None:
        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # logger.debug(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.state.place_id)
            if not contactIDs:
                # logger.debug(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.state.place_id}.")
                continue

            contacts = []
            for contactID in contactIDs:
                uid = self.person_id_map[contactID]
                contacts.append(self.context.agent(uid))
            person.make_contacts(contacts)

    def log_agents(self) -> None:
        """Log the agents at the current time step.

        This method can be used to log agent data at each step of the
        simulation.
        """

        # Notify observers of the step event
        self._notify_step_observers()

    def get_parallel_performance_stats(self) -> dict:
        """Get performance statistics from parallel place updates."""
        if hasattr(self.places_proj, "get_parallel_performance_stats"):
            return self.places_proj.get_parallel_performance_stats()
        return {}

    def at_end(self) -> None:
        """Actions to take at the end of the simulation."""
        # Notify observers of the end event
        self._notify_end_observers()

        self._stop_arrow_server()

        # Log parallel processing performance if enabled
        perf_stats = self.get_parallel_performance_stats()
        if perf_stats and self.rank == 0:
            logger.info(f"Parallel processing performance stats: {perf_stats}")

        self._log_phase_timing_summary()

    def start(self) -> None:
        self.runner.execute()
        end_time = time.time()

        logger.info(f"Simulation took {end_time - self.start_time} seconds.")


# Register CasmPop
Models.add_model(CasmPop.__module__ + "." + CasmPop.__name__, CasmPop)


# 4. Create dictionary mapping of model experiment parameters
# This can be used to run the model with different parameters, e.g. for
# sensitivity analysis or calibration
# Each key in the dictionary is a string describing the experiment, and the
# value is a dictionary of parameters to update from the default parameters

model_parameters = CasmPop.get_default_parameters()

experiment_parameters = {
    "base": model_parameters,
    "dmv": {
        **model_parameters,
        "places.table": "rti_synth_pop_v2_dmv.places",
        "households.table": "rti_synth_pop_v2_dmv.hh",
        "activities.table": "rti_synth_pop_v2_dmv.activities",
        "contacts.table": "rti_synth_pop_v2_dmv.contacts",
        "persons.table": "rti_synth_pop_v2_dmv.persons",
    },
    "dmv_100": {
        **model_parameters,
        "places.table": "rti_synth_pop_v2_dmv_100.places",
        "households.table": "rti_synth_pop_v2_dmv_100.hh",
        "activities.table": "rti_synth_pop_v2_dmv_100.activities",
        "contacts.table": "rti_synth_pop_v2_dmv_100.contacts",
        "persons.table": "rti_synth_pop_v2_dmv_100.persons",
    },
    "dc": {
        **model_parameters,
        "places.table": "rti_synth_pop_v2_dc.places",
        "households.table": "rti_synth_pop_v2_dc.hh",
        "activities.table": "rti_synth_pop_v2_dc.activities",
        "contacts.table": "rti_synth_pop_v2_dc.contacts",
        "persons.table": "rti_synth_pop_v2_dc.persons",
    },
    "dc_5000": {
        **model_parameters,
        "places.table": "rti_synth_pop_v2_dc_5000.places",
        "households.table": "rti_synth_pop_v2_dc_5000.hh",
        "activities.table": "rti_synth_pop_v2_dc_5000.activities",
        "contacts.table": "rti_synth_pop_v2_dc_5000.contacts",
        "persons.table": "rti_synth_pop_v2_dc_5000.persons",
    },
}


# utility functions
def update_activities_data(activities_data: namedtuple, **kwargs) -> namedtuple:
    """Update the activities data."""
    return activities_data._replace(**kwargs)
