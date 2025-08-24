"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""


import time
from collections import deque, namedtuple
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import repast4py.context as ctx
from loguru import logger
from mpi4py import MPI
from pyarrow.dataset import HivePartitioning

from casmsocial.activities import Act
from casmsocial.casmpop import (
    CasmPop,
    MissingRequiredParameterError,
    SimEnvironment,
    update_activities_data,
)
from casmsocial.data_utilities import get_attribute_names_from_data
from casmsocial.factory import Models
from casmsocial.model import Model
from casmsocial.person import BehaviorEngine, Person, PersonData
from casmsocial.place import Place, PlaceData
from casmsocial.sim_time import SimTime


# utility functions for heat-related computations
def if_place_has_cooling_center(place) -> bool:
    """Check if a place has a cooling center.

    Arguments:
        place: Place: The place to check.

    Returns:
        bool: True if the place has a cooling center, False otherwise.
    """
    return getattr(place.data, "cooling_center", False) and place.data.cooling_center > 0.0


def if_place_has_air_conditioning(place) -> bool:
    """Check if a place has air conditioning.

    Arguments:
        place: Place: The place to check.

    Returns:
        bool: True if the place has air conditioning, False otherwise.
    """
    return "AIR" in get_attribute_names_from_data(place.data) and place.data.AIR


def filter_heat_indices(heat_indices: list[float], threshold: float) -> list[float]:
    """Filter out all heat indices above the threshold."""
    exceeded = True
    return [t for t in heat_indices if (exceeded := exceeded and t > threshold)]


class MissingEnvironmentFile(Exception):
    def __init__(self, value):
        """Constructor for the MissingEnvironmentFile exception."""
        super().__init__(f"Missing environment file: {value}")
        self.value = value


# 1. define the environment
class HeatRiskEnvironment(SimEnvironment):
    """HeatRiskEnvironment class"""

    def __init__(self, name: str):
        """Constructor for the HeatRiskEnvironment class."""
        super().__init__(name)

        logger.info("Initializing HeatRiskEnvironment...")

        # 1. check if the required parameters are present
        # and if the files exist
        logger.info("Checking required parameters for HeatRiskEnvironment...")
        required_keys = ["environment.file", "closest_cooling_center.file", "heat_threshold"]
        if not all(key in Model.get_model().params for key in required_keys):
            logger.error(f"Error: Missing required parameters in model: {required_keys}")
            raise MissingRequiredParameterError(required_keys)

        # 2. establish the connection to the database
        # this will be used to query the microweather data
        # and to update the microweather snapshot
        self.conn = Model.get_model().conn
        if self.conn is None:
            logger.error("Database connection is not set. Cannot update microweather snapshot.")
            return

        # 3. initialize the environment: microweather data
        #    - this is the data that will be used to compute the heat index and other
        #      heat-related values for each place
        self.microweather_arrow_file_path = Model.get_model().data_path / Model.get_model().params["environment.file"]
        if not self.microweather_arrow_file_path.exists():
            logger.error(f"Error: Environment file {self.microweather_arrow_file_path} not found.")
            raise MissingEnvironmentFile(self.microweather_arrow_file_path)
        logger.info(f"Loading environment data from {self.microweather_arrow_file_path}...")
        microweather_dataset = ds.dataset(self.microweather_arrow_file_path, format="parquet", partitioning="hive")
        microweather_table = microweather_dataset.to_table()
        self.microweather_df = pl.from_arrow(microweather_table)

        # convert the time column to datetime
        self.microweather_df = self.microweather_df.with_columns(
            [
                pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S")  # or include time if needed
            ]
        )
        logger.info(f"Loaded microweather data with {self.microweather_df.shape[0]} rows")
        logger.info(f"Microweather data schema is {self.microweather_df.schema}")

        # set default heat threshold (deprecated)
        self.__heat_threshold = Model.get_model().params.get(
            "heat_threshold", 32.2222
        )  # default heat threshold in Celsius 32.2222 (90.0 Fahrenheit)

        # values from the microweather data
        self.fields = [
            "T_xy",  # temperature in Celsius
            "heat_index",  # heat index in Celsius
            "dew_point",  # dew point in Celsius
            "wbgt",  # wet bulb globe temperature in Celsius
        ]
        self.environment_tuple = namedtuple("HeatRiskEnvironment", self.fields)

    @property
    def heat_threshold(self) -> float:
        return self.__heat_threshold

    def setup(self) -> None:
        """Setup the environment."""
        logger.info("Setting up the HeatRiskEnvironment")

        context = Model.get_model().context
        cal = Model.get_model().cal

        self.update_snapshot(context, cal)

    def update_snapshot(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the microweather snapshot.
        This method updates the microweather snapshot in the database
        with the current microweather data for the given time.
        Arguments:
            context: ctx.SharedContext: The shared context.
            cal: SimTime: The calendar.
        """
        # check if the database connection is set
        if self.conn is None:
            self.conn = Model.get_model().conn
            if self.conn is None:
                logger.error("Database connection is not set. Cannot update microweather snapshot.")
                return

        # get the current time from the calendar: needed to filter the microweather data
        current_time = cal.datetime

        # check if the microweather data is available for the current time
        logger.info(f"Updating microweather snapshot at {current_time}")
        weather_snapshot = self.microweather_df.filter(pl.col("time") == current_time.replace(tzinfo=None))
        if weather_snapshot.is_empty() or weather_snapshot.shape[0] == 0:
            logger.error(f"No microweather data found for time {current_time}.")
            # raise ValueError(f"No microweather data found for time {current_time}.")
            return
        logger.info(f"Microweather data for {current_time}:\n{weather_snapshot.shape[0]} rows")
        logger.info(f"\n{weather_snapshot.head(5)}")

        # convert the weather snapshot to a DuckDB table (update the weather table)
        # self.conn.execute("""
        #     CREATE OR REPLACE TABLE 'weather'
        #     AS SELECT * FROM weather_snapshot;
        #     """)
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE weather AS
            SELECT
                w.time,
                w.place_id,
                w.T_xy,
                w.heat_index,
                w.dew_point,
                w.wbgt,
                cc.AIR,
                cc.cooling_center,
                cc.distance_to_closest_cooling_center_m
            FROM weather_snapshot AS w
            JOIN closest_cooling_center AS cc ON w.place_id = cc.sp_id; -- Join condition
            """
        )
        weather_with_cc = self.conn.execute("SELECT * FROM weather").fetchall()
        if not weather_with_cc:
            logger.error("No weather data found after updating the snapshot.")
            return
        # log the updated weather data
        logger.info(f"Updated microweather snapshot with {len(weather_with_cc)} rows.")
        places_proj = context.get_projection("places_projection")
        for row in weather_with_cc:
            place_id = row[1]  # second column is place_id
            place = places_proj.lookup_place(place_id)
            if place is None:
                logger.error(f"Place with ID {place_id} not found in places projection.")
                continue
            # update the place data with the microweather data
            place.data.T_xy = row[2]  # T_xy
            place.data.heat_index = row[3]  # heat_index
            place.data.dew_point = row[4]  # dew_point
            place.data.wbgt = row[5]  # wbgt
            place.data.AIR = row[6]  # AIR
            place.data.cooling_center = row[7]  # cooling_center
            place.data.distance_to_closest_cooling_center_m = row[8]  # distance to closest cooling center in meters
            # set the closest cooling center ID
            place.data.closest_cooling_center_id = row[0]  # first column is time

        logger.info("Microweather snapshot updated successfully.")

    def teardown(self) -> None:
        """Teardown the environment."""
        pass

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        super().step(context, cal)  # updates the social environment

        # now update the physical environment
        self.update_snapshot(context, cal)

    def get_closest_cooling_center(self, place_id: int, n: int = 3) -> list[tuple[int, float]]:
        """Get the closest cooling station.

        Arguments:
            place_id: int: The ID of the place to find the closest cooling station for.
            n: int: The number of closest places to return.

        Returns:
            list[tuple[int, float]]: The closest cooling stations.
        """
        result = self.conn.execute(
            """
            SELECT cc.sp_id, cc.location, cc.cooling_center, distance_to_closest_cooling_center_m
            FROM places AS p
            JOIN closest_cooling_center AS cc ON p.sp_id = cc.sp_id
            WHERE p.sp_id = ?
            ORDER BY distance_to_closest_cooling_center_m ASC
            LIMIT ?;
            """,
            (place_id, n),
        ).fetchall()
        candidates = [(place, distance) for place, _, _, distance in result]

        return candidates


# 2. Define a PlaceData Class
@dataclass
class PlaceDataWithClimate(PlaceData):
    """Place with heat index data."""

    AIR: bool = False
    cooling_center: float = 0.0  # bool = False
    T_xy: float = float("nan")  # temperature in Celsius
    heat_index: float = float("nan")  # heat index in Celsius
    dew_point: float = float("nan")  # dew point in Celsius
    wbgt: float = float("nan")  # wet bulb globe temperature in Celsius
    closest_cooling_center_id: int = None  # ID of the closest cooling center
    distance_to_closest_cooling_center_m: float = float("nan")  # distance to the closest cooling center in meters


# 3. Define a PersonData Class with heat risk data
@dataclass
class PersonDataWithHeatRisk(PersonData):
    """Data for a Person."""

    outside_worker: bool = False
    heat_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    prob_heat_event: float = 0.0
    experienced_heat_event: bool = False
    moved_to_cooling_center: bool = False
    heat_event_place_id: int = None
    cooling_center_id: int = None


# 4. Define a HeatRiskBehaviorEngine Class
class HeatRiskBehaviorEngine(BehaviorEngine):
    """HeatRiskBehaviorEngine class

    This class defines the behavior engine for a person in the heat risk model.

    After the model has a probability of a heat event, the person decides to seek cooling.
    - 1) prob
    - 2) knowing the probability of
    """

    def __init__(self, person: Person):
        """Constructor for the HeatRiskBehaviorEngine class."""
        super().__init__(person)

    def compute_prob_heat_event(self, threshold: float) -> float:
        """Compute the probability of a heat event."""
        heat_indices = self.agent.state.heat_indices

        # filter out all heat indices above the threshold
        heat_index = heat_indices[0]
        heat = filter_heat_indices(heat_indices, threshold)
        hours_above_threshold = len(heat)

        # note: length of heat is the number of hours above the threshold
        # prob_heat_event = \
        #     1 - (1 - ((heat_indices[0] - threshold/80.0) ** 2) ** (3 * len(heat)))
        prob_heat_event = 1 - (1 - ((heat_index - threshold) / 80.0) ** 2) ** (3 * hours_above_threshold)
        return prob_heat_event

    def decide_to_seek_cooling(self, place: Place, cal: SimTime) -> bool:
        """Decide to seek cooling.

        Probability of a heat event has already been computed and the person has not experienced a heat event.

        Arguments:
            place: Place: The place where the person is located.
            cal: SimTime: The calendar.

        Returns:
            bool: True if the person decides to seek cooling, False otherwise.
        """
        # TODO (2025-02-26 jcline): implement this method
        #   - This is where the person decides to seek cooling
        #   - This could be based on the heat index, probability of a heat event,
        #     or other factors
        #   - Also looking at the place data for cooling centers and
        #     air conditioning - looking for the three closest places

        consider_seeking_cooling = np.random.rand() < self.agent.state.prob_heat_event
        if not consider_seeking_cooling:
            return False

        logger.info(f"Person {self.agent.id} is considering seeking cooling")

        # 2) find the closest cooling center
        cooling_center_candidate_id = place.data.closest_cooling_center_id
        if cooling_center_candidate_id is None:
            # no cooling center is accessible from the person's current place
            logger.warning(f"Person {self.agent.id} has no cooling center at place {place.id}.")

            # just use the person's current place as the cooling center
            cooling_center_candidate_id = place.id

        # 3) decide to seek cooling
        #    - inputs: probability of a heat event, heat index, distance to cooling center
        #    - later, could add more factors
        #    - for now, just move to the closest cooling center
        seeking_cooling = np.random.rand() < self.agent.state.prob_heat_event
        if seeking_cooling:
            self.move_to_cooling_center(cooling_center_candidate_id, cal.hour_of_day)
            return True

        return False

    def move_to_cooling_center(self, place_id: int, current_hour: int) -> None:
        """Move the person to a cooling center.

        Arguments:
            place_id: int: The ID of the cooling center place.
            current_hour: int: The current hour of the day.
        """
        # find the next hour
        next_hour = current_hour + 1
        start_time = next_hour * 60

        # currently set end time to the end of the day
        end_time = 1440  # 24 hours

        # find the activity and schedule indices
        activity_names = CasmPop.get_activity_names()
        activity_id = activity_names.index("cooling_center")

        person = self.agent
        if person.state.moved_to_cooling_center:
            logger.error(f"Person {person.id} has already moved to a cooling center")
            return
        person.state.moved_to_cooling_center = True
        person.state.cooling_center_id = place_id

        schedule_names = [schedule.name for schedule in person.schedules.schedules]
        schedule_idx = schedule_names.index("cooling_center")
        if self.agent.state.activities_idx == schedule_idx:
            logger.error(f"Person {person.id} is already in the cooling center schedule")
            return

        # need to modify the person's activities to include the cooling
        activities_data = person.state.places
        person.state.places = update_activities_data(activities_data, cooling_center=place_id)

        act_go_to_cooling_center = Act(person.id, activity_id, 1.0, start_time, end_time)

        # add the activity to the schedule
        person.schedules.schedules[schedule_idx].addAct(act_go_to_cooling_center)
        logger.debug(f"Person {person.id} updated schedule to move to cooling center {place_id}")
        for act in person.schedules.schedules[schedule_idx].acts:
            logger.debug(f"  {act}")
        previous_schedule = person.state.activities_idx
        person.state.activities_idx = schedule_idx
        logger.debug(f"Person {person.id} moved to schedule {schedule_idx} from {previous_schedule}")

    def decide(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Decide what to do."""
        # TODO (2025-02-26 jcline): implement this method
        #   - This is where the person decides what to do
        #   - This could be based on the heat index, probability of a heat event,
        #     or other factors

        # check if the person has already experienced a heat event
        if self.agent.state.experienced_heat_event:
            logger.debug(f"Person {self.agent.id} has already experienced a heat event")
            return

        if self.agent.state.moved_to_cooling_center:
            logger.debug(f"Person {self.agent.id} has already moved to a cooling center")
            return

        # get the heat index at the person's location
        places_proj = context.get_projection("places_projection")
        place = places_proj.lookup_place(self.agent.currentPlaceID)
        if not place:
            logger.error(f"Person {self.agent.id} does not have a place")
            return

        if place.data.T_xy == float("nan"):
            logger.error(f"Person {self.agent.id} has no temperature data at place {place.id}")
            return

        # note: if the place has air conditioning, we assume the heat index is 22.2222C (72F)
        # - 'T_xy' is the temperature in F at the place, not the heat index
        # - 'heat_index' is the heat index in C at the place
        # - The other values are also in C -- not sure how to process them yet
        # all environment variables including closest cooling center are in place.data

        # log the environment values
        person = self.agent
        local_heat_index = place.data.T_xy  # or should this be heat_index
        if place.data.AIR and person.state.outside_worker:
            local_heat_index = 22.2222  # assume air conditioning sets the heat index to 22.2222C (72F)

        person.state.heat_indices.appendleft(local_heat_index)

        # update the probability of a heat event
        environment = Model.get_model().get_environment()
        person.state.prob_heat_event = self.compute_prob_heat_event(environment.heat_threshold)
        logger.debug(
            f"Person {self.agent.id} has heat index {local_heat_index}C and prob_heat_event {person.state.prob_heat_event}"
        )
        return  # skip the rest of the decision-making process for now

        # compute whether a heat event has occurred
        if np.random.rand() < self.agent.state.prob_heat_event:
            # if a heat event has occurred, set the flag and return
            #   - This person has experienced a heat event, so they will not seek cooling
            #   - This is a simplification for now - person will conitnue to act as if they have not experienced a heat event
            #   - This person is now immune to future heat events
            person.state.experienced_heat_event = True
            person.state.heat_event_place_id = place.id

            logger.info(f"Person {self.agent.id} has experienced a heat event at hour {cal.hour_of_day}")
            return

        if self.decide_to_seek_cooling(place, cal):
            person.state.heat_event_place_id = place.id
            logger.info(f"Person {self.agent.id} is seeking cooling at hour {cal.hour_of_day}")
            return
        else:
            logger.debug(f"Person {self.agent.id} is not seeking cooling at hour {cal.hour_of_day}")


# 6a. Define agent log data
@dataclass
class PersonLogData:
    """Data for logging person agent information."""

    minute_of_day: int
    rank: int  # rank of the agent in the MPI communicator
    agent_id: int
    x: float
    y: float
    heatIndex: float
    hrsAboveHeatThreshold: int
    probHeatEvent: float
    experiencedHeatEvent: bool
    movedToCoolingCenter: bool
    heatEventPlaceId: int
    coolingCenterId: int


# 6b. Define the HeatRiskModel class
class HeatRiskModel2(CasmPop):
    """HeatRiskModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the HeatRiskModel class"""
        super().__init__(comm, params)

        # create the agent log file path
        if "agent_log_file" not in self.params:
            raise MissingRequiredParameterError(["agent_log_file"])
        self.agent_log_file = self.data_path / self.params["agent_log_file"]
        if not self.agent_log_file.parent.exists():
            self.agent_log_file.parent.mkdir(parents=True, exist_ok=True)

        # show the initialization time
        logger.info(f"HeatRiskModel2 initialized at time={time.time() - self.start_time} seconds")

        # initialize the environment
        self._heat_threshold = 90.0

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def build_context(self) -> None:
        """Initialize population"""

        # register the environment
        logger.info(f"Registering HeatRiskEnvironment at time={time.time()-self.start_time} seconds...")
        CasmPop.register_environment(HeatRiskEnvironment("HeatRiskEnvironment"))
        logger.info("HeatRiskEnvironment registered at time={time.time()-self.start_time} seconds. ")

        # register the person and place agent types
        logger.info(f"Registering person type (TYPE={Person.TYPE})...")
        CasmPop.setPersonClass(Person, PersonDataWithHeatRisk)

        logger.info(f"Registering place type (TYPE={Place.TYPE})...")
        CasmPop.setPlaceClass(Place, PlaceDataWithClimate)

        # register the person behavior engine
        Person.registerBehaviorEngine(HeatRiskBehaviorEngine)

        # register the activities
        CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
        CasmPop.register_activity_names(["home", "work", "school", "cooling_center"])

        logger.debug("Now running initialize population for CasmPop...")

        super().build_context()
        logger.info("Population initialized for HeatRiskModel2")

        # set up the environment
        environment = self.get_environment()
        self._heat_threshold = environment.heat_threshold
        logger.info(f"Heat threshold set to {self._heat_threshold}C")

        # check the first agent
        person = next(self.context.agents())
        logger.info(person)
        candidates = environment.get_closest_cooling_center(person.currentPlaceID, 3)
        if not candidates:
            logger.error(f"No cooling stations found for person {person.id} at place {person.currentPlaceID}.")
        else:
            logger.info(f"Closest cooling stations for person {person.id} at place {person.currentPlaceID}:")
            for row in candidates:
                logger.info(f"  Cooling station {row[0]} at {row[1]} with distance {row[1]} meters")

        # log the agents
        logger.info("Logging agents after population initialization")
        self.log_agents()

    def create_input_tables(self):
        logger.info("Creating input tables for HeatRiskModel2...")
        super().create_input_tables()

        # load the closest cooling stations data
        closest_cooling_center_arrow_file_path = (
            Model.get_model().data_path / Model.get_model().params["closest_cooling_center.file"]
        )
        if not closest_cooling_center_arrow_file_path.exists():
            logger.error(f"Error: Closest cooling station file {closest_cooling_center_arrow_file_path} not found.")
            raise MissingEnvironmentFile(closest_cooling_center_arrow_file_path)
        closest_cooling_center_df = pl.read_parquet(closest_cooling_center_arrow_file_path)
        logger.info(f"Loaded closest cooling centers data with {closest_cooling_center_df.shape[0]} rows")
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE closest_cooling_center AS
            SELECT * FROM closest_cooling_center_df;
            """
        )

    def step(self) -> None:
        """Step the model."""
        super().step()
        logger.info("Running step for HeatRiskModel")

    def get_person_log_data(self, person: Person) -> PersonLogData:
        """Get the agent data for logging."""
        heat_threshold = self.get_environment().heat_threshold
        heat = filter_heat_indices(person.state.heat_indices, heat_threshold)

        return PersonLogData(
            minute_of_day=self.cal.minute_of_day,
            rank=self.comm.Get_rank(),
            agent_id=person.id,
            x=person.pt.x,
            y=person.pt.y,
            heatIndex=person.state.heat_indices[0],
            hrsAboveHeatThreshold=len(heat),
            probHeatEvent=person.state.prob_heat_event,
            experiencedHeatEvent=person.state.experienced_heat_event,
            movedToCoolingCenter=person.state.moved_to_cooling_center,
            heatEventPlaceId=person.state.heat_event_place_id,
            coolingCenterId=person.state.cooling_center_id,
        )

    def log_agents(self) -> None:
        """Log the agents' data."""
        # create a DataFrame for the agent logs
        logger.info("Logging agents' data...")
        agent_log_df = pl.DataFrame([self.get_person_log_data(person) for person in self.context.agents(agent_type=0)])

        # convert the DataFrame to an Arrow Table
        agent_log_table = agent_log_df.to_arrow()

        # Define partition schema
        partition_schema = pa.schema(
            [
                pa.field("minute_of_day", pa.int32()),
                pa.field("rank", pa.int32()),
            ]
        )

        # Set Hive-style partitioning
        partitioning = HivePartitioning(partition_schema)

        # Write dataset
        ds.write_dataset(
            data=agent_log_table,
            base_dir=self.agent_log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )


# Register HeatRiskModel
Models.add_model(HeatRiskModel2.__module__ + "." + HeatRiskModel2.__name__, HeatRiskModel2)
