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
from repast4py.space import ContinuousPoint as cpt  # noqa: F401

from casmsocial.activities import Act
from casmsocial.data_utilities import get_attribute_names_from_data
from casmsocial.factory import Models
from casmsocial.model import Model
from casmsocial.person import BehaviorEngine, Person, PersonConfig, PersonData
from casmsocial.place import Place, PlaceConfig, PlaceData, RemotePlace, find_closest_location
from casmsocial.sim_time import SimTime
from casmsocial.social_model import MissingRequiredParameterError, SimEnvironment, SIModel, update_activities_data


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
        required_keys = ["environment.file", "weather_at_places.file"]
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

        # 3. initialize the environment: microweather and person weather data
        self.microweather_arrow_file_path = (
            Model.get_model().data_input_path / Model.get_model().params["environment.file"]
        )
        if not self.microweather_arrow_file_path.exists():
            logger.error(f"Error: Environment file {self.microweather_arrow_file_path} not found.")
            raise MissingEnvironmentFile(self.microweather_arrow_file_path)
        self.weather_at_places_arrow_file_path = (
            Model.get_model().data_input_path / Model.get_model().params["weather_at_places.file"]
        )
        if not self.weather_at_places_arrow_file_path.parent.exists():
            logger.error(
                f"Error: Person weather file path {self.weather_at_places_arrow_file_path.parent} does not exist."
            )
            raise MissingEnvironmentFile(self.weather_at_places_arrow_file_path.parent)

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

        # dataframe for weather data for each place
        self.local_weather_df = pl.DataFrame()  # local weather data for the environment

        # set default heat threshold (deprecated)
        self.__heat_threshold = 90.0  # default heat threshold in Fahrenheit
        self.environment_tuple = namedtuple("HeatRiskEnvironment", ["heatIndex", "heatIndexIndoors"])

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
        """Update the microweather snapshot."""
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

        # convert the weather snapshot to a DuckDB table
        query = """
            CREATE OR REPLACE TABLE 'weather'
            AS SELECT * FROM weather_snapshot;
            CREATE OR REPLACE TABLE 'places_linked_to_weather'
            AS SELECT sp_id AS place_id, gridindex FROM places;
            -- CREATE OR REPLACE TABLE 'person_locations'
            -- AS SELECT person_id, place_id FROM person_last_known_location;
            """
        self.conn.execute(query)

        logger.info("Microweather snapshot updated successfully.")

        query = """
            CREATE OR REPLACE TABLE weather_at_places AS
            SELECT
                w.time,
                pl.place_id,
                w.T_xy,
                w.heat_index,
                w.dew_point,
                w.wbgt
            FROM
                weather w
            JOIN
                places_linked_to_weather pl ON w.gridindex = pl.gridindex
            """
        self.conn.execute(query)
        self.weather_at_places_df = self.conn.execute("SELECT * FROM weather_at_places").pl()
        logger.info(f"weather at all places data updated with {self.weather_at_places_df.shape[0]} rows")
        logger.info(f"\n{self.weather_at_places_df.head(5)}")

        # replace 'time' column with string version
        self.weather_at_places_df = self.weather_at_places_df.with_columns(
            [pl.col("time").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("time")]
        )
        return

        # convert the DataFrame to an Arrow table
        weather_at_places_table = self.weather_at_places_df.to_arrow()

        # Define partition schema
        partition_schema = pa.schema([pa.field("time", pa.string())])

        # Set Hive-style partitioning
        partitioning = HivePartitioning(partition_schema)

        # Write dataset
        ds.write_dataset(
            data=weather_at_places_table,
            base_dir=self.weather_at_places_arrow_file_path,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

    def teardown(self) -> None:
        """Teardown the environment."""
        pass

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        super().step(context, cal)  # updates the social environment

        # now update the physical environment
        self.update_snapshot(context, cal)

    def get_values_at_place(self, place: Place) -> namedtuple:
        """Get the value at a place.

        Arguments:
            place: Place: The place to get the value for.
        Returns:
            namedtuple: The value at the place.
        """

        # heatIndex = self.heatIndex_map.get(place.id, self.meanheatindex)
        # heatIndexIndoors = heatIndex

        # if "AIR" in get_attribute_names_from_data(place.data) and place.data.AIR:
        #     heatIndexIndoors = 72

        heatIndex = 72.0
        heatIndexIndoors = 72.0
        return self.environment_tuple(heatIndex=heatIndex, heatIndexIndoors=heatIndexIndoors)

    def get_closest_cooling_station(
        self, place: Place, local_places: list[Place], n: int = 3
    ) -> list[tuple[Place, float]]:
        """Get the closest cooling station.

        Arguments:
            lat: float: The latitude of the person.
            lon: float: The longitude of the person.
            local_places: list[Place]: The list of places to search.
            n: int: The number of closest places to return.

        Returns:
            list[tuple[Place, float]]: The closest cooling stations.
        """
        # if place.id in self.closest_cooling_station:
        #     return self.closest_cooling_station[place.id]
        # lat = place.data.latitude
        # lon = place.data.longitude
        # candidates = find_closest_location(lat, lon, local_places, n=n, filter_func=if_place_has_cooling_center)
        # self.closest_cooling_station[place.id] = candidates
        candidates = []
        return candidates


# 2. Define a PlaceData Class
@dataclass
class PlaceDataWithClimate(PlaceData):
    """Place with heat index data."""

    AIR: bool = False
    cooling_center: float = 0.0  # bool = False


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

    def decide_to_seek_cooling(self, context: ctx.SharedContext, cal: SimTime) -> bool:
        """Decide to seek cooling.

        Probability of a heat event has already been computed and the person has not experienced a heat event.

        Arguments:
            context: ctx.SharedContext: The shared context.
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

        # 1) get the location of the person
        person = self.agent
        places_proj = context.get_projection("places_projection")
        place = places_proj.lookup_place(person.currentPlaceID)  # get_place_for_agent(person)
        lat = place.data.latitude
        lon = place.data.longitude

        # 2) find the closest cooling center
        searching_for_cooling = False
        cooling_center_candidate = None
        if searching_for_cooling:
            local_places = places_proj.get_local_places()

            candidates = find_closest_location(lat, lon, local_places, n=3, filter_func=if_place_has_cooling_center)
            # candidates = \
            #    Model.get_model().get_environment().get_closest_cooling_station(place, local_places, n=3)
            if len(candidates) == 0:
                logger.error(f"No cooling centers found near person {self.agent.id}")
                return False  # no cooling centers found
            for place, distance in candidates:
                logger.debug(f"  closest cooling center {place.id} at {distance} km")

            # selecting the first one for now
            cooling_center_candidate = candidates[0][0]
        else:
            # just use the person's current place as the cooling center
            cooling_center_candidate = place

        # 3) decide to seek cooling
        #    - inputs: probability of a heat event, heat index, distance to cooling center
        #    - later, could add more factors
        #    - for now, just move to the closest cooling center
        seeking_cooling = np.random.rand() < self.agent.state.prob_heat_event
        if seeking_cooling:
            self.move_to_cooling_center(cooling_center_candidate, cal.hour_of_day)
            return True

        return False

    def move_to_cooling_center(self, place: Place, current_hour: int) -> None:
        """Move the person to a cooling center.

        Arguments:
            place: Place: The cooling center to move to is available.
            current_hour: int: The current hour of the day.
        """
        # find the next hour
        next_hour = current_hour + 1
        start_time = next_hour * 60

        # currently set end time to the end of the day
        end_time = 1440  # 24 hours

        # find the activity and schedule indices
        activity_names = SIModel.get_activity_names()
        activity_id = activity_names.index("cooling_center")

        person = self.agent
        if person.state.moved_to_cooling_center:
            logger.error(f"Person {person.id} has already moved to a cooling center")
            return
        person.state.moved_to_cooling_center = True
        person.state.cooling_center_id = place.id

        schedule_names = [schedule.name for schedule in person.schedules.schedules]
        schedule_idx = schedule_names.index("cooling_center")
        if self.agent.state.activities_idx == schedule_idx:
            logger.error(f"Person {person.id} is already in the cooling center schedule")
            return

        # need to modify the person's activities to include the cooling
        activities_data = person.state.places
        person.state.places = update_activities_data(activities_data, cooling_center=place.id)

        act_go_to_cooling_center = Act(person.id, activity_id, 1.0, start_time, end_time)

        # add the activity to the schedule
        person.schedules.schedules[schedule_idx].addAct(act_go_to_cooling_center)
        logger.debug(f"Person {person.id} updated schedule to move to cooling center {place.id}")
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

        environment = Model.get_model().get_environment()
        environment_values = environment.get_values_at_place(place)

        person = self.agent
        local_heat_index = environment_values.heatIndexIndoors
        if person.state.outside_worker:
            local_heat_index = environment_values.heatIndex

        person.state.heat_indices.appendleft(local_heat_index)

        # update the probability of a heat event
        person.state.prob_heat_event = self.compute_prob_heat_event(environment.heat_threshold)

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

        if self.decide_to_seek_cooling(context, cal):
            person.state.heat_event_place_id = place.id
            logger.info(f"Person {self.agent.id} is seeking cooling at hour {cal.hour_of_day}")
            return
        else:
            logger.debug(f"Person {self.agent.id} is not seeking cooling at hour {cal.hour_of_day}")


# 6. Define the HeatRiskModel class
class HeatRiskModel2(SIModel):
    """HeatRiskModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the HeatRiskModel class"""
        super().__init__(comm, params)

        # add queries for the model
        self.queries["create_closest_cooling_station"] = """
            load spatial;
            CREATE OR REPLACE TABLE closest_cooling_center AS
            SELECT
                p.sp_id,
                p.location,
                p.cooling_center, -- Include original cooling_center status for completeness
                cc.sp_id AS closest_cooling_center_id,
                ST_Distance(p.location, cc.location) AS distance_to_closest_cooling_center_m
            FROM
                places AS p
            CROSS JOIN -- Use CROSS JOIN to get all combinations initially
                places AS cc
            WHERE
                cc.cooling_center = TRUE -- Filter for only cooling centers in the right side of the join
            QUALIFY ROW_NUMBER() OVER (PARTITION BY p.sp_id ORDER BY ST_Distance(p.location, cc.location) ASC) = 1;
            """
        self.queries["update_weather_places"] = """
            CREATE OR REPLACE TABLE weather_at_places AS
            SELECT
                w.time,
                p.place_id,
                w.T_xy,
                w.heat_index,
                w.dew_point,
                w.wbgt
            FROM
                weather w
            JOIN
                places pl ON w.gridindex = pl.gridindex;
            """

        # show
        logger.info(f"HeatRiskModel2 initialized at time={time.time() - self.start_time} seconds")

        # initialize the environment
        self._heat_threshold = 90.0

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def initialize_population(self) -> None:
        """Initialize population"""

        # register the environment
        logger.info(f"Registering HeatRiskEnvironment at time={time.time()-self.start_time} seconds...")
        SIModel.register_environment(HeatRiskEnvironment("HeatRiskEnvironment"))
        logger.info("HeatRiskEnvironment registered at time={time.time()-self.start_time} seconds. ")

        # register the place types
        SIModel.register_place_config(PlaceConfig(name="Places", place_type=Place, dataType=PlaceDataWithClimate))

        # register the remote place type
        SIModel.register_remote_place_config(
            PlaceConfig(name="RemotePlace", place_type=RemotePlace, dataType=PlaceData)
        )

        # register the person type
        SIModel.register_person_config(
            PersonConfig(
                name="Person",
                person_type=Person,
                dataType=PersonDataWithHeatRisk,
                behaviorEngine=HeatRiskBehaviorEngine,
            )
        )

        # register the activities
        SIModel.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
        SIModel.register_activity_names(["home", "work", "school", "cooling_center"])

        logger.debug("Now running initialize population for SIModel...")

        super().initialize_population()
        logger.info("Population initialized for HeatRiskModel2")

        # set up the environment
        self._heat_threshold = 90.0
        # self._heat_threshold = float(self.params['heat_threshold'])

        # create table for closest cooling centers
        logger.info("Creating table for closest cooling centers...")
        self.conn.execute(self.queries["create_closest_cooling_station"])
        logger.info("Created table for closest cooling centers.")

        query = """
        SELECT * FROM closest_cooling_center;
        """
        closest_cooling_station = self.conn.execute(query).pl()
        if closest_cooling_station.is_empty():
            logger.warning("No cooling stations found in the environment data.")
        else:
            logger.info(f"Found {closest_cooling_station.shape[0]} cooling stations in the environment data.")
            logger.info(f"Cooling stations:\n{closest_cooling_station.head(5)}")
        # self.closest_cooling_station = closest_cooling_station  # store the closest cooling stations in a DataFrame
        # self.closest_cooling_station = {}  # cache for closest cooling stations

        # check the first agent
        person = next(self.context.agents())
        logger.info(person)
        # test_person_serialization(person)
        # test_activities(person)
        # test_add_move_to_cooling_center(person)
        # test_activities(person)
        # set up polars table for the agents
        self.agent_data = pl.DataFrame()

        # # initialize the logging
        # self.agent_logger = logging.TabularLogger(
        #     self.comm,
        #     self.params["agent_log_file"],
        #     [
        #         "tick",
        #         "agent_id",
        #         "x",
        #         "y",
        #         "heatIndex",
        #         "hrsAboveHeatThreshold",
        #         "probHeatEvent",
        #         "experiencedHeatEvent",
        #         "movedToCoolingCenter",
        #         "heatEventPlaceId",
        #         "coolingCenterId",
        #     ],
        # )
        self.log_agents()

    def step(self) -> None:
        """Step the model."""
        super().step()
        logger.info("Running step for HeatRiskModel")

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.minute_of_day

        logger.info(f"Logging agents at tick {tick}")
        # get the current state of the agents
        # persons_data_df = pl.DataFrame([person.state for person in self.context.agents(agent_type=0)])
        # logger.info(f"Logging {len(persons_data_df)} agents at tick {tick}")
        # persons_data_df = persons_data_df.with_columns(
        #     [
        #         pl.col("tick").fill_null(tick),
        #         pl.col("heat_indices").apply(lambda x: x[0], return_dtype=pl.Float64),
        #         pl.col("heat_indices").apply(lambda x: len(filter_heat_indices(x, self.get_environment().heat_threshold))),
        #     ]
        # )
        return

        heat_threshold = self.get_environment().heat_threshold

        for person in self.context.agents():
            heat = filter_heat_indices(person.state.heat_indices, heat_threshold)
            self.agent_logger.log_row(
                tick,
                person.id,
                person.pt.x,
                person.pt.y,
                person.state.heat_indices[0],
                len(heat),
                person.state.prob_heat_event,
                person.state.experienced_heat_event,
                person.state.moved_to_cooling_center,
                person.state.heat_event_place_id,
                person.state.cooling_center_id,
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        # self.agent_logger.close()
        pass


# Register HeatRiskModel
Models.add_model(HeatRiskModel2.__module__ + "." + HeatRiskModel2.__name__, HeatRiskModel2)
