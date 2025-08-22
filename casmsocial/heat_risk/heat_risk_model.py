"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""


from collections import deque, namedtuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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
from casmsocial.household import Household
from casmsocial.model import Model
from casmsocial.person import BehaviorEngine, Person, PersonConfig, PersonData
from casmsocial.place import Place, PlaceConfig, PlaceData, RemotePlace, find_closest_location
from casmsocial.school import School
from casmsocial.sim_time import SimTime
from casmsocial.social_model import AgentTypeConfig, SimEnvironment, SIModel, update_activities_data
from casmsocial.workplace import Workplace


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


# 1. define the environment
class HeatRiskEnvironment(SimEnvironment):
    """HeatRiskEnvironment class"""

    def __init__(self, name: str):
        """Constructor for the HeatRiskEnvironment class."""
        super().__init__(name)

        # load the heat index data
        theModel = Model.get_model()
        self.heatindex_by_hour_place_file_path = theModel.data_input_path / theModel.params["heatIndex.file"]
        if self.heatindex_by_hour_place_file_path.exists():
            logger.debug(f"Loading heat map places from {self.heatindex_by_hour_place_file_path}")
        else:
            logger.error(f"Error: Heat map places file {self.heatindex_by_hour_place_file_path} not found.")
            exit(1)

        # initialize the heat threshold
        self._heat_threshold = 90.0
        # self._heat_threshold = float(theModel.params['heat_threshold'])
        # self.heat_indices = deque([float("nan")])

        self.environment_tuple = namedtuple("HeatRiskEnvironment", ["heatIndex", "heatIndexIndoors"])
        self.heatIndex_map: dict[int, float] = {}
        self.meanheatindex: float = float("nan")

        self.closest_cooling_station = dict[int, list[tuple[Place, float]]]
        self.cooling_stations = dict[int, list[tuple[Place, float]]]

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def setup(self) -> None:
        """Setup the environment."""
        pass

    def teardown(self) -> None:
        """Teardown the environment."""
        pass

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        super().step(context, cal)  # updates the social environment

        # now update the physical environment
        logger.debug(f"Updating the environment for hour {cal.hour_of_day}")

        # update the heat indices
        heatindex_by_hour_place = (
            pd.read_parquet(
                self.heatindex_by_hour_place_file_path,
                engine="pyarrow",
                filters=[("time_hour", "=", cal.hour_of_day)],
            )
            .loc[:, ["sp_id", "heatIndex"]]
            .dropna()
        )

        # logger.debug(f"size of heatindex_by_hour_place = {len(heatindex_by_hour_place)}")
        minheatindex = heatindex_by_hour_place["heatIndex"].min()
        maxheatindex = heatindex_by_hour_place["heatIndex"].max()
        self.meanheatindex = heatindex_by_hour_place["heatIndex"].mean()
        logger.debug(
            f"min heat index = {minheatindex}, "
            f"max heat index = {maxheatindex}, "
            f"mean heat index = {self.meanheatindex}"
        )

        self.heatIndex_map = heatindex_by_hour_place.set_index("sp_id")["heatIndex"].to_dict()

    def get_values_at_place(self, place: Place) -> namedtuple:
        """Get the value at a place.

        Arguments:
            place: Place: The place to get the value for.
        Returns:
            namedtuple: The value at the place.
        """

        heatIndex = self.heatIndex_map.get(place.id, self.meanheatindex)
        heatIndexIndoors = heatIndex

        if "AIR" in get_attribute_names_from_data(place.data) and place.data.AIR:
            heatIndexIndoors = 72

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
        if place.id in self.closest_cooling_station:
            return self.closest_cooling_station[place.id]
        lat = place.data.latitude
        lon = place.data.longitude
        candidates = find_closest_location(lat, lon, local_places, n=n, filter_func=if_place_has_cooling_center)
        self.closest_cooling_station[place.id] = candidates
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
class HeatRiskModel(SIModel):
    """HeatRiskModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the HeatRiskModel class"""
        super().__init__(comm, params)

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def build_context(self) -> None:
        """Initialize population"""

        # register the environment
        SIModel.register_environment(HeatRiskEnvironment("HeatRiskEnvironment"))

        # register the person and place agent types
        logger.info(f"Registering person type (TYPE={Person.TYPE})...")
        SIModel.register_agent_type_config(
            AgentTypeConfig(name="Person", agent_type=Person, agent_data_type=PersonData)
        )

        logger.info(f"Registering place type (TYPE={Place.TYPE})...")
        SIModel.register_agent_type_config(AgentTypeConfig(name="Place", agent_type=Place, agent_data_type=PlaceData))

        # register the place types
        SIModel.register_place_config(
            PlaceConfig(name="Household", place_type=Household, dataType=PlaceDataWithClimate)
        )
        SIModel.register_place_config(
            PlaceConfig(name="Workplace", place_type=Workplace, dataType=PlaceDataWithClimate)
        )
        SIModel.register_place_config(PlaceConfig(name="School", place_type=School, dataType=PlaceDataWithClimate))

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

        super().build_context()

        self._heat_threshold = 90.0
        # self._heat_threshold = float(self.params['heat_threshold'])

        # check the first agent
        person = next(self.context.agents())
        logger.info(person)
        # test_person_serialization(person)
        # test_activities(person)
        # test_add_move_to_cooling_center(person)
        # test_activities(person)
        # set up polars table for the agents

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
            base_dir=self.params["agent_log_file"],
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )


# Register HeatRiskModel
Models.add_model(HeatRiskModel.__module__ + "." + HeatRiskModel.__name__, HeatRiskModel)


# 7. create test functions
def test_add_move_to_cooling_center(person: Person):
    """Test the add_move_to_cooling_center function."""
    logger.debug(f"Testing add_move_to_cooling_center for person {person.id}")
    # TODO (2025-02-26 jcline): implement this function
    #   - This is where the person is moved to a cooling center
    #   - Find the closest cooling center
    #   - Schedule moving the person to the cooling center
    current_hour = 12
    next_hour = current_hour + 1
    start_time = next_hour * 60
    end_time = 1440  # 24 hours

    activity_names = SIModel.get_activity_names()
    activity_id = activity_names.index("cooling_center")

    schedule_names = [schedule.name for schedule in person.schedules.schedules]
    schedule_idx = schedule_names.index("cooling_center")

    # need to modify the person's activities to include the cooling
    activities_data = person.state.places
    person.state.places = update_activities_data(activities_data, cooling_center=10)

    act_go_to_cooling_center = Act(person.id, activity_id, 1.0, start_time, end_time)

    # add the activity to the schedule
    person.schedules.schedules[schedule_idx].addAct(act_go_to_cooling_center)
