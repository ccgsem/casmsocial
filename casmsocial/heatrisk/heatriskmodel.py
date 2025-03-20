"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""


from collections import deque, namedtuple
from dataclasses import dataclass, field

import pandas as pd
import repast4py.context as ctx
from loguru import logger
from mpi4py import MPI
from repast4py import logging
from repast4py.space import ContinuousPoint as cpt  # noqa: F401

from casmsocial.activities import Act
from casmsocial.calendar import Calendar
from casmsocial.datautility import get_attribute_names_from_data

# model factory
from casmsocial.factory import Models

# place types
from casmsocial.household import Household
from casmsocial.model import Model
from casmsocial.person import BehaviorEngine, Person, PersonConfig, PersonData
from casmsocial.place import Place, PlaceConfig, PlaceData, RemotePlace, find_closest_location
from casmsocial.school import School
from casmsocial.socialmodel import SimEnvironment, SIModel, update_activities_data
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

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def setup(self) -> None:
        """Setup the environment."""
        pass

    def teardown(self) -> None:
        """Teardown the environment."""
        pass

    def step(self, context: ctx.SharedContext, cal: Calendar) -> None:
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


# 4. Define a HeatRiskBehaviorEngine Class
class HeatRiskBehaviorEngine(BehaviorEngine):
    """HeatRiskBehaviorEngine class"""

    def __init__(self, person: Person):
        """Constructor for the HeatRiskBehaviorEngine class."""
        super().__init__(person)

    def decide(self, context: ctx.SharedContext, cal: Calendar) -> None:
        """Decide what to do."""
        # TODO (2025-02-26 jcline): implement this method
        #   - This is where the person decides what to do
        #   - This could be based on the heat index, probability of a heat event,
        #     or other factors
        #   - For now, we will return False

        # get the heat index at the person's location
        places_proj = context.get_projection("places_projection")
        place = places_proj.get_place_for_agent(self.agent)
        if not place:
            logger.error(f"Person {self.agent.id} does not have a place")
            return

        environment = Model.get_model().get_environment()
        environment_values = environment.get_values_at_place(place)
        localHeatIndex = environment_values.heatIndexIndoors
        if self.agent.state.outside_worker:
            localHeatIndex = environment_values.heatIndex

        self.agent.state.heat_indices.append(localHeatIndex)

        # update the probability of a heat event
        self.agent.state.prob_heat_event = self.compute_prob_heat_event(environment.heat_threshold)

        if self.agent.state.prob_heat_event > 0.0001:
            lat = place.data.latitude
            lon = place.data.longitude
            logger.debug(
                f"Person {self.agent.id} at ({lat}, {lon}) has a probability of a heat event of {self.prob_heat_event}"
            )

            local_places = places_proj.get_local_places()
            candidates = find_closest_location(lat, lon, local_places, n=3, filter_func=if_place_has_cooling_center)
            for place, distance in candidates:
                logger.debug(f"  closest cooling center {place.id} at {distance} km")

            if self.consider_to_seek_cooling():
                logger.debug(f"Person {self.agent.id} decided to seek cooling")
                self.move_to_cooling_center(candidates[0][0], cal.hour_of_day)

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

    def consider_to_seek_cooling(self) -> bool:
        """Decide to seek cooling."""
        # TODO (2025-02-26 jcline): implement this method
        #   - This is where the person decides to seek cooling
        #   - This could be based on the heat index, probability of a heat event,
        #     or other factors
        #   - Also looking at the place data for cooling centers and
        #     air conditioning - looking for the three closest places
        #   - For now, we will return False
        return False

    def decide_to_seek_cooling(self) -> bool:
        """Decide to seek cooling."""
        # TODO (2025-02-26 jcline): implement this method
        return False

    def move_to_cooling_center(self, place: Place, current_hour: int) -> None:
        """Move the person to a cooling center."""

        # find the next hour
        next_hour = current_hour + 1
        start_time = next_hour * 60

        # currently set end time to the end of the day
        end_time = 1440  # 24 hours

        # find the activity and schedule indices
        activity_names = SIModel.get_activity_names()
        activity_id = activity_names.index("cooling_center")

        person = self.agent

        schedule_names = [schedule.name for schedule in person.schedules.schedules]
        schedule_idx = schedule_names.index("cooling_center")
        if self.agent.state.activities_idx == schedule_idx:
            logger.debug(f"Person {person.id} is already in the cooling center schedule")
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


# 6. Define the HeatRiskModel class
class HeatRiskModel(SIModel):
    """HeatRiskModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the HeatRiskModel class"""
        super().__init__(comm, params)

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def initialize_population(self) -> None:
        """Initialize population"""

        # register the environment
        SIModel.register_environment(HeatRiskEnvironment("HeatRiskEnvironment"))

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

        super().initialize_population()

        self._heat_threshold = 90.0
        # self._heat_threshold = float(self.params['heat_threshold'])

        # check the first agent
        person = next(self.context.agents())
        logger.debug(f"person={person}")
        # test_person_serialization(person)
        # test_activities(person)
        # test_add_move_to_cooling_center(person)
        # test_activities(person)

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            self.params["agent_log_file"],
            ["tick", "agent_id", "x", "y", "heatIndex", "hrsAboveHeatThreshold", "probHeatEvent"],  # , 'meet_count']
        )
        self.log_agents()

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day

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
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()


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
