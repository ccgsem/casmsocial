"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""

import math
from collections import deque
from dataclasses import dataclass, field
from heapq import nsmallest

import pandas as pd
from mpi4py import MPI
from repast4py import logging
from repast4py.space import ContinuousPoint as cpt  # noqa: F401

from casmsocial.activities import Act, Schedules
from casmsocial.datautility import get_attribute_names_from_data

# model factory
from casmsocial.factory import Models

# place types
from casmsocial.household import Household
from casmsocial.model import Model
from casmsocial.person import Person, PersonConfig, PersonData
from casmsocial.place import Place, PlaceConfig, PlaceData, RemotePlace
from casmsocial.school import School
from casmsocial.socialmodel import SIModel, update_activities_data
from casmsocial.workplace import Workplace


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in km

def find_closest_cooling_centers(lat, lon, places, n=3):
    """
    Find the `n` closest cooling centers to the given coordinates.

    Args:
        lat (float): Latitude of the target location.
        lon (float): Longitude of the target location.
        places (list[Place]): List of Place objects.
        n (int): Number of closest cooling centers to return (default is 3).

    Returns:
        list[tuple[Place, float]]: List of tuples containing Place objects and their distances.
    """
    cooling_centers = [p for p in places if getattr(p.data, 'cooling_center', False) and p.data.cooling_center > 0.0]

    if not cooling_centers:
        return []  # No cooling centers found

    closest_places = nsmallest(n, cooling_centers, key=lambda p: haversine_distance(lat, lon, p.data.latitude, p.data.longitude))
    return [(place, haversine_distance(lat, lon, place.data.latitude, place.data.longitude)) for place in closest_places]


# 1. utility functions for heat-related computations
def filter_heat_indices(
    heat_indices: list[float],
    threshold: float
) -> list[float]:
    """Filter out all heat indices above the threshold."""
    exceeded = True
    return \
        [t for t in heat_indices if (exceeded := exceeded and  t > threshold)]


# 2. Define a PlaceData Class
@dataclass
class PlaceDataWithClimate(PlaceData):
    """Place with heat index data."""
    heatIndex: float = float('nan')
    heatIndexIndoors: float = float('nan')
    AIR: bool = False
    cooling_center: float = 0.0 #bool = False


# 3. Define a PersonData Class with heat risk data
@dataclass
class PersonDataWithHeatRisk(PersonData):
    """Data for a Person."""
    outside_worker: bool = False
    heatIndices: deque = field(default_factory=lambda: deque([float('nan')]))
    probHeatEvent: float = 0.0


# 4. Define a PersonWithHeatRisk Class
class PersonWithHeatRisk(Person):
    """Person with heat risk data."""
    def __init__(
        self,
        local_id: int,
        rank: int,
        schedules: Schedules,
        places: list[int],
        initDict: dict
    ):
        """Constructor for the PersonWithHeatRisk class."""
        super().__init__(
            local_id,
            rank,
            schedules,
            places,
            initDict)

    def compute_prob_heat_event(
        self,
        threshold: float
    ) -> float:
        """Compute the probability of a heat event."""
        heat_indices = self.state.heatIndices

        # filter out all heat indices above the threshold
        heat_index = heat_indices[0]
        heat = filter_heat_indices(heat_indices, threshold)
        hours_above_threshold = len(heat)

        # note: length of heat is the number of hours above the threshold
        # prob_heat_event = \
        #     1 - (1 - ((heat_indices[0] - threshold/80.0) ** 2) ** (3 * len(heat)))
        prob_heat_event = \
            1 - (1 - ((heat_index - threshold)/80.0) ** 2) ** (3*hours_above_threshold)
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
        activity_id = activity_names.index('cooling_center')

        schedule_names = [schedule.name for schedule in self.schedules.schedules]
        schedule_idx = schedule_names.index('cooling_center')
        if self.state.activities_idx == schedule_idx:
            print(f"Person {self.id} is already in the cooling center schedule")
            return

        # need to modify the person's activities to include the cooling
        activities_data = self.state.places
        self.state.places = update_activities_data(activities_data, cooling_center=place.id)

        act_go_to_cooling_center = \
            Act(
                self.id,
                activity_id,
                1.0, start_time, end_time)

        # add the activity to the schedule
        self.schedules.schedules[schedule_idx].addAct(act_go_to_cooling_center)
        print(f"Person {self.id} updated schedule to move to cooling center {place.id}")
        for act in self.schedules.schedules[schedule_idx].acts:
            print(f"  {act}")
        previous_schedule = self.state.activities_idx
        self.state.activities_idx = schedule_idx
        print(f"Person {self.id} moved to schedule {schedule_idx} from {previous_schedule}")

    def step(self) -> None:
        """Step the person forward one time step."""
        super().step()

        model = Model.get_model()
        if model is not None:
            place = model.places_proj.get_place_for_agent(self)
        else:
            print("model is unavailable!")
            return

        if not place:
            print(f"=====>Person {self.id} is without a place!")
            return

        # update the heat index for the person
        localHeatIndex = place.data.heatIndexIndoors
        if self.state.outside_worker:
            localHeatIndex = place.data.heatIndex

        self.state.heatIndices.appendleft(localHeatIndex)

        # update the probability of a heat event
        self.state.probHeatEvent = \
            self.compute_prob_heat_event(
                model.heat_threshold
            )

        if self.state.probHeatEvent > 0.0001:

            # find the closest cooling centers
            current_hour = model.cal.hour_of_day
            lat = place.data.latitude
            lon = place.data.longitude
            print(f"Person {self.id} at ({lat},{lon}) is experiencing a heat event probability of {self.state.probHeatEvent} at hour {current_hour}")
            local_places = model.places_proj.get_local_places()
            candidates = find_closest_cooling_centers(lat, lon, local_places, n=3)
            print(f"Person {self.id} is considering the following cooling centers:")
            selection = None
            for place, distance in candidates:
                print(f"  {place.id} at ({place.data.latitude},{place.data.longitude}) is {distance} km away")
                if selection is None:
                    selection = place

            if selection is not None:
                print(f"Person {self.id} is moving to cooling center {selection.id}")
                self.move_to_cooling_center(selection, current_hour)

            if self.consider_to_seek_cooling():
                print(f"Person {self.id} is seeking cooling")
                if self.decide_to_seek_cooling():
                    pass


# 6. Define the HeatRiskModel class
class HeatRiskModel(SIModel):
    """ HeatRiskModel class """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: dict
    ):
        """ Constructor for the HeatRiskModel class """
        super().__init__(comm, params)

        # load environment file
        # heat_index_file_path = data_input_path / self.params['heat.index.file']
        self.heatindex_by_hour_place_file_path = \
            self.data_input_path / self.params['heatIndex.file']
        if self.heatindex_by_hour_place_file_path.exists():
            print(f"Loading heat map places from {self.heatindex_by_hour_place_file_path}")
        else:
            print(f"Error: Heat map places file {self.heatindex_by_hour_place_file_path} not found.")
            exit(1)

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def initializePopulation(self) -> None:
        """Initialize population"""

        # register the place types
        SIModel.register_place_config(
            PlaceConfig(
                name='Household',
                type=Household,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_hh_id'
            )
        )
        SIModel.register_place_config(
            PlaceConfig(
                name='Workplace',
                type=Workplace,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_work_id'
            )
        )
        SIModel.register_place_config(
            PlaceConfig(
                name='School',
                type=School,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_school_id'
            )
        )

        # register the remote place type
        SIModel.register_remote_place_config(
            PlaceConfig(
                name='RemotePlace',
                type=RemotePlace,
                dataType=PlaceData,
                personPlaceField=''
            )
        )

        # register the person type
        SIModel.register_person_config(
            PersonConfig(
                name='Person',
                type=PersonWithHeatRisk,
                dataType=PersonDataWithHeatRisk
            )
        )

        # register the activities
        SIModel.register_planned_activity_names(
            ['sp_hh_id', 'sp_work_id', 'sp_school_id'])
        SIModel.register_activity_names(
            ['home', 'work', 'school', 'cooling_center'])

        print("Now running initialize population for SIModel...")
        super().initializePopulation()

        self._heat_threshold = 90.0
        # self._heat_threshold = float(self.params['heat_threshold'])

        # initialize the heat threshold
        self.heat_indices = deque([float('nan')])

        # check the first agent
        person = next(self.context.agents())
        print(f"person={person}")
        # test_person_serialization(person)
        # test_activities(person)
        # test_add_move_to_cooling_center(person)
        # test_activities(person)

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            self.params['agent_log_file'],
            [
                'tick',
                'agent_id',
                'x',
                'y',
                'heatIndex',
                'hrsAboveHeatThreshold',
                'probHeatEvent'
            ]  # , 'meet_count']
        )
        self.log_agents()

    def update_environment(self) -> None:
        """Update the environment for the current time step."""
        super().update_environment()

         # update the heat indices
        heatindex_by_hour_place = \
            pd.read_parquet(
                self.heatindex_by_hour_place_file_path,
                engine='pyarrow',
                filters=[("time_hour", "=", self.cal.hour_of_day)]
            ).loc[:, ['sp_id', 'heatIndex']].dropna()

        # print(f"size of heatindex_by_hour_place = {len(heatindex_by_hour_place)}")
        minheatindex = heatindex_by_hour_place['heatIndex'].min()
        maxheatindex = heatindex_by_hour_place['heatIndex'].max()
        meanheatindex = heatindex_by_hour_place['heatIndex'].mean()
        print(
            f"min heat index = {minheatindex}, "
            f"max heat index = {maxheatindex}, "
            f"mean heat index = {meanheatindex}")

        heatIndex_map = \
            heatindex_by_hour_place.set_index('sp_id')['heatIndex'].to_dict()

        # metrics
        countOfHeatIndexMatches = 0
        # countOfHeatIncidents = 0
        countOfAirConditionedPlaces = 0
        # countOfOutsideWorkers = 0

        #for place in self.local_places:
        local_places = self.places_proj.get_local_places()
        print(f"number of local places = {len(local_places)}")
        places_with_cooling_center = [place for place in local_places if getattr(place.data, 'cooling_center', False) and place.data.cooling_center > 0.0]

        print(f"number of places with cooling centers = {len(places_with_cooling_center)}")

        # for place in places_with_cooling_centers:
        #     print(f"place {place.id} has place.data={place.data}")

        for place in local_places:

            # if "cooling_center" in get_attribute_names_from_data(place.data):
            #     if place.data.cooling_center > 0.0:
            #         print(f"place {place.id} is a cooling center")

            # update the heat index for the place
            place.step()

            if place.id in heatIndex_map:
                place.data.heatIndex= heatIndex_map[place.id]
                countOfHeatIndexMatches+=1
            else:
                place.data.heatIndex= meanheatindex

            # Take air conditioned places as 72 degrees and non-air conditioned
            #  places as the heat index
            if 'AIR' in get_attribute_names_from_data(place.data) and place.data.AIR:
                countOfAirConditionedPlaces += 1
                place.data.heatIndexIndoors = 72
            else:
                place.data.heatIndexIndoors = place.data.heatIndex

        #     localHeatIndex = place.data.heatIndex

        #     peopleAtPlace = self.places_proj.get_agents_at_place(place)
        #     # if len(peopleAtPlace) > 0:
        #     #     print(f"place {place.id} has {len(peopleAtPlace)} people")
        #     for person in peopleAtPlace:

        #         # adjust the heat index for outside workers
        #         personHeatIndex = localHeatIndex
        #         if person.state.outside_worker:
        #             countOfOutsideWorkers += 1
        #             personHeatIndex = place.data.heatIndex

        #         person.state.heatIndices.appendleft(personHeatIndex)

        #         person.state.probHeatEvent = compute_prob_heat_event(
        #             person.state.heatIndices,
        #             self.heat_threshold
        #         )
        #         if person.state.probHeatEvent > 0.0001:
        #             countOfHeatIncidents += 1

        # print(f"number of heat index matches = {countOfHeatIndexMatches}")
        # print(f"number of heat incidents = {countOfHeatIncidents}")
        # print(f"number of air conditioned places = {countOfAirConditionedPlaces}")
        # print(f"number of outside workers = {countOfOutsideWorkers}")

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day

        for person in self.context.agents():
            heat = filter_heat_indices(
                person.state.heatIndices,
                self.heat_threshold)
            self.agent_logger.log_row(
                tick,
                person.id,
                person.pt.x,
                person.pt.y,
                person.state.heatIndices[0],
                len(heat),
                person.state.probHeatEvent
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()


# Register HeatRiskModel
Models.add_model(
    HeatRiskModel.__module__ + '.' + HeatRiskModel.__name__,
    HeatRiskModel)


# 7. create test functions
def test_add_move_to_cooling_center(person: Person):
    """Test the add_move_to_cooling_center function."""
    print(f"Testing add_move_to_cooling_center for person {person.id}")
    # TODO (2025-02-26 jcline): implement this function
    #   - This is where the person is moved to a cooling center
    #   - Find the closest cooling center
    #   - Schedule moving the person to the cooling center
    current_hour = 12
    next_hour = current_hour + 1
    start_time = next_hour * 60
    end_time = 1440  # 24 hours

    activity_names = SIModel.get_activity_names()
    activity_id = activity_names.index('cooling_center')

    schedule_names = [schedule.name for schedule in person.schedules.schedules]
    schedule_idx = schedule_names.index('cooling_center')

    # need to modify the person's activities to include the cooling
    activities_data = person.state.places
    person.state.places = update_activities_data(activities_data, cooling_center=10)

    act_go_to_cooling_center = \
        Act(
            person.id,
            activity_id,
            1.0, start_time, end_time)

    # add the activity to the schedule
    person.schedules.schedules[schedule_idx].addAct(act_go_to_cooling_center)

