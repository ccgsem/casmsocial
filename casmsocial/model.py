from repast4py import (
    # core,
    # random,
    space,
    schedule,
    logging
    # parameters
)
from repast4py import context as ctx
import repast4py

# import numpy as np
from typing import Callable, Dict
from collections import deque
from mpi4py import MPI
# from dataclasses import dataclass
from dotenv import (
    find_dotenv,
    load_dotenv
)
import os
import pathlib

import numpy as np
import pandas as pd
import geopandas as gpd

from abc import ABC, abstractmethod

from casmsocial.person import Person
from casmsocial.place import (
    Place,
    register_place_type,
    get_place_type_idx
)

from casmsocial.calendar import Calendar

from casmsocial.modelsetup import (
    ModelSetup
)

# place types
from casmsocial.household import Household
from casmsocial.work import Work
from casmsocial.school import School


class Model(ABC):
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        """Step the model forward one time step."""
        pass


# model factory implementation
__MODELS = {}


def get_models(        
) -> dict[str, Callable[[MPI.Intracomm, dict], Model]]:
    """
    Returns a dictionary of available models with their creators, must be
    callable with the following signature:
    model_creator(
        comm: MPI.Intracomm,
        params: dict) -> Model
    """
    return __MODELS


def register_casmsocial_model(
        model_type: str
        ) -> Callable[[MPI.Intracomm, dict], Model]:
    """
    Registers a model creator, must be callable with the following signature:
    model_creator(model_type: MPI.Intracomm, dict) -> Model

    Args:
    model_type - model type, must be a string

    Returns:
    decorator - a decorator to register the model creator
    """
    def decorator(fn):
        __MODELS[model_type] = fn
        return fn
    return decorator


# model creator
def get_casmsocial_model(
        model_type: str
        ) -> Callable[[MPI.Intracomm, dict], Model]:
    """
    Returns an casmsocial model creator, must be callable with the following
    signature:
    model_creator(model_type: MPI.Intracomm, dict) -> Model

    Args:
    model_type - model type, must be a string
    """

    if model_type not in __MODELS:
        #logger.info("Available models:")
        print("Available models:")
        for key in __MODELS.keys():
            #logger.info(key)
            print(key)
        raise ValueError(f"Unsupported model type: {model_type}")
    return __MODELS[model_type]


class ModelNotFoundError(Exception):
    """ exception if model not found """
    pass


# utility functions for heat-related computations
def filter_heat_indices(
    heat_indices: list[float],
    threshold: float
) -> list[float]:
    """Filter out all heat indices above the threshold."""
    exceeded = True
    return \
        [t for t in heat_indices if (exceeded := exceeded and  t > threshold)]


def compute_prob_heat_event(
    heat_indices: list[float],
    threshold: float
) -> float:
    """Compute the probability of a heat event."""
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


# register the 'casmsocial' model
@register_casmsocial_model('casmsocial_GeoModel')
def create_casmsocial_GeoModel(
    comm: MPI.Intracomm,
    params: dict
) -> Model:
    print("Registering casmsocial model")
    return GeoModel(comm, params)


class GeoModel(Model):
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_repeating_event(1.1, 10, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        # load $CASMSOCIAL_DATA_PATH from .env
        load_dotenv(find_dotenv())
        data_input_path = os.environ.get("CASMSOCIAL_DATA_PATH")

        if not data_input_path:
            data_input_path = pathlib.Path.cwd()
        else:
            data_input_path = pathlib.Path(data_input_path)
        
        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world
        
        # bounds for continuous space (bounds from 'hh_utm' bounding box)
        bb = params['world.bounding.box']
        print(f"world bounding box = {bb}")
        box = space.BoundingBox(
            xmin=bb[0],
            xextent=bb[1],
            ymin=bb[2],
            yextent=bb[3]
        )

        # create a SharedContext with the given bounding box
        self.cspace = space.SharedCSpace(
            name='space',
            bounds=box,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Multiple,
            tree_threshold=100,
            buffer_size=10,
            comm=comm
        )
        self.context.add_projection(self.cspace)

        rank = comm.Get_rank()
        self.steps_per_day = int(params['steps.per.day'])
        self.cal = Calendar()

        # register the place types
        register_place_type(Household)
        register_place_type(Work)
        register_place_type(School)

        person_places = ['sp_hh_id', 'sp_work_id', 'sp_school_id']

        # initialize the places
        #  - place_map is a dict of placeID->place object
        #  - local_places is a list of place objects "located" on this process
        place_filenames = [
            data_input_path / filename for filename in params['place.files']
        ] 
        self.place_map, self.local_places = ModelSetup.initPlaces(
            rank,
            place_filenames,
            self.cspace
        )
        print(f"size of place_map = {len(self.place_map)}")

        # activitiesMap is a dict of personID->Schedule object
        activitiesMap = ModelSetup.initActivities(
            data_input_path / params['activity.file']
        )

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons at each
        # place
        self.contact_map = {}
        if 'contact.file' in params:
            print("Loading contact file...")

            self.contact_map = ModelSetup.initContacts(
                data_input_path / params['contact.file']
            )
        else:
            print("Error: contact file not specified.")

        print(F"rank {rank}: contacts size={len(self.contact_map)}")

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        self.agent_id_map = {}
        self.agent_id_map = ModelSetup.initPersons(
            data_input_path / params['person.file'],
            self.place_map,
            activitiesMap,
            person_places,
            rank,
            self.context,
            self.cspace,
            self.rng)

        print(F"rank {rank}: number of person agents={len(self.agent_id_map)}")


        # saved = []
        # for p in self.context.agents():
        #     print(
        #         f"{p}, schedules={p.schedules.data()}, places={p.places}, "
        #         f"pt={p.pt}, currentPlaceID={p.currentPlaceID}"
        #     )
        #     result = p.save()
        #     print(result)
        #     saved.append(result)
        #     if len(saved) > 0:
        #         break

        # restored = []
        # for i in saved:
        #     p = Person.restore(i)
        #     restored.append(p)
        #     print(
        #         f"{p}, schedules={p.schedules.data()}, places={p.places}, "
        #         f"pt={p.pt}, currentPlaceID={p.currentPlaceID}"
        #     )

        # load environment file
        # heat_index_file_path = data_input_path / params['heat.index.file']
        self.heatindex_by_hour_place_file_path = \
            data_input_path / params['heatIndex.file']
        if self.heatindex_by_hour_place_file_path.exists():
            print(f"Loading heat map places from {self.heatindex_by_hour_place_file_path}")
        else:
            print(f"Error: Heat map places file {self.heatindex_by_hour_place_file_path} not found.")
            exit(1)

        self._heat_threshold = 90.0
        # self._heat_threshold = float(params['heat_threshold'])

        # initialize the heat threshold
        self.heat_indices = deque([float('nan')])

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            comm,
            params['agent_log_file'],
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

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def movePersons(self):
        """Move all persons"""
        # to_move = []
        # next_place = Place()
        countOfBadMoves = 0

        for person in self.context.agents():
            result = person.move(self.cal, self.cspace, self.place_map)
            if not result:
                countOfBadMoves += 1

        print(f"number of bad moves = {countOfBadMoves}")

    def step(self) -> None:
        """Step the model forward one time step."""
        tick = self.runner.schedule.tick

        self.cal.increment()

        print(
            "Step on "
            f"day {self.cal.day_of_year}, "
            f"hour {self.cal.hour_of_day}, "
            f"minute {self.cal.minute_of_day}"
        )

        self.update_environment()

        for person in self.context.agents():
            person.step(self.cal)
 
        self.log_agents()

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
            if person.id not in self.agent_id_map:
                self.agent_id_map[person.id] = person.uid

    def add_people_to_places(self) -> None:
        for person in self.context.agents():
            if person.state.currentPlaceID not in self.place_map:
                print(f"Person {person.id} has no place.")
                return
            self.place_map[person.state.currentPlaceID].addPerson(person)

    def make_contacts(self, tick) -> None:

        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # print(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.state.currentPlaceID)
            if not contactIDs:
                # print(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.currentPlaceID}.")
                continue

            contacts = []
            for contactID in contactIDs:
                contacts.append(
                    self.context.agent(self.agent_id_map[contactID])
                )
            person.make_contacts(contacts)

    def update_environment(self) -> None:
        """Update the environment for the current time step."""
        tick = self.cal.minute_of_day

        self.movePersons()
        # self.context.synchronize(Person.restore)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

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
        countOfHeatIncidents = 0
        countOfAirConditionedPlaces = 0
        countOfOutsideWorkers = 0

        for place in self.local_places:

            place.step(self.cal, self.rng)
                
            if place.id in heatIndex_map:
                place.heatIndex = heatIndex_map[place.id]
                countOfHeatIndexMatches+=1
            else:
                place.heatIndex = meanheatindex

            # Take air conditioned places as 72 degrees    
            if place.AIR:
                countOfAirConditionedPlaces += 1
                localHeatIndex = 72
            else:
                localHeatIndex = place.heatIndex

            # if len(place.peopleAtPlace) > 0:
            #     print(f"place {place.id} has {len(place.peopleAtPlace)} people")

            for person in place.peopleAtPlace:

                # adjust the heat index for outside workers
                personHeatIndex = localHeatIndex
                if person.state.outside_worker:
                    countOfOutsideWorkers += 1
                    personHeatIndex = place.heatIndex

                person.state.heatIndices.appendleft(personHeatIndex)

                person.state.probHeatEvent = compute_prob_heat_event(
                    person.state.heatIndices,
                    self.heat_threshold
                )
                if person.state.probHeatEvent > 0.0001:
                    countOfHeatIncidents += 1

        print(f"number of heat index matches = {countOfHeatIndexMatches}")
        print(f"number of heat incidents = {countOfHeatIncidents}")
        print(f"number of air conditioned places = {countOfAirConditionedPlaces}")
        print(f"number of outside workers = {countOfOutsideWorkers}")

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
                person.state.location.x,
                person.state.location.y,
                person.state.heatIndices[0],
                len(heat),
                person.state.probHeatEvent
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()

    def start(self) -> None:
        self.runner.execute()
        self.at_end()