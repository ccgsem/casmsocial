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
from typing import Dict
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

from casmsocial.person import Person
from casmsocial.place import (
    Place,
    get_place_type_idx
)

from casmsocial.calendar import Calendar

from casmsocial.modelsetup import (
    ModelSetup
    # initPersons, initPlaces, initActivities, initContacts
)


class Model(object):
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
        # load $OMMUNITYSIM_DATA_PATH from .env
        load_dotenv(find_dotenv())
        data_input_path = os.environ.get("CASMSOCIAL_DATA_PATH")

        if not data_input_path:
            data_input_path = pathlib.Path.cwd()
        else:
            data_input_path = pathlib.Path(data_input_path)
        
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

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world
        
        # bounds for continuous space (bounds from 'hh_utm' bounding box)
        box = space.BoundingBox(
            xmin=682382,  # remainder: .07291591
            xextent=65291,  # remainder:  .067068616976, params['world.width']
            ymin=3933088,  # remainder: .46059242
            yextent=61693 # remainder: .09425332444, params['world.height']
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

        # place_map is a dict of placeID->place object
        # local_places is a list of place objects "located" on this process
        self.place_map, self.local_places = ModelSetup.initPlaces(
            rank,
            data_input_path / params['household.file'],
            data_input_path / params['work.file'],
            data_input_path / params['school.file'],
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
        self.contact_map = ModelSetup.initContacts(
            data_input_path / params['contact.file']
        )

        print(F"rank {rank}: contacts size={len(self.contact_map)}")

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        self.agent_id_map = {}
        self.agent_id_map = ModelSetup.initPersons(
            data_input_path / params['person.file'],
            self.place_map,
            activitiesMap,
            rank,
            self.context,
            self.cspace,
            self.rng)

        print(F"rank {rank}: number of person agents={len(self.agent_id_map)}")

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            comm,
            params['agent_log_file'],
            ['tick', 'agent_id', 'x', 'y', 'heatIndex']  # , 'meet_count']
        )

        # self.meet_log = MeetLog()
        # loggers = \
        #     logging.create_loggers(
        #         self.meet_log,
        #         op=MPI.SUM,
        #         names={'total_meets': 'total'},
        #         rank=rank
        #     )
        # loggers += \
        #     logging.create_loggers(
        #         self.meet_log,
        #         op=MPI.MIN,
        #         names={'min_meets': 'min'},
        #         rank=rank
        #     )
        # loggers += \
        #     logging.create_loggers(
        #         self.meet_log,
        #         op=MPI.MAX,
        #         names={'max_meets': 'max'},
        #         rank=rank
        #     )
        # self.data_set = \
        #     logging.ReducingDataSet(
        #         loggers,
        #         MPI.COMM_WORLD,
        #         params['meet_log_file']
        #     )

        # count the initial colocations at time 0 and log
        # for person in self.context.agents():
        #     person.count_colocations(self.cspace, self.meet_log)
        # self.data_set.log(0)
        # self.meet_log.max_meets = \
        #     self.meet_log.min_meets = self.meet_log.total_meets = 0
        self.log_agents()

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
        self.hours_above_heat_threshold = 0

    def movePersons(self):
        """Move all persons"""
        # to_move = []
        # next_place = Place()

        for person in self.context.agents():
            pass

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

    def load_environment_parameters(self) -> None:
        """Load the environment parameters."""
        pass

    def update_environment(self) -> None:
        """Update the environment for the current time step."""
        tick = self.cal.minute_of_day

        countOfBadMoves = 0

        for person in self.context.agents():
            result = person.move(self.cal, self.cspace, self.place_map)
            if not result:
                countOfBadMoves += 1

        # self.context.synchronize(Person.restore)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

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
        if meanheatindex > self.get_heat_threshold():
            self.hours_above_heat_threshold += 1
        else:
            self.hours_above_heat_threshold = 0
        prob_heat_event = self.compute_prob_heat_event(
            meanheatindex,
            self.hours_above_heat_threshold
        )
        print(f"probability of heat event = {prob_heat_event}")

        heatIndex_map = heatindex_by_hour_place.set_index('sp_id')['heatIndex'].to_dict()

        countOfHeatIndexMatches = 0
        for place in self.local_places:

            place.step(self.cal, self.rng)
                
            if place.id in heatIndex_map:
                place.heatIndex = heatIndex_map[place.id]
                countOfHeatIndexMatches+=1
            else:
                place.heatIndex = meanheatindex

            for person in place.peopleAtPlace:
                person.state.heatIndex = place.heatIndex

        print(f"number of bad moves = {countOfBadMoves}")
        print(f"number of heat index matches = {countOfHeatIndexMatches}")

    def get_heat_threshold(self) -> float:
        return self._heat_threshold

    def compute_prob_heat_event(
        self,
        heat_index: float,
        hours_above_threshold: int
    ) -> float:
        """Compute the probability of a heat event."""
        prob_heat_event = 1 - (1 - ((heat_index - 90)/80)** 2) ** (3*hours_above_threshold)
        return prob_heat_event

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day
        for person in self.context.agents():
            self.agent_logger.log_row(
                tick,
                person.id,
                person.state.location.x,
                person.state.location.y,
                person.state.heatIndex
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()

    def start(self) -> None:
        self.runner.execute()
