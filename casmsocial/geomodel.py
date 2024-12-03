# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the GeoModel
"""
from repast4py import (
    space,
    schedule,
    logging
)
from repast4py import context as ctx
import repast4py

from casmsocial.model import Model
from casmsocial.person import Person
from casmsocial.place import (
    PlaceData,
    PlaceConfig,
    Places
)
from casmsocial.calendar import Calendar
from casmsocial.modelsetup import (
    ModelSetup
)
# place types
from casmsocial.household import Household
from casmsocial.work import Work
from casmsocial.school import School
from casmsocial.modelfactory import (
    register_casmsocial_model
)

from typing import Dict
from collections import deque
from mpi4py import MPI
from dotenv import (
    find_dotenv,
    load_dotenv
)
import os
import pathlib

import pandas as pd
import time


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
    The GeoModel class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    The GeoModel class is a subclass of the Model class, which is an abstract
    base class that defines the interface for all models in the casmsocial.
    The GeoModel class implements the start and step methods, which are called
    by the run function in the casmsocial module to start and run the model.

    The GeoModel class adds the following functionality to the Model class:

    - The GeoModel class initializes geographic places and agents.
    - The GeoModel class updates the  environment for the current time step.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        # start timer
        # self.timer = repast4py.Timer()
        # self.timer.start()
        self.start_time = time.time()

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
            rank,
            self.context,
            self.cspace,
            self.rng)

        print(F"rank {rank}: number of person agents={len(self.agent_id_map)}")

        self.data_input_path = data_input_path

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

        self.movePersons()
        # self.context.synchronize(Person.restore)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

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
            if person.currentPlaceID not in self.place_map:
                print(f"Person {person.id} has no place.")
                return
            self.place_map[person.currentPlaceID].addPerson(person)

    def make_contacts(self, tick) -> None:

        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # print(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.currentPlaceID)
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
        pass

    def log_agents(self) -> None:
        """Log the agents at the current time step."""
        pass

    def at_end(self) -> None:
        """Actions to take at the end of the simulation."""
        pass

    def start(self) -> None:
        self.runner.execute()
        self.at_end()
        end_time = time.time()

        print(f"Simulation took {end_time - self.start_time} seconds.")
