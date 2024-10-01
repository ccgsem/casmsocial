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
# from repast4py.space import DiscretePoint as dpt

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

from casmsocial.person import Person
# from casmsocial.place import Place
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

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world
        # grid
        box = space.BoundingBox(
            0,
            params['world.width'],
            0, params['world.height'],
            0,
            0
        )
        # create a SharedGrid of 'box' size with sticky borders that allows
        # multiple agents in each grid location.
        self.grid = space.SharedGrid(
            name='grid',
            bounds=box,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Multiple,
            buffer_size=2,
            comm=comm
        )
        self.context.add_projection(self.grid)

        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        # load $OMMUNITYSIM_DATA_PATH from .env
        load_dotenv(find_dotenv())
        data_input_path = os.environ.get("CASMSOCIAL_DATA_PATH")

        if not data_input_path:
            data_input_path = pathlib.Path.cwd()
        else:
            data_input_path = pathlib.Path(data_input_path)

        rank = comm.Get_rank()
        self.steps_per_day = int(params['steps.per.day'])
        self.cal = Calendar()

        # place_map is a dict of placeID->place object
        # local_places is a list of place objects "located" on this process
        self.place_map, self.local_places = ModelSetup.initPlaces(
            rank,
            data_input_path / params['household.file'],
            data_input_path / params['school.file'],
            data_input_path / params['work.file'],
            self.grid,
        )

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
            self.grid,
            self.rng)

        print(F"rank {rank}: number of person agents={len(self.agent_id_map)}")

        # for i in range(params['person.count']):
        #     # get a random x,y location in the grid
        #     pt = self.grid.get_random_local_pt(rng)
        #     # create and add the walker to the context
        #     personSchedule = Schedule(1, [0])
        #     person = Person(i, rank, personSchedule, [0], pt)
        #     self.context.add(person)
        #     self.grid.move(person, pt)

        # pt = self.grid.get_random_local_pt(rng)
        # place = Place(rank, pt)
        # self.place_map = { rank: place }

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            comm,
            params['agent_log_file'],
            ['tick', 'agent_id', 'agent_uid_rank']
        )  # , 'meet_count'])

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
        #     person.count_colocations(self.grid, self.meet_log)
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

    def movePersons(self):
        """Move all persons"""
        # to_move = []
        # next_place = Place()

        for person in self.context.agents():
            pass

    def step(self) -> None:
        tick = self.runner.schedule.tick

        self.cal.increment()

        print(
            "Step on "
            f"day {self.cal.day_of_year}, "
            f"hour {self.cal.hour_of_day}, "
            f"minute {self.cal.minute_of_day}"
        )

        # for person in self.context.agents():
        #     print(
        #         f"Agent {person.id} is at place {person.currentPlaceID} "
        #         f"at tick {tick}."
        #     )
        #     person_data = person.save()
        #     print(f"agent id = {person_data[0]}")
        #     print(f"activities = {person_data[1]}")
        #     print(f"places = {person_data[2]}")
        #     print(f"pt = {person_data[3]}")
        #     print(f"currentPlaceID = {person_data[4]}")
        #     print(f"risk = {person_data[5]}")

        tick = self.cal.minute_of_day
        countOfBadMoves = 0

        for person in self.context.agents():
            result = person.move(self.cal, self.grid, self.place_map)
            if not result:
                countOfBadMoves += 1

        self.context.synchronize(Person.restore)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

        for person in self.context.agents():
            person.step(self.cal)

        for place in self.local_places:
            place.step(self.cal, self.rng)

        print(f"number of bad moves = {countOfBadMoves}")

        # for person in self.context.agents():
        #     person.count_colocations(self.grid)

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

    def log_agents(self) -> None:
        tick = self.runner.schedule.tick
        for person in self.context.agents():
            self.agent_logger.log_row(tick, person.id, person.uid_rank)
            # , person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()

    def start(self) -> None:
        self.runner.execute()
