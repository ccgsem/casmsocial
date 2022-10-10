from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt

import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

from Human import Human, restoreHuman
from Schedule import Schedule
from Place import Place
from Calendar import Calendar

from InitMethods import initHumans, initPlaces, initSchedules, initContacts

class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_repeating_event(1.1, 10, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)

        rank = comm.Get_rank()
        self.steps_per_day = int(params['steps.per.day'])
        self.cal = Calendar(self.steps_per_day)

        scheduleMap = initSchedules(params['activity.file'])
        
        # place_map is a dict of placeID->place object
        # local_places is a list of place objects "located" on this process
        self.place_map, self.local_places = initPlaces(
            rank,
            params['household.file'],
            params['school.file'],
            params['work.file'],
            self.grid
            )

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons at each place
        self.contact_map = initContacts(params['contact.file'])

        self.rng = repast4py.random.default_rng
        
        # agent_id_map is a map of personID->repast4py.Agent.uid
        self.agent_id_map = {}
        self.agent_id_map = initHumans(params['person.file'], self.place_map, scheduleMap, rank, self.context, self.grid, rng)
        
        # for i in range(params['human.count']):
        #     # get a random x,y location in the grid
        #     pt = self.grid.get_random_local_pt(rng)
        #     # create and add the walker to the context
        #     humanSchedule = Schedule(1, [0])
        #     human = Human(i, rank, humanSchedule, [0], pt)
        #     self.context.add(human)
        #     self.grid.move(human, pt)

        # pt = self.grid.get_random_local_pt(rng)
        # place = Place(rank, pt)
        # self.place_map = { rank: place }

        # initialize the logging
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'], ['tick', 'agent_id', 'agent_uid_rank'])#, 'meet_count'])

        # self.meet_log = MeetLog()
        # loggers = logging.create_loggers(self.meet_log, op=MPI.SUM, names={'total_meets': 'total'}, rank=rank)
        # loggers += logging.create_loggers(self.meet_log, op=MPI.MIN, names={'min_meets': 'min'}, rank=rank)
        # loggers += logging.create_loggers(self.meet_log, op=MPI.MAX, names={'max_meets': 'max'}, rank=rank)
        # self.data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, params['meet_log_file'])

        # count the initial colocations at time 0 and log
        # for human in self.context.agents():
        #     human.count_colocations(self.grid, self.meet_log)
        # self.data_set.log(0)
        # self.meet_log.max_meets = self.meet_log.min_meets = self.meet_log.total_meets = 0
        self.log_agents()

    def step(self):
        tick = self.runner.schedule.tick
        self.cal.calendarStep()

        for human in self.context.agents():
            human.move(tick, self.grid, self.place_map)

        self.context.synchronize(restoreHuman)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

        for human in self.context.agents():
            human.step(self.cal)

        for place in self.local_places:
            place.step(self.cal)

        # for human in self.context.agents():
        #     human.count_colocations(self.grid)

        # self.data_set.log(tick)
        # clear the meet log counts for the next tick
        # self.meet_log.max_meets = self.meet_log.min_meets = self.meet_log.total_meets = 0

    def reset(self):
        for place in self.local_places:
            place.reset()

    def get_local_ids(self):
        for human in self.context.agents():
            if human.id not in self.agent_id_map:
                self.agent_id_map[human.id] = human.uid

    def add_people_to_places(self):
        for human in self.context.agents():
            self.place_map[human.currentPlaceID].addPerson(human)

    def make_contacts(self, tick):
        for human in self.context.agents():
            contactsLen = len(self.contact_map[human.id])

            cycledStep = tick % self.steps_per_day
            humansContactMap = self.contact_map[human.id]
            contactIDs = humansContactMap[cycledStep] if cycledStep in humansContactMap else []
            contacts = []
            for contactID in contactIDs:
                contacts.append(self.context.agent(self.agent_id_map[contactID]))
            human.make_contacts(contacts)

    def log_agents(self):
        tick = self.runner.schedule.tick
        for human in self.context.agents():
            self.agent_logger.log_row(tick, human.id, human.uid_rank)#, human.meet_count)

        self.agent_logger.write()

    def at_end(self):
        # self.data_set.close()
        self.agent_logger.close()

    def start(self):
        self.runner.execute()