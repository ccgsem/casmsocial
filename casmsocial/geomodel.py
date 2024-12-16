# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the GeoModel
"""
from mpi4py import MPI
from repast4py import (
    space,
    schedule
)
from repast4py import context as ctx
import repast4py
import random

from casmsocial.model import Model
from casmsocial.person import (
    Person,
    person_cache
)
from casmsocial.calendar import Calendar
from casmsocial.modelsetup import (
    ModelSetup
)
# note: place types are set by derived Model classes

from casmsocial.modelfactory import (
    register_casmsocial_model
)
from casmsocial.message import Message

from typing import (
    Dict,
    List
)
from collections import deque
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
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # start timer
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
            data_input_path / filename for filename in params['places.files']
        ]
        
        self.place_map, self.local_places = ModelSetup.createPlaces(
            self.rank,
            place_filenames,
            self.cspace
        )
        print(f"size of place_map = {len(self.place_map)}")

        # Send rank of local places to other ranks
        send_buffers = [[] for _ in range(self.size)]
        for place in self.local_places:
            for rank in range(self.size):
                msg = {
                    'place_id': place.id,
                    'rank': place.rank
                }
                if rank != self.rank:
                    send_buffers[rank].append(msg)

        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for msg in all_messages:
            place = self.place_map[msg['place_id']]
            place.rank = msg['rank']

        # activitiesMap is a dict of personID->Schedule object
        activitiesMap = ModelSetup.createActivities(
            data_input_path / params['activities.file']
        )

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons at each
        # place
        self.contact_map = {}
        if 'contact.file' in params:
            print("Loading contact file...")

            self.contact_map = ModelSetup.createContacts(
                data_input_path / params['contact.file']
            )
        else:
            print("Error: contact file not specified.")

        print(F"rank {rank}: contacts size={len(self.contact_map)}")

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        self.person_id_map = {}
        self.person_id_map = ModelSetup.createPersons(
            data_input_path / params['persons.file'],
            self.place_map,
            activitiesMap,
            rank,
            self.context,
            self.cspace,
            self.rng)

        print(F"rank {rank}: number of person agents={len(self.person_id_map)}")

        self.data_input_path = data_input_path

        saved = []
        for p in self.context.agents():
            print(
                f"{p}, schedules={p.schedules}, places={p.places}, "
                f"pt={p.pt}, currentPlaceID={p.currentPlaceID}"
                f"state={p.state}"
            )
            result = p.save()
            print(result)
            saved.append(result)
            if len(saved) > 0:
                break

        restored = []
        for i in saved:
            p = Person.restore(i)
            restored.append(p)
            print(
                f"{p}, schedules={p.schedules}, places={p.places}, "
                f"pt={p.pt}, currentPlaceID={p.currentPlaceID}"
                f"state={p.state}"
            )

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
        self.context.synchronize(Person.restore)

        self.get_local_ids()

        self.add_people_to_places()
        self.make_contacts(tick)

        self.update_environment()

        self.send_messages_between_agents()

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
            if person.id not in self.person_id_map:
                self.person_id_map[person.id] = person.uid

    def add_people_to_places(self) -> None:
        for person in self.context.agents():
            if person.state.place_id not in self.place_map:
                print(f"Person {person.id} has no place.")
                return
            self.place_map[person.state.place_id].addPerson(person)

    def make_contacts(self, tick) -> None:

        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # print(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.state.place_id)
            if not contactIDs:
                # print(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.state.place_id}.")
                continue

            contacts = []
            for contactID in contactIDs:
                contacts.append(
                    self.context.agent(self.person_id_map[contactID])
                )
            person.make_contacts(contacts)

    def send_messages_between_agents(self) -> None:
        """Send messages between agents."""
        messages_to_send: List[Message] = []
        remote_person_ids = []

        # send the first round of messages
        # Step 1: Send and receive messages
        agents = self.context.agents(shuffle=True)
    
        for person in agents:

            recipient = random.choice(agents)

            if person.id != recipient.id:  # Avoid sending to self
                message = person.create_message(
                    recipient= recipient.id,
                    message=f"Hello from {person.uid}",
                    timestamp=(
                        "Step on "
                        f"day {self.cal.day_of_year}, "
                        f"hour {self.cal.hour_of_day}, "
                        f"minute {self.cal.minute_of_day}"
                    )
                )

            messages = person.send_messages()
            if len(messages) > 0:
                print(f"Person {person} has messages.")

                for message in messages:
                    recipient = message.recipient
                    if recipient in self.person_id_map:  # message to local person
                        recipient_uid = self.person_id_map[recipient]
                        print(f"Message from {message.sender} to {person_cache[recipient_uid].state}:")
                        person_cache[recipient_uid].receive_message(message)
                    else:   # message to remote person
                        print(f"Message from {message.sender} to {recipient}:")

                        # get remote person ID
                        remote_person_ids.append(recipient)

                        # create message to send to other rank
                        message_to_send = message
                        message_to_send.recipients = [recipient]
                        messages_to_send.append(message_to_send)
                        
            else:
                print(f"Person {person.id} has no messages.")
                continue

        # Exchange messages between processors
        all_messages = \
            self.exchange_messages(
                remote_person_ids,
                messages_to_send
            )

        # Step 2: Deliver messages from remote processors
        for message in all_messages:
            recipient = message.recipient
            if recipient in self.person_id_map:
                recipient_uid = self.person_id_map[recipient]
                print(
                    f"Remote message from {message.sender} to "
                    f"{person_cache[recipient_uid].state}:"
                )
                person_cache[recipient_uid].receive_message(message)
            else:
                print(f"Remote message from {message.sender} to {recipient} not delivered")
        
        # Step 3: Process messages
        for person in agents:
            person.process_messages()
    
    def get_remote_person_id_map(
            self,
            remote_person_ids: List[int]) -> Dict[int, int]:
        """Get the remote person ID map."""
        remote_person_id_map = {}

        # 1. Send request for remote person IDs to other ranks
        send_buffers = [[] for _ in range(self.size)]
        for person_id in remote_person_ids:
            for rank in range(self.size):
                if rank != self.rank:
                    send_buffers[rank].append(person_id)

        # 2. Receive remote person IDs from other ranks and map and send UIDs      
        received_buffers = self.comm.alltoall(send_buffers)
        send_buffers = [[] for _ in range(self.size)]  # reset send_buffers

        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for person_id in all_messages:
            if person_id in self.person_id_map:
                for rank in range(self.size):
                    if rank != self.rank:
                        person_uid = self.person_id_map[person_id]
                        msg = {'id': person_id, 'uid': person_uid}
                        send_buffers[rank].append(msg)

        # 3. Send remote person ID->UID map to other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for msg in all_messages:
            remote_person_id_map[msg['id']] = msg['uid']

        return remote_person_id_map

    def exchange_messages(
            self,
            remote_person_ids: List[int],
            messages_to_send: list[Message]
        ) -> List[Message]:
        """Exchange messages between processors using MPI."""
        remote_person_id_map = self.get_remote_person_id_map(remote_person_ids)

        # Step 1: Send messages with recipients mapped to other ranks
        send_buffers = [[] for _ in range(self.size)]
        for msg in messages_to_send:
            recipient_id = msg.recipient
            if recipient_id in remote_person_id_map:
                recipient_uid = remote_person_id_map[recipient_id]
                recipient_rank = recipient_uid.rank
            send_buffers[recipient_rank].append(msg)

        # Step 2: Receive messages from other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        return all_messages

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
