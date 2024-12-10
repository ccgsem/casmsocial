""" Person Agent Base Class """
from __future__ import annotations
from repast4py import core
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt

from dataclasses  import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type
)
from collections import deque

from casmsocial.calendar import Calendar
from casmsocial.activities import(
    Activities,
    Schedules
)
from casmsocial.message import Message

import numpy as np

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


@dataclass(slots=True)
class PersonData():
    """Data for a Person."""
    person_id: int
    place_id: int
    activity_id: int
    location: cpt
    places: List[int]
    outside_worker: bool
    heatIndices: deque
    probHeatEvent: float


# @dataclass(slots=True)
class Person(core.Agent):
    TYPE = 0  # class variable
    personDataType = Type[dataclass]  # class variable

    # schedules: Optional[Schedules] = \
    #     field(default=tuple[Activities(0, tuple[0, 0, 0])])
    # places: Optional[list[int]] = field(default_factory=list)
    # currentPlaceID: Optional[str] = field(default=None)

    def __init__(
        self,
        local_id: int,
        rank: int,
        schedules: Schedules,
        places: list[int],
        starting_location: cpt,
        initDict: Dict):
        """Constructor for the Person class.
        
        Arguments:
            local_id: The ID for this person on this process, combines with the
                rank to form a simulation-wide unique ID.
            rank: The rank of this process.
            schedules: An object containing a set of one or more
                activity sequences. Each activity sequence provides a schedule
                for this Person. The schedule is a list of activities with start
                and end times. Each activity has a place type (int). The place
                types are implementation-agnostic so "0" could mean "home" in
                one simulation or "grocery store" in another. If there are
                multiple activity sequences, one sequence could be for weekdays
                and another for weekends, for example.
            places: A list of place_id's that correspond to values coming out of 
                the schedule. e.g. if the schedule returns "0", this Person will 
                try to go to the place with the ID of places[0]
            starting_location: A ContinuousPoint for this Person's starting location 
                on a cspace projection. Set to null and override the move() function 
                if not using a cspace projection.
            initDict: A dictionary of initial values for the person
        """
        super().__init__(
            id=local_id,
            type=Person.TYPE,
            rank=rank)
        
        self.schedules: Schedules = schedules
                
        self.state = PersonData(
            person_id=local_id,
            place_id = places[0],
            activity_id=0,
            location=starting_location,
            places=places,
            outside_worker=bool(initDict.get('outside_worker', False)),
            heatIndices=deque([float('nan')]),
            probHeatEvent=0.0
        )

        self.messages_outgoing: List[Message] = []
        self.messages_sent: List[Message] = []
        self.messages_incoming: List[Message] = []

        #print(f"Person {self.id} is ready!")

    def save(self) -> Tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person. 
        """ 
        return (
            self.uid,   # 0: uid is a tuple
            self.schedules.data(),    # 1: schedules is a Schedules object
            self.state.person_id,   # 2: person_id
            self.state.place_id,  # 3: place_id
            self.state.activity_id,  # 4: activity_id
            tuple(e for e in self.state.location.coordinates),  # 5: location
            tuple(self.state.places),  # 6: convert list to tuple
            self.state.outside_worker,  # 7:  outside_worker
            tuple(self.state.heatIndices),   # 8: heatIndex
            self.state.probHeatEvent  # 9: probHeatEvent
            )

    def move(self, cal: Calendar, cspace, place_map: Dict) -> bool:
        """Move to the place indicated by the schedule for this tick.
        """
        success = False
        # next_activity_id = int(self.selectNextPlace(cal))
        next_activity_id = self.selectNextPlace(cal)
        next_place_id = self.state.places[next_activity_id]

        place = place_map.get(next_place_id)
        if place is None:
            next_place_id = 0  # reset to home

        if place is not None:
            success = True
            self.state.place_id = next_place_id
            # print(
            #    f"Rank {rank}: "
            #    f"Agent {self.id} is moving to place {self.state.place_id}")
            placeLocation = place.location
            self.state.location = cspace.move(self, placeLocation)
            self.state.location = placeLocation
        else:
            print(f"move for act {next_activity_id} to place {next_place_id} failed.")
            print(f"places = {self.state.places}")
            print(f"Remaining a currentPlaceID = {self.state.place_id}")

        return success
    
    def selectActivities(self, cal: Calendar) ->  int:
        """Select the activities for the time of day and day of week.
        """
        activities_idx = 0
        if not cal.is_weekday() and len(self.schedules) > 1:
            activities_idx = 1
        return activities_idx
    
    def selectNextPlace(
            self,
            cal: Calendar) -> int:
        """Select the next place to go to based on the schedule for time of
        day and day of week.
        """
        time = cal.minute_of_day

        activities_idx = self.selectActivities(cal)
        act = self.schedules[activities_idx].activityAt(time)

        next_activity_id = 0  # home is the default
        if act is not None:
            if act.activity_id < len(self.state.places):
                next_activity_id = int(act.activity_id)
            # else:  if the activity is not in the list of places, go home
 
        return next_activity_id

    def count_colocations(self, cspace):
        # subtract self
        num_here = cspace.get_num_agents(self.state.location) - 1
        print(f"Agent {self.id} sees {num_here} other agents.")
        # meet_log.total_meets += num_here
        # if num_here < meet_log.min_meets:
        #     meet_log.min_meets = num_here
        # if num_here > meet_log.max_meets:
        #     meet_log.max_meets = num_here
        # self.meet_count += num_here

    def make_contacts(self, contacts):
        pass
    
    def create_message(
            self,
            recipient: int,
            message: str,
            timestamp: str,
            metadata: Dict = {},
            attachments: Dict = {})->Message:
        """Create a message to send to other agents."""
        msg = Message(
            sender=self.uid,
            recipient=recipient,
            message=message,
            timestamp=timestamp,
            metadata=metadata,
            attachments=attachments
        )

        self.messages_outgoing.append(msg)

        return(msg)
    
    def send_messages(self)->List[Message]:
        """Send messages to other agents."""
        outgoing_messages = self.messages_outgoing
        self.messages_sent.extend(outgoing_messages)
        self.messages_outgoing = []

        return outgoing_messages

    def receive_message(self, message: Message)->bool:
        """Process received messages."""
        self.messages_incoming.append(message)

        return True

    def process_messages(self)->None:
        """modify the state of the person based on the messages received."""
        for msg in self.messages_incoming:
            # process the message
            print(f"Agent {self.id} received message: {msg.message}")
        
        self.messages_incoming = []
        self.messages_outgoing = []
    
    def step(self, calendar: Calendar):
        pass

    @classmethod
    def restore(cls, person_data: Tuple) -> Person:
        """Creates or updates a local person from person_data.

        Args:
            person_data: tuple containing the data returned by Person.save(). 
        """ 
        uid = person_data[0]

        pt_array = list(person_data[5])
        pt = cpt(pt_array[0], pt_array[1], 0)
        
        schedules = Schedules.restore(person_data[1])
        person = Person(uid[0], uid[2], schedules, person_data[6], pt)

        # person.state = PersonData(*person_data[2:])
        person.state.person_id = person_data[2]
        person.state.place_id = person_data[3]
        person.state.activity_id = person_data[4]
        # person.state.location = pt  # person_data[5]
        # person.places = person_data[6]

        person.state.outside_worker = person_data[7]
        person.state.heatIndices = deque(person_data[8])
        person.state.probHeatEvent = person_data[9]

        return person


person_cache = {}
person_id_map = {}
