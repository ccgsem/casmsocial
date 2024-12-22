""" Person Agent Base Class """
from __future__ import annotations
from repast4py import core
from repast4py.space import ContinuousPoint as cpt

from casmsocial.place import PlacesProjection
from casmsocial.calendar import Calendar
from casmsocial.activities import Schedules
from casmsocial.datautility import create_dataclass_record_from_dict
from casmsocial.message import Message

from dataclasses  import (
    astuple,
    dataclass,
    field
)
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type
)
from collections import deque

import numpy as np
import math

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


# 1. Define a PersonData Class
@dataclass(slots=True)
class PersonData:
    """Data for a Person."""
    person_id: int
    place_id: int
    activity_id: int
    places: List[int]


@dataclass
class ChiSimPersonData:
    person_id: int
    act_type: int
    place_id: int
    places: List[int]


# 2. Define a Person Class
# @dataclass(slots=True)
class Person(core.Agent):
    TYPE = 0  # class variable
    __personDataClass = Type[dataclass]  # class variable

    @classmethod
    def registerPersonDataClass(cls, persondataclass: Type[dataclass]) -> None:
        """Register Person dataclass."""
        cls.__personDataClass = persondataclass
    @classmethod
    def getPersonDataClass(cls) -> Type[dataclass]:
        """Returns Person dataclass."""
        return cls.__personDataClass

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
            initDict: A dictionary of initial values for the person
        """
        super().__init__(
            id=local_id,
            type=Person.TYPE,
            rank=rank)

        # `location` is currently referenced required but not used
        if 'x' not in initDict:
            initDict['x'] = 0
        if 'y' not in initDict:
            initDict['y'] = 0
        if math.isinf(initDict['x']) or math.isinf(initDict['y']):
            initDict['x'] = 0
            initDict['y'] = 0

        self.location = cpt(x=int(initDict['x']), y=int(initDict['y']), z=0)
        
        self.schedules: Schedules = schedules
                
        # map input parameters to dict
        initDict['person_id'] = local_id
        initDict['place_id'] = places[0]
        initDict['activity_id'] = 0
        # initDict['location'] = starting_location
        initDict['places'] = places

        self.state = \
            create_dataclass_record_from_dict(
                Person.getPersonDataClass(),
                initDict          
            )

        self.messages_outgoing: List[Message] = []
        self.messages_sent: List[Message] = []
        self.messages_incoming: List[Message] = []

        #print(f"Person {self.id} is ready!")

    @property
    def pt(self) -> cpt:
        """"""
        return self.location
    
    @property
    def currentPlaceID(self) -> int:
        return self.state.place_id
    
    @property
    def places(self) -> list[int]:
        return self.state.places

    def save(self) -> Tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person. 
        """ 
        return (
            self.uid,   # 0: uid is a tuple
            self.schedules.data(),    # 1: schedules is a Schedules object
            tuple(e for e in self.location.coordinates),  # 2: location
            astuple(self.state)  # 3: state is a PersonData object
            )

    def move(
            self,
            cal: Calendar,
            places_proj: PlaceProjection
        ) -> bool:
        """Move to the place indicated by the schedule for this tick.
        """
        success = False
        next_activity_id = self.selectNextPlace(cal)
        next_place_id = self.places[next_activity_id]

        if next_place_id == self.currentPlaceID:
            # already at the place
            # print(
            #     f"Agent {self.id} is already at place {self.currentPlaceID}")
            return True
        
        if not next_place_id:
            print(f"Agent {self.id} has no place to go - going remote.")
            print(f"places = {self.places}")
            print(f"schedule = {self.schedules}")
            next_place_id = 0  # reset to home

        place = places_proj.lookup_place(next_place_id) # place_map.get(next_place_id)
        if place is None:
            print(f"Place {next_place_id} not found.")
            print(f"places = {self.places}")
            return False

        if place is not None:
            success = True
            self.state.place_id = next_place_id
            print(
               f"Rank {rank}: "
               f"Agent {self.id} is moving to place {self.state.place_id}")
            places_proj.move_agent_to_place(self, place)
        else:
            print(f"move for act {next_activity_id} to place {next_place_id} failed.")
            print(f"places = {self.places}")
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
            if act.activity_id < len(self.places):
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
        # person_data: Tuple = (
        #     self.uid,
        #     self.schedules.data(),
        #     tuple(e for e in self.location.coordinates),
        #     astuple(self.state)

        # 0: uid is a tuple
        uid = person_data[0]
       
        schedules = Schedules.restore(person_data[1])

        pt_array = list(person_data[2])
        pt = cpt(pt_array[0], pt_array[1], 0)

        # person_data[3] is a PersonData object as tuple
        # the third element of the tuple is the places list
        places = person_data[3][3]

        person = Person(uid[0], uid[2], schedules, places, {})

        # restore the state
        person.state = Person.getPersonDataClass()(*person_data[3])

        return person
    
    def __str__(self):
        return (
            "Person: "
            f"id={self.id}, "
            f"pt={self.pt}, "
            f"currentPlaceID={self.currentPlaceID}, "
            f"schedules={self.schedules}, "
            f"state={self.state}"
        )


# 3. Define a PersonConfig NamedTuple
PersonConfig = NamedTuple(
    'PersonConfig',
    [
        ('name', str),
        ('type', Type[Person]),
        ('dataType', Type[PersonData])
    ]
)


# 4. test code
def test_person():
    """Test the Person class."""
    person_data = {
        'person_id': 1,
        'place_id': 0,
        'activity_id': 0,
        'places': [0, 1, 2]
    }

    person_data_class = dataclass(
        PersonData,
        frozen=True
    )

    Person.registerPersonDataClass(person_data_class)

    person = Person(
        local_id=1,
        rank=0,
        schedules=Schedules(),
        places=[0, 1, 2],
        starting_location=cpt(0, 0, 0),
        initDict=person_data
    )

    print(person)

    person_data = person.save()
    print(person_data)

    restored_person = Person.restore(person_data)
    print(restored_person)

    print("Person test passed.")

def test_activities(person: Person):
    schedules = person.schedules
    for act in schedules[0].acts:
        print(act)

if __name__ == "__main__":
    test_person()


person_cache = {}
person_id_map = {}
