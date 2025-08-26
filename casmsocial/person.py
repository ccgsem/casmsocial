""" Person Agent Base Class """
from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import NamedTuple

import repast4py.context as ctx
from loguru import logger
from mpi4py import MPI
from repast4py import core
from repast4py.space import ContinuousPoint as cpt

from casmsocial.activities import Schedules
from casmsocial.data_utilities import create_dataclass_record_from_dict
from casmsocial.message import Message
from casmsocial.sim_time import SimTime

rank = MPI.COMM_WORLD.Get_rank()


class BehaviorEngine:
    """
    A behavioral engine that governs the decision-making of a Person agent.
    """

    def __init__(self, agent: Person):
        self.agent = agent
        # self.social_network = social_network
        # self.environment = environment

    def decide(self, context: ctx.SharedContext, cal: SimTime):
        """
        Simulate decision-making based on the agent's attributes, social network, and environment.
        """
        pass
        # temperature = self.environment.get("temperature", 20)
        # precipitation = self.environment.get("precipitation", 0)
        # social_influence = sum(friend.income for friend in self.social_network) / len(self.social_network) if self.social_network else 0

        # if self.agent.age < 18:
        #     self.agent.current_action = "Studying"
        # elif self.agent.income + social_influence > 50000:
        #     self.agent.current_action = "Investing"
        # elif temperature < 0 and precipitation > 10:
        #     self.agent.current_action = "Staying Indoors"
        # else:
        #     self.agent.current_action = "Working"


# 1. Define a PersonData Class
@dataclass(slots=True)
class PersonData:
    """Data for a Person."""

    person_id: int
    place_id: int
    activity_id: int
    places: namedtuple
    activities_idx: int
    minute_last_moved: int = 0


@dataclass(slots=True)
class PersonLastKnownPlace:
    """Data for a Person's last known place.
    This is used to store the last known place of a person.
    """

    person_id: int
    place_id: int
    minute_last_updated: int = 0


@dataclass
class ChiSimPersonData:
    person_id: int
    act_type: int
    place_id: int
    places: namedtuple


# 2. Define a Person Class
# @dataclass(slots=True)
class Person(core.Agent):
    """Person class."""

    # class variables
    TYPE = 0  # class variable
    __person_data_class: type[dataclass] = PersonData
    __behavior_engine: BehaviorEngine = BehaviorEngine

    @classmethod
    def setPersonDataClass(cls, person_data_class: type[dataclass]) -> None:
        """Register a person data class for the person."""
        cls.__person_data_class = person_data_class

    @classmethod
    def getPersonDataClass(cls) -> type[dataclass]:
        """Get the person data class for the person."""
        return cls.__person_data_class

    @classmethod
    def registerBehaviorEngine(cls, behavior_engine: BehaviorEngine) -> None:
        """Register a behavior engine for the person."""
        cls.__behavior_engine = behavior_engine

    @classmethod
    def getBehaviorEngine(cls) -> BehaviorEngine:
        """Get the behavior engine for the person."""
        return cls.__behavior_engine

    # schedules: Optional[Schedules] = \
    #     field(default=tuple[Activities(0, tuple[0, 0, 0])])
    # places: Optional[list[int]] = field(default_factory=list)
    # currentPlaceID: Optional[str] = field(default=None)

    def __init__(self, local_id: int, rank: int, schedules: Schedules, places: namedtuple, initDict: dict):
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
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)

        # `location` is currently referenced required but not used
        if "x" not in initDict:
            initDict["x"] = 0
        if "y" not in initDict:
            initDict["y"] = 0
        if math.isinf(initDict["x"]) or math.isinf(initDict["y"]):
            initDict["x"] = 0
            initDict["y"] = 0

        self.location = cpt(x=int(initDict["x"]), y=int(initDict["y"]), z=0)

        self.schedules: Schedules = schedules

        # map input parameters to dict
        initDict["person_id"] = local_id
        initDict["place_id"] = places[0]
        initDict["activity_id"] = 0
        initDict["activities_idx"] = 0
        # initDict['location'] = starting_location
        initDict["places"] = places

        self.state = create_dataclass_record_from_dict(Person.getPersonDataClass(), initDict)

        self.messages_outgoing: list[Message] = []
        self.messages_sent: list[Message] = []
        self.messages_incoming: list[Message] = []

        # create the default behavior engine
        self.behavior_engine = Person.getBehaviorEngine()(self)

    @property
    def pt(self) -> cpt:
        """"""
        return self.location

    @property
    def currentPlaceID(self) -> int:
        return self.state.place_id

    @property
    def places(self) -> list[int]:
        return list(self.state.places)

    @property
    def last_known_place(self) -> PersonLastKnownPlace:
        """Returns the last known location of this person."""
        return PersonLastKnownPlace(
            person_id=self.state.person_id,
            place_id=self.state.place_id,
            minute_last_updated=self.state.minute_last_moved,
        )

    def setBehaviorEngine(self, behaviorEngine: type[BehaviorEngine]):
        """Sets the behavior engine for this person."""
        self.behavior_engine = behaviorEngine()(self)

    def save(self) -> tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person.
        """
        return (
            self.uid,  # 0: uid is a tuple
            self.schedules.data(),  # 1: schedules is a Schedules object
            tuple(e for e in self.location.coordinates),  # 2: location
            astuple(self.state),  # 3: state is a PersonData object
        )

    def move(self, cal: SimTime, places_proj) -> bool:
        """Move to the place indicated by the schedule for this tick.
        Args:
            cal: The calendar for the current time.
            places_proj: The PlacesProjection (or EnhancedPlacesProjection) to use for moving the agent.
        Returns:
            True if the agent moved to the next place, False otherwise.
        """
        success = False
        next_activity_id = self.selectNextPlace(cal)
        next_place_id = self.places[next_activity_id]

        if not next_place_id:
            logger.debug(f"Agent {self.id} has no place to go - going remote.")
            logger.debug(f"places = {self.places}")
            logger.debug(f"schedule = {self.schedules}")
            next_place_id = 0  # reset to home

        if next_place_id == self.currentPlaceID:
            # already at the place
            # logger.debug(
            #     f"Agent {self.id} is already at place {self.currentPlaceID}")
            return False

        place = places_proj.lookup_place(next_place_id)  # place_map.get(next_place_id)
        if place is None:
            logger.debug(f"Place {next_place_id} not found.")
            logger.debug(f"places = {self.places}")
            return False

        if place is not None:
            success = True
            self.state.place_id = next_place_id
            logger.debug(f"Rank {rank}: " f"Agent {self.id} is moving to place {self.state.place_id}")
            places_proj.move_agent_to_place(self, place)
            self.state.minute_last_moved = cal.minute_of_day
        else:
            logger.debug(f"move for act {next_activity_id} to place {next_place_id} failed.")
            logger.debug(f"places = {self.places}")
            logger.debug(f"Remaining a currentPlaceID = {self.state.place_id}")

        return success

    def selectActivities(self, cal: SimTime) -> int:
        """Select the activities for the time of day and day of week.
        Args:
            cal: The calendar for the current time.
        Returns:
            The index of the activities to use for the current time.
        """
        # Select the activities based on the time of day and day of week
        activities_idx = self.state.activities_idx  # previously selected activities index

        # the first two activities are for weekdays and weekends
        # any additional activities are for unplanned activities
        if activities_idx < 2:  # if sticking to planned activities
            if cal.is_weekday():
                activities_idx = 0
            elif len(self.schedules) > 0:  # if there is a weekend schedule
                activities_idx = 1
            else:  # if unplanned activities
                activities_idx = 0  # default to weekday activities
        else:
            # if unplanned activities, use the current index
            activities_idx = self.state.activities_idx

        return activities_idx

    def selectNextPlace(self, cal: SimTime) -> int:
        """Select the next place to go to based on the schedule for time of
        day and day of week.
        Args:
            cal: The calendar for the current time.
        Returns:
            The ID of the next place to go to.  If the activity is not in the
            list of places, it defaults to home (place_id 0).
        """
        time = cal.minute_of_day

        activities_idx = self.selectActivities(cal)
        act = self.schedules[activities_idx].activityAt(time)

        next_activity_id = 0  # home is the default
        if act is not None and act.activity_id < len(self.places):
            next_activity_id = int(act.activity_id)
            # else:  if the activity is not in the list of places, go home

        return next_activity_id

    def count_colocations(self, cspace):
        # subtract self
        num_here = cspace.get_num_agents(self.state.location) - 1
        logger.debug(f"Agent {self.id} sees {num_here} other agents.")
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
        metadata: dict | None = None,
        attachments: dict | None = None,
    ) -> Message:
        """Create a message to send to other agents."""
        if attachments is None:
            attachments = {}
        if metadata is None:
            metadata = {}
        msg = Message(
            sender=self.uid,
            recipient=recipient,
            message=message,
            timestamp=timestamp,
            metadata=metadata,
            attachments=attachments,
        )

        self.messages_outgoing.append(msg)

        return msg

    def send_messages(self) -> list[Message]:
        """Send messages to other agents."""
        outgoing_messages = self.messages_outgoing
        self.messages_sent.extend(outgoing_messages)
        self.messages_outgoing = []

        return outgoing_messages

    def receive_message(self, message: Message) -> bool:
        """Process received messages."""
        self.messages_incoming.append(message)

        return True

    def process_messages(self) -> None:
        """modify the state of the person based on the messages received."""
        for msg in self.messages_incoming:
            # process the message
            logger.debug(f"Agent {self.id} received message: {msg.message}")

        self.messages_incoming = []
        self.messages_outgoing = []

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        self.behavior_engine.decide(context, cal)
        # self.move(cal, context.get_projection("places"))

    @classmethod
    def restore(cls, person_data: tuple) -> Person:
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

        # pt_array = list(person_data[2])
        # pt = cpt(pt_array[0], pt_array[1], 0)

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
class PersonConfig(NamedTuple):
    name: str
    person_type: type[Person]
    dataType: type[PersonData]
    behaviorEngine: type[BehaviorEngine]


# 4. test code
def test_person():
    """Test the Person class."""
    person_data = {"person_id": 1, "place_id": 0, "activity_id": 0, "places": [0, 1, 2]}

    person_data_class = dataclass(PersonData, frozen=True)

    Person.registerPersonDataClass(person_data_class)

    person = Person(
        local_id=1,
        rank=0,
        schedules=Schedules(),
        places=[0, 1, 2],
        starting_location=cpt(0, 0, 0),
        initDict=person_data,
    )

    logger.debug(person)

    person_data = person.save()
    logger.debug(person_data)

    restored_person = Person.restore(person_data)
    logger.debug(restored_person)

    logger.debug("Person test passed.")


def test_person_serialization(person: Person):
    logger.debug("Testing person serialization.")
    person_data = person.save()
    logger.debug(person_data)

    restored_person = Person.restore(person_data)
    logger.debug(restored_person)


def test_activities(person: Person):
    logger.debug("Testing activities.")
    schedules = person.schedules
    schedule_names = [schedule.name for schedule in schedules.schedules]
    logger.debug(schedule_names)

    for schedule_idx in range(len(schedules)):
        logger.debug(f"Schedule {schedule_idx}:")
        for act in schedules[schedule_idx].acts:
            logger.debug(act)


if __name__ == "__main__":
    test_person()


person_cache = {}
person_id_map = {}
