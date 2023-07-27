""" Person Agent Base Class """
from repast4py import core
from repast4py.space import DiscretePoint as dpt

from typing import Tuple, Dict

from .Schedule import Schedule, restoreSchedule
from .Calendar import Calendar
from .Parameters import Parameters

from csv import DictReader
import numpy as np

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


class Person(core.Agent):
    TYPE = 0

    def __init__(
        self,
        local_id: int,
        rank: int,
        schedule: Schedule,
        places: list[int],
        starting_location: dpt,
        starting_risk: int=0):
        """Constructor for the Person class.
        
        Arguments:
            local_id: The ID for this person on this process, combines with the rank
                to form a simulation-wide unique ID.
            rank: The rank of this process.
            schedule: A Schedule class object that will provide a place type int 
                when given the current simulation tick. The place types are 
                implementation-agnostic so "0" could mean "home" in one simulation 
                or "grocery store" in another.
            places: A list of place_id's that correspond to values coming out of 
                the schedule. e.g. if the schedule returns "0", this Person will 
                try to go to the place with the ID of places[0]
            starting_location: A DiscretePoint for this Person's starting location 
                on a grid projection. Set to null and override the move() function 
                if not using a grid projection.
        """
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)
        self.schedule = schedule
        self.places = places
        self.pt = starting_location
        self.currentPlaceID: str = self.places[0]
        self.risk = starting_risk
        self.influenceSusceptibility = Parameters.influenceSusceptibility
        self.interpersonalInfluence = Parameters.interpersonalInfluence

        #print(f"Person {self.id} is ready!")

    def save(self) -> Tuple:
        """Saves the state of this Person as a Tuple.

        Returns:
            The saved state of this Person. 
        """ 
        return (
            self.uid,
            self.schedule.data(),
            self.places,
            self.pt.coordinates,
            self.currentPlaceID,
            self.risk
            )

    
    def move(self, tick: int, grid, place_map: Dict):
        """Move to the place indicated by the schedule for this tick.
        """ 
        activityType = self.schedule.activityAt(tick)
        assert(activityType < len(self.places))
        self.currentPlaceID = self.places[activityType]
        placeLocation = place_map[self.currentPlaceID].location
        print(f"Rank {rank}: Agent {self.id} is moving to place {self.currentPlaceID} at tick {tick}.")

        self.pt = grid.move(self, placeLocation)

    def count_colocations(self, grid):
        # subtract self
        num_here = grid.get_num_agents(self.pt) - 1
        print(f"Agent {self.id} sees {num_here} other agents.")
        # meet_log.total_meets += num_here
        # if num_here < meet_log.min_meets:
        #     meet_log.min_meets = num_here
        # if num_here > meet_log.max_meets:
        #     meet_log.max_meets = num_here
        # self.meet_count += num_here

    def make_contacts(self, contacts):
        riskTimesInfluence = np.array([c.risk * c.interpersonalInfluence for c in contacts])
        influence = riskTimesInfluence.sum()

        self.updateRiskPerception(self.influenceSusceptibility, influence)
    
    def step(self, calendar: Calendar):
        pass

    def updateRiskPerception(self, susceptibility, influence):
        
        newRisk = susceptibility * influence
        newRisk += (1 - susceptibility) * self.risk

        self.risk = newRisk


person_cache = {}

def restorePerson(person_data: Tuple):
    """Creates or updates a local person from person_data.

    Args:
        person_data: tuple containing the data returned by Person.save(). 
    """ 
    uid = person_data[0]
    pt_array = person_data[3]
    pt = dpt(pt_array[0], pt_array[1], 0)
    currentPlaceID = person_data[4]
    risk = person_data[5]

    if uid in person_cache:
        person = person_cache[uid]
    else:
        schedule = restoreSchedule(person_data[1])
        person = Person(uid[0], uid[2], schedule, person_data[2], pt)
        person_cache[uid] = person

    # Update fields that might be old from the cache
    person.pt = pt 
    person.currentPlaceID = currentPlaceID
    person.risk = risk

    return person


