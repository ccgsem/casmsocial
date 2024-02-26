""" Person Agent Base Class """
from __future__ import annotations
from repast4py import core
from repast4py.space import DiscretePoint as dpt

from typing import Tuple, Dict
from pydantic import BaseModel

from .Schedule import Schedule, restoreSchedule
from .Calendar import Calendar
from .Activities import Act, Activities
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
        activities: Activities,
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
        super().__init__(
            id=local_id,
            type=Person.TYPE,
            rank=rank)

        self.activities = activities
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
            self.activities.data(),
            self.places,
            self.pt.coordinates,
            self.currentPlaceID,
            self.risk
            )

    def move(self, cal: Calendar, grid, place_map: Dict) -> bool:
        """Move to the place indicated by the schedule for this tick.
        """
        success = False
        next_activity_id = self.selectNextPlace(cal)
        next_place_id = self.places[next_activity_id]

        place = place_map.get(next_place_id)
        if place is None:
            next_place_id = 0  # reset to home

        if place is not None:
            success = True
            self.currentPlaceID = next_place_id
            # print(
            #    f"Rank {rank}: "
            #    f"Agent {self.id} is moving to place {self.currentPlaceID}")
            placeLocation = place.location
            self.pt = grid.move(self, placeLocation)
        else:
            print(f"move for act {next_activity_id} to place {next_place_id} failed.")
            print(f"places = {self.places}")
            print(f"Remaining a currentPlaceID = {self.currentPlaceID}")

        return success
    
    def selectNextPlace(
            self,
            cal: Calendar) -> int:
        """Select the next place to go to based on the schedule for this tick.
        """
        time = cal.minute_of_day
        act = self.activities.activityAt(time)

        next_activity_id = 0  # home is the default
        if act is not None:
            if act.activity_id < len(self.places):
                next_activity_id = act.activity_id
            # else:  if the activity is not in the list of places, go home
 
        return next_activity_id

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

    @classmethod
    def restore(cls, person_data: Tuple) -> Person:
        """Creates or updates a local person from person_data.

        Args:
            person_data: tuple containing the data returned by Person.save(). 
        """ 
        uid = person_data[0]
        pt_array = person_data[3]
        pt = dpt(pt_array[0], pt_array[1], 0)
        currentPlaceID = person_data[4]
        risk = person_data[5]

        activities = Activities.restore(person_data[1])
        person = Person(uid[0], uid[2], activities, person_data[2], pt)
        person.currentPlaceID = currentPlaceID
        person.risk = risk

        return person


person_cache = {}
