""" Human Agent Base Class """
from repast4py import core
from repast4py.space import DiscretePoint as dpt

from typing import Tuple, Dict

import Schedule


class Human(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, schedule: Schedule, places: [int], starting_location: dpt):
        """Constructor for the Human class.
        
        Arguments:
            local_id: The ID for this human on this process, combines with the rank
                to form a simulation-wide unique ID.
            rank: The rank of this process.
            schedule: A Schedule class object that will provide a place type int 
                when given the current simulation tick. The place types are 
                implementation-agnostic so "0" could mean "home" in one simulation 
                or "grocery store" in another.
            places: A list of place_id's that correspond to values coming out of 
                the schedule. e.g. if the schedule returns "0", this Human will 
                try to go to the place with the ID of places[0]
            starting_location: A DiscretePoint for this Human's starting location 
                on a grid projection. Set to null and override the move() function 
                if not using a grid projection.
        """
        super().__init__(id=local_id, type=Human.TYPE, rank=rank)
        self.schedule = schedule
        self.places = places
        self.pt = starting_location

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Returns:
            The saved state of this Human. 
        """ 
        return (self.uid, self.schedule.data(), self.places, self.pt.coordinates)

    
    def move(self, tick: int, grid, place_map: Dict):
        """Move to the place indicated by the schedule for this tick.
        """ 
        activityType = self.schedule.activityAt(tick)
        assert(activityType < len(self.places))
        placeId = self.places[activityType]
        placeLocation = place_map[placeId].location

        self.pt = grid.move(self, placeLocation)

human_cache = {}

def restoreHuman(human_data: Tuple):
    """Creates or updates a local human from human_data.

    Args:
        human_data: tuple containing the data returned by Human.save(). 
    """ 
    uid = human_data[0]
    pt_array = human_data[3]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid in human_cache:
        human = human_cache[uid]
    else:
        schedule = restoreSchedule(human_data[1])
        human = Human(uid[0], uid[2], schedule, human_data[2], pt)
        human_cache[uid] = human

    # There is currently no data to update instead of pt
    human.pt = pt 

    return human