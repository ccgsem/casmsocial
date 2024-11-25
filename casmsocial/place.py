""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt

from typing import Type, Dict
import math

from casmsocial.person import Person
from casmsocial.calendar import Calendar


class Place(object):
    """Generic Place Class"""

    place_types = []

    def __init__(
            self,
            initDict: Dict
        ):
        placeId = initDict['sp_id']

        # `location` is currently referenced required but not used
        if 'x' not in initDict:
            initDict['x'] = 0
        if 'y' not in initDict:
            initDict['y'] = 0
        if math.isinf(initDict['x']) or math.isinf(initDict['y']):
            initDict['x'] = 0
            initDict['y'] = 0

        location = cpt(x=int(initDict['x']), y=int(initDict['y']), z=0)

        self.id = placeId
        self.location = location
        self.rank = -1

        self.peopleAtPlace = []
        self.personIdsAtPlace = []
        self.heatIndex = None

        self.AIR = False
        if 'AIR' in initDict:
            self.AIR = bool(initDict['AIR'])

    def reset(self):
        self.peopleAtPlace.clear()

    def addPerson(self, person: Person):
        if person is not None and person not in self.peopleAtPlace:
            self.peopleAtPlace.append(person)

    def peopleAtPlace(self):
        return self.peopleAtPlace

    def step(self, calendar, rng):
        pass

# Register the Place subclass with the Place class
def register_place_type(place_type: type[Place]):
    """Register a place type with the Place class."""
    Place.place_types.append(place_type)

def get_place_type(idx: int) -> Type[Place]:
    """Get a place type from the place_types list."""
    return Place.place_types[idx]

def get_place_type_idx(place_type: Type[Place]) -> int:
    """Get the index of a place type in the place_types list."""
    return Place.place_types.index(place_type)