""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt

from typing import Type

from casmsocial.person import Person
from casmsocial.calendar import Calendar


class Place(object):
    """Generic Place Class"""

    place_types = []

    def __init__(self, placeId: int, location: dpt):
        self.id = placeId
        self.location = location
        self.rank = -1

        self.peopleAtPlace = []
        self.personIdsAtPlace = []
        self.headIndex = None

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

def get_place_type_idx(place_type: Type[Place]) -> int:
    """Get the index of a place type in the place_types list."""
    return Place.place_types.index(place_type)