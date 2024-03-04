""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt

from csv import DictReader

from .person import Person
from .calendar import Calendar

class Place(object):
    def __init__(self, placeId: int, location: dpt):
        self.id = placeId
        self.location = location
        self.rank = -1

        self.peopleAtPlace = []

    def reset(self):
        self.peopleAtPlace.clear()

    def addPerson(self, person: Person):
        if person is not None:
            self.peopleAtPlace.append(person)

    def peopleAtPlace(self):
        return self.peopleAtPlace

    def step(self, calendar, rng):
        pass