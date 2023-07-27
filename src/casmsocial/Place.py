""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt

from csv import DictReader

from .Person import Person
from .Calendar import Calendar

class Place(object):
    def __init__(self, placeId: int, location: dpt):
        self.id = placeId
        self.location = location
        self.fireRisk = 0
        self.perceivedRisk = 0
        self.hasInsurance = False
        self.rank = -1

        self.peopleAtPlace = []

    def reset(self):
        self.peopleAtPlace.clear()

    def addPerson(self, person: Person):
        if person is not None:
            self.peopleAtPlace.append(person)

    def peopleAtPlace(self):
        return self.peopleAtPlace

    def shopForInsurance(self, rng):
        pass

    def reduceFuel(self):
        pass

    def purchaseInsurance(self, offers):
        pass

    def step(self, calendar, rng):
        pass