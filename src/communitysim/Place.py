""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt

from csv import DictReader

from Human import Human
from Calendar import Calendar

class Place:
    def __init__(self, placeId: int, rank, location: dpt):
        self.id = placeId
        self.rank = rank
        self.location = location
        self.fireRisk = 0
        self.insuranceStance = 0
        self.hasInsurance = False

        self.peopleAtPlace = []

    def reset(self):
        self.peopleAtPlace.clear()

    def addPerson(self, person: Human):
        if person is not None:
            self.peopleAtPlace.append(person)

    def peopleAtPlace(self):
        return self.peopleAtPlace

    def updateInsuranceStance(self):
        self.insuranceStance = 0

    def isGettingInsurance(self, rng):
        return False

    def step(self, calendar):
        pass