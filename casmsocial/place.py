""" Generic Place Class """
from repast4py.space import ContinuousPoint as cpt

from dataclasses  import dataclass
from typing import (
    Type,
    Dict
)
import math

from casmsocial.person import Person
from casmsocial.datautility import create_dataclass_record_from_dict


@dataclass
class PlaceData:
    """Data for a Place."""
    place_type: str = "Household"
    place_name: str = ""
    latitude: float = float('nan')
    longitude: float = float('nan')


class Place(object):
    """Generic Place Class"""
    
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        """Constructor for the Place class."""

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

        # create data from initDict
        self.data = \
            create_dataclass_record_from_dict(
                placeDataClass,
                initDict          
            )

    def reset(self):
        self.peopleAtPlace.clear()

    def addPerson(self, person: Person):
        if person is not None and person not in self.peopleAtPlace:
            self.peopleAtPlace.append(person)

    def peopleAtPlace(self):
        return self.peopleAtPlace

    def step(self, calendar, rng):
        pass
