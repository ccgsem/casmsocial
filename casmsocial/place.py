""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt

from dataclasses  import dataclass, fields
from typing import (
    Type,
    List,
    Dict,
    NamedTuple
)
import math

from casmsocial.person import Person
from casmsocial.calendar import Calendar
from casmsocial.datautility import create_dataclass_record_from_dicts


@dataclass
class PlaceData:
    """Data for a Place."""
    place_type: str = "household"
    place_name: str = ""
    latitude: float = float('nan')
    longitude: float = float('nan')
    heatIndex: float = float('nan')
    AIR: bool = False
    # x: float
    # y: float
    # location: cpt
    # rank: int
    # peopleAtPlace: List[Person]
    # personIdsAtPlace: List[int]
    # heatIndex: Optional[float]


class Place(object):
    """Generic Place Class"""
    
    def __init__(
            self,
            placeTypeName: str,
            initDict: Dict,
            placeDataType: Type[dataclass]
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
            create_dataclass_record_from_dicts(
                placeDataType,
                initDict,
                {'place_type': placeTypeName}                
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


# NamedTuple for PlaceConfig
PlaceConfig = NamedTuple(
    'PlaceConfig',
    [
        ('name', str),
        ('type', Type[Place]),
        ('dataType', Type[dataclass])
    ]
)


class Places:
    """Configurations for places."""

    # List of PlaceConfigs
    __configs: List[PlaceConfig] = []

    @classmethod
    def register_place_config(cls, config: PlaceConfig):
        """Add a PlaceConfig to the list of configs."""
        cls.__configs.append(config)

    @classmethod
    def get_place_config(cls, idx: int) -> PlaceConfig:
        """Get a PlaceConfig from the list of configs."""
        return cls.__configs[idx]

    @classmethod
    def get_place_config_idx(cls, name: str) -> int:
        """Get the index of a PlaceConfig in the list of configs."""
        for idx, config in enumerate(cls.__configs):
            if config.name == name:
                return idx
        return -1

    @classmethod
    def get_num_configs(cls) -> int:
        """Get the number of PlaceConfigs in the list of configs."""
        return len(cls.__configs)
