from .place import Place
from .calendar import Calendar

from typing import Dict
from repast4py.space import DiscretePoint as dpt


class Household(Place):
    def __init__(self, initDict: Dict):
        placeId = initDict['sp_id']
        location = dpt(x=int(initDict['x']), y=int(initDict['y']), z=0)
        super().__init__(placeId, location)

    def step(self, calendar: Calendar, rng):
	    pass
