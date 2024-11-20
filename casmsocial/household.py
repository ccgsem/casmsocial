from casmsocial.place import Place
from casmsocial.calendar import Calendar

from typing import Dict
from repast4py.space import DiscretePoint as dpt


class Household(Place):
    """Household class"""
    def __init__(self, initDict: Dict):
        super().__init__(initDict)

    def step(self, calendar: Calendar, rng):
	    pass
