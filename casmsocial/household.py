from casmsocial.place import Place
from casmsocial.calendar import Calendar

from typing import (
    Dict,
    Type
)
from dataclasses  import dataclass


class Household(Place):
    """Household class"""
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        initDict["place_type"] = "Household"
        super().__init__(
             initDict,
             placeDataClass
        )
