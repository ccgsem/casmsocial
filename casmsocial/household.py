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
            placeDataType: Type[dataclass]
        ):
        super().__init__(
             "household",
             initDict,
             placeDataType
        )
