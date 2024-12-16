from casmsocial.place import Place

from typing import (
    Dict,
    Type
)
from dataclasses  import dataclass

class Workplace(Place):
    """Work class"""
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        initDict["place_type"] = "Workplace"
        super().__init__(
             initDict,
             placeDataClass
        )