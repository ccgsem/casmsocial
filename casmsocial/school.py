from casmsocial.place import Place

from typing import (
    Dict,
    Type
)
from dataclasses  import dataclass

class School(Place):
    """School class"""
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        initDict["place_type"] = "School"
        super().__init__(
             initDict,
             placeDataClass
        )