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
            placeDataType: Type[dataclass]
        ):
        super().__init__(
             "school",
             initDict,
             placeDataType
        )