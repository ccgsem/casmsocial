from casmsocial.place import Place

from typing import (
    Dict,
    Type
)
from dataclasses  import dataclass

class Work(Place):
    """Work class"""
    def __init__(
            self,
            initDict: Dict,
            placeDataType: Type[dataclass]
        ):
        super().__init__(
             "work",
             initDict,
             placeDataType
        )