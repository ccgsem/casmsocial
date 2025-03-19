from dataclasses import dataclass

from casmsocial.place import Place


class Workplace(Place):
    """Work class"""

    def __init__(self, initDict: dict, placeDataClass: type[dataclass]):
        initDict["place_type"] = "Workplace"
        super().__init__(initDict, placeDataClass)
