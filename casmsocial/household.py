from dataclasses import dataclass

from casmsocial.place import Place


class Household(Place):
    """Household class"""

    def __init__(self, initDict: dict, placeDataClass: type[dataclass]):
        initDict["place_type"] = "Household"
        super().__init__(initDict, placeDataClass)
