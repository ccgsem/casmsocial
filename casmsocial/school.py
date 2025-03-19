from dataclasses import dataclass

from casmsocial.place import Place


class School(Place):
    """School class"""

    def __init__(self, initDict: dict, placeDataClass: type[dataclass]):
        initDict["place_type"] = "School"
        super().__init__(initDict, placeDataClass)
