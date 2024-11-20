from casmsocial.place import Place

from typing import Dict

class School(Place):
    """School class"""
    def __init__(self, initDict: Dict):
        super().__init__(initDict)