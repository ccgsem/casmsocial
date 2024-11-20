from casmsocial.place import Place

from typing import Dict

class Work(Place):
    """Work class"""
    def __init__(self, initDict: Dict):
        super().__init__(initDict)