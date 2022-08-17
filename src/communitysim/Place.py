""" Generic Place Class """
from repast4py.space import DiscretePoint as dpt

class Place:
    def __init__(self, place_id: int, location: dpt):
        self.id = place_id
        self.location = location