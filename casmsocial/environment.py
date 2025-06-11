from collections import namedtuple

from repast4py.context import SharedContext

from casmsocial.sim_time import SimTime


class Environment:
    """Base class for all environments"""

    def __init__(self, name):
        self.name = name
        self.environment_tuple: namedtuple = None

    @property
    def environment_values(self) -> namedtuple:
        return self.environment_tuple

    def setup(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def teardown(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def step(self, context: SharedContext, cal: SimTime) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def get_values_at(self, x: float, y: float, z: float = 0.0) -> namedtuple:
        return None
