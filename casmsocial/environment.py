from repast4py.context import SharedContext

from casmsocial.sim_time import SimTime


class Environment:
    """Base class for all environments"""

    def __init__(self, name):
        self.name = name

    def setup(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def teardown(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def step(self, context: SharedContext, cal: SimTime) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def at_end(self, context: SharedContext, cal: SimTime) -> None:
        raise NotImplementedError("Subclasses should implement this method")
