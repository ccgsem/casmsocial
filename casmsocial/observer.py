"""Observer base class for CasmSocial models"""

from casmsocial.model import Model


class Observer:
    """Base class for all observers"""

    def __init__(self, name, model: Model = None):
        self.name = name

    def initialize(self, model: Model) -> None:
        """Initialize the observer with the model."""
        pass

    def on_step(self, model: Model) -> None:
        """Called at each step of the simulation."""
        pass

    def on_end(self, model: Model) -> None:
        """Called at the end of the simulation."""
        pass
