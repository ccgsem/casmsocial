from collections import namedtuple


class Environment:
    def __init__(self, name):
        self.name = name

    def setup(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def teardown(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def update(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def get_values_at(self, x: float, y: float, z: float = 0.0) -> namedtuple:
        return None
