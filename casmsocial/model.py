"""
Author: Jon Cline
Created: 02 Dec 2024

Defining abstract Model interface
"""

from abc import ABC, abstractmethod

from mpi4py import MPI


class Model(ABC):
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    theModel: "Model | None" = None

    @classmethod
    def get_model(cls) -> "Model | None":
        """Get the model."""
        return Model.theModel

    @classmethod
    def set_model(cls, new_model: "Model") -> None:
        """Set the model."""
        Model.theModel = new_model

    @abstractmethod
    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the Model class."""

    @abstractmethod
    def start(self) -> None:
        """Start the model."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Step the model forward one time step."""
        pass
