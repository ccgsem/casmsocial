# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 02 Dec 2024

Defining abstract Model interface
"""
from typing import (
    Callable,
    Dict
)
from mpi4py import MPI

from abc import ABC, abstractmethod


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

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        """Constructor for the Model class."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the model."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Step the model forward one time step."""
        pass


# model factory implementation
__MODELS = {}


def get_models(        
) -> dict[str, Callable[[MPI.Intracomm, dict], Model]]:
    """
    Returns a dictionary of available models with their creators, must be
    callable with the following signature:
    model_creator(
        comm: MPI.Intracomm,
        params: dict) -> Model
    """
    return __MODELS


def register_casmsocial_model(
        model_type: str
        ) -> Callable[[MPI.Intracomm, dict], Model]:
    """
    Registers a model creator, must be callable with the following signature:
    model_creator(model_type: MPI.Intracomm, dict) -> Model

    Args:
    model_type - model type, must be a string

    Returns:
    decorator - a decorator to register the model creator
    """
    def decorator(fn):
        __MODELS[model_type] = fn
        return fn
    return decorator


# model creator
def get_casmsocial_model(
        model_type: str
        ) -> Callable[[MPI.Intracomm, dict], Model]:
    """
    Returns an casmsocial model creator, must be callable with the following
    signature:
    model_creator(model_type: MPI.Intracomm, dict) -> Model

    Args:
    model_type - model type, must be a string
    """

    if model_type not in __MODELS:
        #logger.info("Available models:")
        print("Available models:")
        for key in __MODELS.keys():
            #logger.info(key)
            print(key)
        raise ValueError(f"Unsupported model type: {model_type}")
    return __MODELS[model_type]


class ModelNotFoundError(Exception):
    """ exception if model not found """
    pass
