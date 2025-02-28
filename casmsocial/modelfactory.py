# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 02 Dec 2024

Defining modelfactory interface
"""
from typing import (
    Callable
)

from casmsocial.model import Model
from mpi4py import MPI

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


def load_models(
        module_list = []
        ) -> None:
    """ load models from a list of modules """
    for module in module_list:
        __import__(module)

def load_models_from_dotenv(
        ) -> None:
    """ load models from a list of modules """
    import os
    import dotenv

    dotenv.load_dotenv()
    model_list = os.getenv("CASMSOCIAL_MODELS").split(",")
    for model in model_list:
        __import__(model)

# end of file