"""
Author: Jon Cline
Created: 02 Dec 2024

Defining modelfactory interface
"""
from typing import ClassVar

from casmsocial.model import Model

# model factory implementation
__MODELS = {}


class Models:
    """Model Factory class.

    The Model Factory class is responsible for creating models.
    """

    # dictionary of models
    __models: ClassVar[dict[str, type[Model]]] = {}

    @staticmethod
    def add_model(name: str, model: type[Model]) -> type[Model]:
        """Add a model to the factory.

        Args:
        name - the name of the model
        model - the model class

        Raises:
        ValueError - if the model already exists

        Returns:
        model - the model class
        """
        if name in Models.__models:
            raise ModelAlreadyExistsError(name)
        Models.__models[name] = model
        return model

    @staticmethod
    def create_model(name: str) -> type[Model]:
        """Create a model.

        Args:
        name - the name of the model

        Raises:
        ValueError - if the model does not exist

        Returns:
        model - the model class
        """
        if name not in Models.__models:
            print("Available models:")
            for key in Models.__models:
                print(key)
            raise ModelNotFoundError(name)
        return Models.__models[name]

    @staticmethod
    def get_models() -> dict[str, type[Model]]:
        """Get the models dictionary.

        Returns:
        models - the models dictionary
        """
        return Models.__models


class ModelAlreadyExistsError(Exception):
    """Exception raised when a model already exists."""

    def __init__(self, model_name: str):
        self.message = f"Model {model_name} already exists"
        super().__init__(self.message)


class ModelNotFoundError(Exception):
    """ exception if model not found """
    def __init__(self, model_type: str):
        self.message = f"Unsupported model type: {model_type}"
        super().__init__(self.message)


def load_models(
        module_list = None
        ) -> None:
    """ load models from a list of modules """
    if module_list is None:
        module_list = []
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
