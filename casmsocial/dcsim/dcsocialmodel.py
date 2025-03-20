"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""
from loguru import logger
from mpi4py import MPI
from repast4py import logging

# model factory
from casmsocial.factory import Models

# place types
from casmsocial.household import Household
from casmsocial.person import BehaviorEngine, Person, PersonConfig, PersonData
from casmsocial.place import PlaceConfig, PlaceData, RemotePlace
from casmsocial.school import School
from casmsocial.socialmodel import SIModel
from casmsocial.workplace import Workplace


# 1. Define a SIModel-derived Class
class ArtSocModel(SIModel):
    """ArtSocModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the ArtSocModel class"""
        super().__init__(comm, params)

    def initialize_population(self) -> None:
        """Initialize population"""
        # register the place types
        logger.info("Registering place types...")

        SIModel.register_place_config(PlaceConfig(name="Household", place_type=Household, dataType=PlaceData))
        SIModel.register_place_config(PlaceConfig(name="School", place_type=School, dataType=PlaceData))
        SIModel.register_place_config(PlaceConfig(name="Workplace", place_type=Workplace, dataType=PlaceData))

        # register the remote place type
        SIModel.register_remote_place_config(
            PlaceConfig(name="RemotePlace", place_type=RemotePlace, dataType=PlaceData)
        )

        # register the person types
        logger.info("Registering person type...")
        SIModel.register_person_config(
            PersonConfig(name="Person", person_type=Person, dataType=PersonData, behaviorEngine=BehaviorEngine)
        )

        logger.info("Now running initialize population for SIModel...")
        super().initialize_population()

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            self.params["agent_log_file"],
            [
                "tick",
                "agent_id",
                "x",
                "y",
            ],
        )
        self.log_agents()

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day

        for person in self.context.agents():
            self.agent_logger.log_row(tick, person.id)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()


# Register ArtSocModel
Models.add_model(ArtSocModel.__module__ + "." + ArtSocModel.__name__, ArtSocModel)
