"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""
from loguru import logger
from mpi4py import MPI
from repast4py import logging

from casmsocial.factory import Models
from casmsocial.person import Person, PersonData
from casmsocial.place import Place, PlaceData
from casmsocial.social_model import AgentTypeConfig, SIModel


# 1. Define a SIModel-derived Class
class ArtSocModel(SIModel):
    """ArtSocModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the ArtSocModel class"""
        super().__init__(comm, params)

    def build_context(self) -> None:
        """Initialize population"""

        # register the person and place agent types
        logger.info(f"Registering person type (TYPE={Person.TYPE})...")
        SIModel.register_agent_type_config(
            AgentTypeConfig(name="Person", agent_type=Person, agent_data_type=PersonData)
        )

        logger.info(f"Registering place type (TYPE={Place.TYPE})...")
        SIModel.register_agent_type_config(AgentTypeConfig(name="Place", agent_type=Place, agent_data_type=PlaceData))

        # register the activities
        SIModel.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
        SIModel.register_activity_names(["home", "work", "school"])

        logger.info("Now running initialize population for SIModel...")
        super().build_context()

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
