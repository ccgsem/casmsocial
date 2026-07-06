"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""

from loguru import logger
from mpi4py import MPI

from casmsocial.casmpop import AgentLogger, CasmPop
from casmsocial.factory import Models
from casmsocial.person import Person, PersonData
from casmsocial.place import Place, PlaceData

# 1. Define an Observer Data class to provide a structured format for logging
# agent data

# 2 Define an Observer class for logging agent data


# 3. Define a CasmPop-derived Class
class CitySocialModel(CasmPop):
    """CitySocialModel class"""

    @classmethod
    def get_default_parameters(cls) -> dict:
        """Get the default parameters for the CitySocialModel."""
        params = super().get_default_parameters()
        params.update(
            {
                "model.name": cls.__module__ + "." + cls.__name__,
            }
        )
        return params

    @classmethod
    def get_default_observer_parameters(cls):
        return super().get_default_observer_parameters()

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the CitySocialModel class"""
        super().__init__(comm, params)

    def build_context(self) -> None:
        """Initialize population"""

        # add the agent logger observer (using basic Observer for now, but
        # could be extended to log more detailed information or to log at different intervals)
        self.add_observer(AgentLogger("AgentLogger", self))

        # register the person and place agent types
        logger.info(f"Registering person type (TYPE={Person.TYPE})...")
        CasmPop.setPersonClass(Person, PersonData)

        logger.info(f"Registering place type (TYPE={Place.TYPE})...")
        CasmPop.setPlaceClass(Place, PlaceData)

        # register the activities
        CasmPop.register_planned_activity_names(["sp_hh_id", "sp_work_id", "sp_school_id"])
        CasmPop.register_activity_names(["home", "work", "school"])

        logger.info("Now running initialize population for CasmPop...")
        super().build_context()

        # initialize the logging
        self.log_agents()


# Register CitySocialModel
Models.add_model(CitySocialModel.__module__ + "." + CitySocialModel.__name__, CitySocialModel)

# Canonical scenarios for this model are registered in casmdb via:
#   uv run python scripts/register_casmsocial.py --db /path/to/models.db
# To run a named scenario:
#   uv run python scripts/run_scenario.py --db /path/to/models.db \
#       --model casmsocial --model-version <ver> --scenario <name>
