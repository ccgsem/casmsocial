"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""

from dataclasses import dataclass
from loguru import logger
from mpi4py import MPI
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.dataset import HivePartitioning

from casmsocial.casmpop import CasmPop
from casmsocial.factory import Models
from casmsocial.person import Person, PersonData
from casmsocial.place import Place, PlaceData


# 1. Define PersonLogData class
@dataclass
class PersonLogData:
    tick: int
    rank: int  # rank of the agent in the MPI communicator
    agent_id: int
    x: float
    y: float
    place_id: int


# 2. Define a CasmPop-derived Class
class ArtSocModel(CasmPop):
    """ArtSocModel class"""

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the ArtSocModel class"""
        super().__init__(comm, params)

    def build_context(self) -> None:
        """Initialize population"""

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
        self.agent_log_file = self.params["agent_log_file"]
        self.log_agents()

    def get_person_log_data(self, person: Person) -> PersonLogData:
        """Get the agent data for logging."""
        return PersonLogData(
            tick=self.cal.tick,
            rank=self.comm.Get_rank(),
            agent_id=person.id,
            x=person.pt.x,
            y=person.pt.y,
            place_id=person.currentPlaceID,
        )

    def log_agents(self) -> None:
        # create a DataFrame for the agent logs
        logger.info("Logging agents' data...")

        agent_log_df = pl.DataFrame([self.get_person_log_data(person) for person in self.context.agents(agent_type=0)])

        # convert the DataFrame to an Arrow Table
        agent_log_table = agent_log_df.to_arrow()

        # Define partition schema
        partition_schema = pa.schema(
            [
                pa.field("tick", pa.int32()),
                pa.field("rank", pa.int32()),
            ]
        )

        # Set Hive-style partitioning
        partitioning = HivePartitioning(partition_schema)

        # Write dataset
        ds.write_dataset(
            data=agent_log_table,
            base_dir=self.agent_log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

    def at_end(self) -> None:
        super().at_end()


# Register ArtSocModel
Models.add_model(ArtSocModel.__module__ + "." + ArtSocModel.__name__, ArtSocModel)
