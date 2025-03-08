"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""
from mpi4py import MPI
from repast4py import logging

# model factory
from casmsocial.factory import Models

# place types
from casmsocial.household import Household
from casmsocial.person import Person, PersonConfig, PersonData
from casmsocial.place import PlaceConfig, PlaceData, RemotePlace
from casmsocial.school import School
from casmsocial.socialmodel import SIModel
from casmsocial.workplace import Workplace


# 1. Define a SIModel-derived Class
class ArtSocModel(SIModel):
    """ ArtSocModel class """

    def __init__(
        self,
        comm: MPI.Intracomm,
        params: dict
    ):
        """ Constructor for the ArtSocModel class """
        super().__init__(comm, params)

    def initializePopulation(self) -> None:
        """Initialize population"""
        # register the place types
        print("Registering place types...")

        SIModel.register_place_config(
            PlaceConfig(
                name='Household',
                type=Household,
                dataType=PlaceData,
                personPlaceField='sp_hh_id'
            )
        )
        SIModel.register_place_config(
            PlaceConfig(
                name='School',
                type=School,
                dataType=PlaceData,
                personPlaceField='sp_school_id'
            )
        )
        SIModel.register_place_config(
            PlaceConfig(
                name='Workplace',
                type=Workplace,
                dataType=PlaceData,
                personPlaceField='sp_work_id'
            )
        )

        # register the remote place type
        SIModel.register_remote_place_config(
            PlaceConfig(
                name='RemotePlace',
                type=RemotePlace,
                dataType=PlaceData,
                personPlaceField=''
            )
        )

        # register the person types
        print("Registering person type...")
        SIModel.register_person_config(
            PersonConfig(
                name='Person',
                type=Person,
                dataType=PersonData
            )
        )

        print("Now running initialize population for SIModel...")
        super().initializePopulation()

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            self.params['agent_log_file'],
            [
                'tick',
                'agent_id',
                'x',
                'y',
            ]
        )
        self.log_agents()

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day

        for person in self.context.agents():

            self.agent_logger.log_row(
                tick,
                person.id
            )

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()


# Register ArtSocModel
Models.add_model(
    ArtSocModel.__module__ + '.' + ArtSocModel.__name__,
    ArtSocModel)
