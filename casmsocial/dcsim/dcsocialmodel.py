# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 09 Dec 2024

Defining the artificial social model for the Artificial Societies project
"""

from mpi4py import MPI
from repast4py import (
    logging
)

from casmsocial.model import Model
from casmsocial.geomodel import GeoModel
from casmsocial.place import PlaceData
from casmsocial.places import (
    PlacesConfig,
    Places
)

# place types
from casmsocial.household import Household
from casmsocial.workplace import Workplace
from casmsocial.school import School

# model factory
from casmsocial.modelfactory import (
    register_casmsocial_model
)

from dataclasses  import dataclass
from typing import Dict
from collections import deque
import pandas as pd


# register the 'casmsocial' model
@register_casmsocial_model('casmsocial_dcsim_ArtSocModel')
def create_casmsocial_GeoModel(
    comm: MPI.Intracomm,
    params: dict
) -> Model:
    print("Registering casmsocial_dcsim_ArtSocModel model")
    return ArtSocModel(comm, params)


class ArtSocModel(GeoModel):
    """ HeatRiskModel class """
    
    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        """ Constructor for the HeatRiskModel class """
        # register the place types
        Places.register_place_config(
            PlacesConfig(
                name='Household',
                type=Household,
                dataType=PlaceData,
                personPlaceField='sp_hh_id'
            )
        )
        Places.register_place_config(
            PlacesConfig(
                name='School',
                type=School,
                dataType=PlaceData,
                personPlaceField='sp_school_id'
            )
        )
        Places.register_place_config(
            PlacesConfig(
                name='Work',
                type=Workplace,
                dataType=PlaceData,
                personPlaceField='sp_work_id'
            )
        )

        super().__init__(comm, params)

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            comm,
            params['agent_log_file'],
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
        for person in self.people:
            self.agent_logger.log_row(
                tick,
                person.id,
                person.state.location.x,
                person.state.location.y,
            )

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()

