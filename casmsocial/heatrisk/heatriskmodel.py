# -*- coding: utf-8 -*-
"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""

from mpi4py import MPI
from repast4py import (
    logging
)

from casmsocial.model import Model
from casmsocial.geomodel import GeoModel
from casmsocial.place import (
    PlaceData,
    PlaceConfig,
    Places
)

# place types
from casmsocial.household import Household
from casmsocial.work import Work
from casmsocial.school import School

from casmsocial.person import (
    Person,
    PersonData
)

# model factory
from casmsocial.modelfactory import (
    register_casmsocial_model
)

from dataclasses  import dataclass
from typing import Dict
from collections import deque
import pandas as pd


# utility functions for heat-related computations
def filter_heat_indices(
    heat_indices: list[float],
    threshold: float
) -> list[float]:
    """Filter out all heat indices above the threshold."""
    exceeded = True
    return \
        [t for t in heat_indices if (exceeded := exceeded and  t > threshold)]


def compute_prob_heat_event(
    heat_indices: list[float],
    threshold: float
) -> float:
    """Compute the probability of a heat event."""
    # filter out all heat indices above the threshold
    heat_index = heat_indices[0]
    heat = filter_heat_indices(heat_indices, threshold)
    hours_above_threshold = len(heat)

    # note: length of heat is the number of hours above the threshold
    # prob_heat_event = \
    #     1 - (1 - ((heat_indices[0] - threshold/80.0) ** 2) ** (3 * len(heat)))
    prob_heat_event = \
        1 - (1 - ((heat_index - threshold)/80.0) ** 2) ** (3*hours_above_threshold)
    return prob_heat_event


@dataclass
class PlaceDataWithClimate(PlaceData):
    """Place with heat index data."""
    heatIndex: float = float('nan')
    AIR: bool = False


@dataclass
class PersonDataWithHeatRisk(PersonData):
    """Data for a Person."""
    outside_worker: bool
    heatIndices: deque
    probHeatEvent: float


# register the 'casmsocial' model
@register_casmsocial_model('casmsocial_heatrisk_HeatRiskModel')
def create_casmsocial_GeoModel(
    comm: MPI.Intracomm,
    params: dict
) -> Model:
    print("Registering casmsocial_heatrisk_HeatRiskModel model")
    return HeatRiskModel(comm, params)


class HeatRiskModel(GeoModel):
    """ HeatRiskModel class """
    
    def __init__(
        self,
        comm: MPI.Intracomm,
        params: Dict
    ):
        """ Constructor for the HeatRiskModel class """
        # register the place types
        Places.register_place_config(
            PlaceConfig(
                name='Household',
                type=Household,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_hh_id'
            )
        )
        Places.register_place_config(
            PlaceConfig(
                name='School',
                type=School,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_school_id'
            )
        )
        Places.register_place_config(
            PlaceConfig(
                name='Work',
                type=Work,
                dataType=PlaceDataWithClimate,
                personPlaceField='sp_work_id'
            )
        )

        Person.person_data_type = PersonDataWithHeatRisk

        super().__init__(comm, params)

        # load environment file
        # heat_index_file_path = data_input_path / params['heat.index.file']
        self.heatindex_by_hour_place_file_path = \
            self.data_input_path / params['heatIndex.file']
        if self.heatindex_by_hour_place_file_path.exists():
            print(f"Loading heat map places from {self.heatindex_by_hour_place_file_path}")
        else:
            print(f"Error: Heat map places file {self.heatindex_by_hour_place_file_path} not found.")
            exit(1)

        self._heat_threshold = 90.0
        # self._heat_threshold = float(params['heat_threshold'])

        # initialize the heat threshold
        self.heat_indices = deque([float('nan')])

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            comm,
            params['agent_log_file'],
            [
                'tick',
                'agent_id',
                'x',
                'y',
                'heatIndex',
                'hrsAboveHeatThreshold',
                'probHeatEvent'
            ]  # , 'meet_count']
        )
        self.log_agents()

    @property
    def heat_threshold(self) -> float:
        return self._heat_threshold

    def update_environment(self) -> None:
        """Update the environment for the current time step."""
        super().update_environment()

         # update the heat indices
        heatindex_by_hour_place = \
            pd.read_parquet(
                self.heatindex_by_hour_place_file_path,
                engine='pyarrow',
                filters=[("time_hour", "=", self.cal.hour_of_day)]
            ).loc[:, ['sp_id', 'heatIndex']].dropna()
        
        # print(f"size of heatindex_by_hour_place = {len(heatindex_by_hour_place)}")
        minheatindex = heatindex_by_hour_place['heatIndex'].min()
        maxheatindex = heatindex_by_hour_place['heatIndex'].max()
        meanheatindex = heatindex_by_hour_place['heatIndex'].mean()        
        print(
            f"min heat index = {minheatindex}, "
            f"max heat index = {maxheatindex}, "
            f"mean heat index = {meanheatindex}")

        heatIndex_map = \
            heatindex_by_hour_place.set_index('sp_id')['heatIndex'].to_dict()

        # metrics
        countOfHeatIndexMatches = 0
        countOfHeatIncidents = 0
        countOfAirConditionedPlaces = 0
        countOfOutsideWorkers = 0

        for place in self.local_places:

            place.step(self.cal, self.rng)
                
            if place.id in heatIndex_map:
                place.data.heatIndex= heatIndex_map[place.id]
                countOfHeatIndexMatches+=1
            else:
                place.data.heatIndex= meanheatindex

            # Take air conditioned places as 72 degrees    
            if place.data.AIR:
                countOfAirConditionedPlaces += 1
                localHeatIndex = 72
            else:
                localHeatIndex = place.data.heatIndex

            # if len(place.peopleAtPlace) > 0:
            #     print(f"place {place.id} has {len(place.peopleAtPlace)} people")

            for person in place.peopleAtPlace:

                # adjust the heat index for outside workers
                personHeatIndex = localHeatIndex
                if person.state.outside_worker:
                    countOfOutsideWorkers += 1
                    personHeatIndex = place.data.heatIndex

                person.state.heatIndices.appendleft(personHeatIndex)

                person.state.probHeatEvent = compute_prob_heat_event(
                    person.state.heatIndices,
                    self.heat_threshold
                )
                if person.state.probHeatEvent > 0.0001:
                    countOfHeatIncidents += 1

        print(f"number of heat index matches = {countOfHeatIndexMatches}")
        print(f"number of heat incidents = {countOfHeatIncidents}")
        print(f"number of air conditioned places = {countOfAirConditionedPlaces}")
        print(f"number of outside workers = {countOfOutsideWorkers}")

    def log_agents(self) -> None:
        # tick = self.runner.schedule.tick
        tick = self.cal.hour_of_day

        for person in self.context.agents():
            heat = filter_heat_indices(
                person.state.heatIndices,
                self.heat_threshold)
            self.agent_logger.log_row(
                tick,
                person.id,
                person.state.location.x,
                person.state.location.y,
                person.state.heatIndices[0],
                len(heat),
                person.state.probHeatEvent
            )
            # person.uid_rank, person.meet_count)

        self.agent_logger.write()

    def at_end(self) -> None:
        # self.data_set.close()
        self.agent_logger.close()
