""" Generic Place Class """
from repast4py.space import ContinuousPoint as cpt
import repast4py.core as core
from repast4py.core import SharedProjection
# import repast4py.schedule as schedule
# from repast4py.context import SharedContext

from dataclasses  import dataclass
from typing import (
    Type,
    Dict,
    List,
    NamedTuple
)
import math

# from casmsocial.person import Person
from casmsocial.datautility import create_dataclass_record_from_dict


# 1. Define a PlaceData Class
@dataclass
class PlaceData:
    """Data for a Place."""
    place_type: str = "Household"
    place_name: str = ""
    latitude: float = float('nan')
    longitude: float = float('nan')
    x: float = float('nan')
    y: float = float('nan')


# 2. Define a Place Class
class Place:
    """Generic Place Class"""
    
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        """Constructor for the Place class."""

        placeId = initDict['sp_id']

        # `location` is currently referenced required but not used
        if 'x' not in initDict:
            initDict['x'] = 0
        if 'y' not in initDict:
            initDict['y'] = 0
        if math.isinf(initDict['x']) or math.isinf(initDict['y']):
            initDict['x'] = 0
            initDict['y'] = 0

        self.location = cpt(x=int(initDict['x']), y=int(initDict['y']), z=0)

        self.id = placeId
        self.rank = initDict['rank']

        # create data from initDict
        self.data = \
            create_dataclass_record_from_dict(
                placeDataClass,
                initDict          
            )
    @property
    def pt(self) -> cpt:
        return self.location

    def step(self, calendar, rng):
        pass


# 3. Define a PlaceConfig NamedTuple
PlaceConfig = NamedTuple(
    'PlacesConfig',
    [
        ('name', str),
        ('type', Type[Place]),
        ('dataType', Type[dataclass]),
        ('personPlaceField', str)
    ]
)


# 4. Define a Custom Projection for Agent-Place Association
class PlacesProjection(SharedProjection):
    def __init__(self, name, comm):
        """Constructor for the PlacesProjection class.
        Args:
            name (str): The name of the projection.
            comm (MPI.Comm): The MPI communicator.
        """
        super().__init__(name, comm)
        self.agent_place_map = {}  # Map agents=>place
        self.place_agent_map = {}  # Map Place ID to list of agents
        self.place_map = {}        # Map place_id=>place

        self.rank = comm.Get_rank()

    def add(self, agent) -> None:
        """Add an agent to the projection.
        Args:
            agent (CustomAgent): The agent to add.
        """
        self.agent_place_map[agent.id] = None

    def add_place(self, place: Place) -> None:
        """Add a place to the projection
        Args:
            place (Place): The place to add.
        """
        self.place_map[place.id] = place

    def lookup_place(self, place_id: int) -> Place:
        """Lookup place by place id
        Args:
            place_id (int): The place id to lookup.
        Returns:
            Place: The place object.
        """
        return self.place_map.get(place_id)
    
    def get_local_places(self) -> List[Place]:
        """Get the list of local places.
        Returns:
            List[Place]: The list of local places.
        """
        return [place for place in self.place_map.values() if place.rank == self.rank]  

    def assign_agent_to_place(self, agent, place):
        """Assign an agent to a specific Place."""
        self._remove_agent_from_current_place(agent)
        self._add_agent_to_new_place(agent, place)

    def move_agent_to_place(self, agent, new_place):
        """
        Move an agent from its current place to a new place.
        Args:
            agent (CustomAgent): The agent to move.
            new_place (Place): The new Place object to assign to the agent.
        """
        if agent.id not in self.agent_place_map:
            raise ValueError(f"Agent with ID {agent.id} is not in the projection.")
        print(f"Moving Agent {agent.id} from {self.agent_place_map[agent.id]} to {new_place}")
        self.assign_agent_to_place(agent, new_place)

    def _remove_agent_from_current_place(self, agent):
        """Remove the agent from its current place."""
        current_place = self.agent_place_map.get(agent.id)
        if current_place and current_place.id in self.place_agent_map:
            self.place_agent_map[current_place.id].remove(agent)
            if not self.place_agent_map[current_place.id]:
                del self.place_agent_map[current_place.id]  # Clean up empty place entry

    def _add_agent_to_new_place(self, agent, place):
        """Add the agent to a new place."""
        self.agent_place_map[agent.id] = place
        if place.id not in self.place_agent_map:
            self.place_agent_map[place.id] = []
        self.place_agent_map[place.id].append(agent)

    def get_place_for_agent(self, agent) -> Place:
        return self.agent_place_map.get(agent.id, None)

    def remove(self, agent) -> None:
        """Remove an agent from the projection."""
        place = self.agent_place_map.get(agent.id, None)
        if place and place.id in self.place_agent_map:
            self.place_agent_map[place.id].remove(agent)
        if agent.id in self.agent_place_map:
            del self.agent_place_map[agent.id]

    def get_agents_at_place(self, place) -> List[core.Agent]:
        """
        Retrieve all agents currently assigned to a given Place.
        Args:
            place (Place): The Place object.
        Returns:
            List[Agent]: List of agents at the specified place.
        """
        return self.place_agent_map.get(place.id, [])
    
    def get_agent_by_id(self, agent_id) -> core.Agent:
        if agent_id in self.agent_place_map:
            return next((agent for agent in self.place_agent_map.get(self.agent_place_map[agent_id].id, [])
                        if agent.id == agent_id), None)
        return None
    
    def __repr__(self):
        return f"PlaceProjection(agent_place_map={self.agent_place_map})"


# 5. Define a Remote Place Class
class RemotePlace(Place):
    """ Remote Place class

    A remote place is a place that is outside of the simulation area.
    """
    def __init__(
            self,
            initDict: Dict,
            placeDataClass: Type[dataclass]
        ):
        """Constructor for the RemotePlace class."""
        initDict["place_type"] = "RemotePlace"
        super().__init__(
             initDict,
             placeDataClass
        )