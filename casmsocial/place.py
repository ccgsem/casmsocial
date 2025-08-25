""" Generic Place Class """
import math
from dataclasses import dataclass
from heapq import nsmallest
from typing import NamedTuple

import repast4py.core as core
from loguru import logger
from repast4py.core import SharedProjection
from repast4py.space import ContinuousPoint as cpt

# from casmsocial.person import Person
from casmsocial.data_utilities import create_dataclass_record_from_dict


# 1. Define a PlaceData Class
@dataclass
class PlaceData:
    """Data for a Place."""

    place_type: str = "Household"
    place_name: str = ""
    latitude: float = float("nan")
    longitude: float = float("nan")
    x: float = float("nan")
    y: float = float("nan")


# 2. Define a Place Class
class Place(core.Agent):
    """Generic Place Class"""

    TYPE = 1
    __place_data_class: type[dataclass] = PlaceData

    @classmethod
    def getPlaceDataClass(cls) -> type[dataclass]:
        """Get the place data class."""
        return cls.__place_data_class

    @classmethod
    def setPlaceDataClass(cls, place_data_class: type[dataclass]) -> None:
        """Set the place data class."""
        cls.__place_data_class = place_data_class

    def __init__(self, initDict: dict, placeDataClass: type[dataclass]):
        """Constructor for the Place class."""
        local_id = initDict.get("sp_id")
        rank = initDict.get("rank", 0)

        super().__init__(local_id, rank)

        # `location` is currently referenced required but not used
        if "x" not in initDict:
            initDict["x"] = 0
        if "y" not in initDict:
            initDict["y"] = 0
        if math.isinf(initDict["x"]) or math.isinf(initDict["y"]):
            initDict["x"] = 0
            initDict["y"] = 0

        self.location = cpt(x=int(initDict["x"]), y=int(initDict["y"]), z=0)

        if "rank" not in initDict:
            initDict["rank"] = 0
        self.rank = initDict["rank"]

        # create data from initDict
        self.data = create_dataclass_record_from_dict(placeDataClass, initDict)

        # Initialize occupants set
        self.occupants = set()

    @property
    def pt(self) -> cpt:
        return self.location

    def add_occupant(self, person) -> None:
        """Add an occupant to the place."""
        self.occupants.add(person)

    def remove_occupant(self, person) -> None:
        """Remove an occupant from the place."""
        self.occupants.discard(person)

    def get_occupants(self) -> set:
        """Get the occupants of the place."""
        return self.occupants


# 3. Define a PlaceConfig NamedTuple
class PlaceConfig(NamedTuple):
    name: str
    place_type: type[Place]
    dataType: PlaceData


# 4. Define a Custom Projection for Agent-Place Association
class PlacesProjection(SharedProjection):
    def __init__(self, name, comm):
        """Constructor for the PlacesProjection class.
        Args:
            name (str): The name of the projection.
            comm (MPI.Comm): The MPI communicator.
        """
        super().__init__(name, comm)
        self.name = name
        self.agent_place_map = {}  # Map agents=>place
        self.place_agent_map = {}  # Map Place ID to list of agents
        self.place_map = {}  # Map place_id=>place

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

    def get_local_places(self) -> list[Place]:
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
            raise AgentNotInProjectionError(agent.id)
        logger.debug(f"Moving Agent {agent.id} from {self.agent_place_map[agent.id]} to {new_place}")
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

    def get_agents_at_place(self, place) -> list[core.Agent]:
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
            return next(
                (
                    agent
                    for agent in self.place_agent_map.get(self.agent_place_map[agent_id].id, [])
                    if agent.id == agent_id
                ),
                None,
            )
        return None

    def __repr__(self):
        return f"PlaceProjection(agent_place_map={self.agent_place_map})"


# 5. Define a custom exception for agent not in projection
class AgentNotInProjectionError(ValueError):
    def __init__(self, agent_id):
        super().__init__(f"Agent with ID {agent_id} is not in the projection.")


# 6. Define a Remote Place Class
class RemotePlace(Place):
    """Remote Place class

    A remote place is a place that is outside of the simulation area.
    """

    def __init__(self, initDict: dict, placeDataClass: type[dataclass]):
        """Constructor for the RemotePlace class."""
        initDict["place_type"] = "RemotePlace"
        super().__init__(initDict, placeDataClass)


# utility functions for places
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in km


def find_closest_location(lat, lon, places, n=3, filter_func=None):
    """
    Find the `n` closest places to the given coordinates.

    Args:
        lat (float): Latitude of the target location.
        lon (float): Longitude of the target location.
        places (list[Place]): List of Place objects.
        n (int): Number of closest places to return (default is 3).
        filter_func (function): Function to filter places (default is None).

    Returns:
        list[tuple[Place, float]]: List of tuples containing Place objects and their distances.
    """
    if filter_func is None:
        filter_func = lambda p: True

    places_of_interest = [place for place in places if filter_func(place)]
    closest_places = nsmallest(
        n, places_of_interest, key=lambda p: haversine_distance(lat, lon, p.data.latitude, p.data.longitude)
    )
    return [
        (place, haversine_distance(lat, lon, place.data.latitude, place.data.longitude)) for place in closest_places
    ]
