""" Generic Place Class """
import math
from dataclasses import dataclass
from heapq import nsmallest
from typing import NamedTuple, Optional

import repast4py.core as core
from loguru import logger
from repast4py.core import SharedProjection

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

        super().__init__(local_id, Place.TYPE, rank)

        if "rank" not in initDict:
            initDict["rank"] = 0
        self.rank = initDict["rank"]

        # create data from initDict
        self.data = create_dataclass_record_from_dict(placeDataClass, initDict)

        # Initialize occupants set
        self.occupants = set()

    # Legacy methods - for compatibility with old projection
    def add_occupant(self, person) -> None:
        """Add an occupant to the place."""
        self.occupants.add(person)

    def remove_occupant(self, person) -> None:
        """Remove an occupant from the place."""
        self.occupants.discard(person)

    def get_occupants(self) -> set:
        """Get the occupants of the place."""
        return self.occupants

    # New internal methods for enhanced projection
    def _add_occupant_internal(self, agent: core.Agent) -> None:
        """Internal method to add occupant - only called by enhanced projection."""
        self.occupants.add(agent)

    def _remove_occupant_internal(self, agent: core.Agent) -> None:
        """Internal method to remove occupant - only called by enhanced projection."""
        self.occupants.discard(agent)

    # Enhanced convenience methods
    def get_occupant_count(self) -> int:
        """Get the number of occupants at this place."""
        return len(self.occupants)

    def has_occupant(self, agent: core.Agent) -> bool:
        """Check if an agent is currently at this place."""
        return agent in self.occupants

    def get_occupants_by_type(self, agent_type: type) -> set[core.Agent]:
        """Get occupants of a specific type."""
        return {agent for agent in self.occupants if isinstance(agent, agent_type)}


# 3. Define a PlaceConfig NamedTuple
class PlaceConfig(NamedTuple):
    name: str
    place_type: type[Place]
    dataType: PlaceData


# 4. Define a Custom Projection for Agent-Place Association (DEPRECATED - Use EnhancedPlacesProjection)
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


# 4b. Enhanced PlacesProjection - New Implementation
class EnhancedPlacesProjection(SharedProjection):
    """Enhanced PlacesProjection with cleaner agent-place relationship management."""

    def __init__(self, name: str, comm):
        super().__init__(name, comm)
        self.name = name
        self.rank = comm.Get_rank()

        # Core data structures - single source of truth
        self._places: dict[int, Place] = {}  # place_id -> Place
        self._agent_locations: dict[int, int] = {}  # agent_id -> place_id

        # Indexes for efficient lookups (maintained automatically)
        self._local_places: set[int] = set()  # place_ids on this rank

    def add_place(self, place: "Place") -> None:
        """Add a place to the projection."""
        self._places[place.id] = place
        if place.rank == self.rank:
            self._local_places.add(place.id)
        logger.debug(f"Added place {place.id} (rank {place.rank}) to projection")

    def add(self, agent: core.Agent) -> None:
        """Add an agent to the projection (without assigning to a place)."""
        if agent.id in self._agent_locations:
            logger.warning(f"Agent {agent.id} already in projection")
            return
        # Agent is added but not yet assigned to any place
        logger.debug(f"Added agent {agent.id} to projection")

    def assign_agent_to_place(self, agent: core.Agent, place: "Place") -> None:
        """Assign an agent to a place, handling all synchronization automatically."""
        old_place = self.get_place_for_agent(agent)

        # Remove from old place if exists
        if old_place is not None:
            old_place._remove_occupant_internal(agent)

        # Assign to new place
        self._agent_locations[agent.id] = place.id
        place._add_occupant_internal(agent)

        logger.debug(f"Assigned agent {agent.id} to place {place.id}")

    def move_agent_to_place(self, agent: core.Agent, new_place: "Place") -> None:
        """Move an agent from current place to a new place."""
        if agent.id not in self._agent_locations:
            raise AgentNotInProjectionError(agent.id)

        old_place_id = self._agent_locations.get(agent.id)

        logger.debug(f"Moving agent {agent.id} from place {old_place_id} to {new_place.id}")
        self.assign_agent_to_place(agent, new_place)

    def remove(self, agent: core.Agent) -> None:
        """Remove an agent from the projection entirely."""
        if agent.id not in self._agent_locations:
            return

        # Remove from current place
        place_id = self._agent_locations[agent.id]
        place = self._places.get(place_id)
        if place:
            place._remove_occupant_internal(agent)

        # Remove from projection
        del self._agent_locations[agent.id]
        logger.debug(f"Removed agent {agent.id} from projection")

    # Efficient lookup methods
    def get_place_for_agent(self, agent: core.Agent) -> Optional["Place"]:
        """Get the place where an agent is currently located."""
        place_id = self._agent_locations.get(agent.id)
        return self._places.get(place_id) if place_id else None

    def lookup_place(self, place_id: int) -> Optional["Place"]:
        """Get a place by its ID (compatibility method)."""
        return self._places.get(place_id)

    def get_place_by_id(self, place_id: int) -> Optional["Place"]:
        """Get a place by its ID."""
        return self._places.get(place_id)

    def get_agents_at_place(self, place: "Place") -> set[core.Agent]:
        """Get all agents currently at a place."""
        return place.get_occupants()  # Delegate to place's occupants

    def get_local_places(self) -> list["Place"]:
        """Get all places on the current rank."""
        return [self._places[place_id] for place_id in self._local_places]

    def get_all_places(self) -> list["Place"]:
        """Get all places in the projection."""
        return list(self._places.values())

    # Validation and consistency checks
    def validate_consistency(self) -> bool:
        """Validate internal consistency between projection and place occupants."""
        inconsistencies = []

        for agent_id, place_id in self._agent_locations.items():
            place = self._places.get(place_id)
            if place is None:
                inconsistencies.append(f"Agent {agent_id} assigned to non-existent place {place_id}")
                continue

            # Check if place knows about this agent
            if not any(occupant.id == agent_id for occupant in place.get_occupants()):
                inconsistencies.append(f"Agent {agent_id} in projection but not in place {place_id} occupants")

        if inconsistencies:
            logger.error(f"Projection inconsistencies found: {inconsistencies}")
            return False
        return True

    def get_statistics(self) -> dict:
        """Get projection statistics for monitoring."""
        place_occupancy = {}
        for place in self._places.values():
            place_occupancy[place.id] = len(place.get_occupants())

        return {
            "total_places": len(self._places),
            "local_places": len(self._local_places),
            "total_agents": len(self._agent_locations),
            "place_occupancy": place_occupancy,
            "avg_occupancy": sum(place_occupancy.values()) / len(place_occupancy) if place_occupancy else 0,
        }

    def __repr__(self):
        return f"EnhancedPlacesProjection(agents={len(self._agent_locations)}, places={len(self._places)})"


# Convenience alias - makes migration easier
PlacesProjectionV2 = EnhancedPlacesProjection


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
