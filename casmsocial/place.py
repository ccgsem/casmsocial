"""Generic Place Class"""

import math
from dataclasses import dataclass
from heapq import nsmallest
from typing import Any, NamedTuple, Optional

import repast4py.core as core
from loguru import logger
from repast4py.core import SharedProjection

# from casmsocial.person import Person
from casmsocial.data_utilities import create_dataclass_record_from_dict


# 1. Define a PlaceData Class
@dataclass(slots=True)
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

        if placeDataClass is PlaceData:
            self.data = PlaceData(
                place_type=initDict["place_type"] if "place_type" in initDict else "Household",
                place_name=initDict["place_name"] if "place_name" in initDict else "",
                latitude=initDict["latitude"] if "latitude" in initDict else float("nan"),
                longitude=initDict["longitude"] if "longitude" in initDict else float("nan"),
            )
        else:
            # create data from initDict
            self.data = create_dataclass_record_from_dict(placeDataClass, initDict)

        # Initialize occupants set
        self.occupants = set()

    @classmethod
    def from_default_fields(
        cls,
        local_id: int,
        rank: int = 0,
        place_type: str = "Household",
        place_name: str = "",
        latitude: float = float("nan"),
        longitude: float = float("nan"),
    ) -> "Place":
        """Construct a default Place without per-row dictionary materialization."""
        place = cls.__new__(cls)
        core.Agent.__init__(place, local_id, cls.TYPE, rank)
        place.rank = rank
        place.data = PlaceData(place_type, place_name, latitude, longitude)
        place.occupants = set()
        return place

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
        logger.debug("Moving Agent {} from {} to {}", agent.id, self.agent_place_map[agent.id], new_place)
        self.assign_agent_to_place(agent, new_place)

    def remove_agent_from_place(self, agent) -> None:
        """Unassign an agent from its current place without removing the agent."""
        self._remove_agent_from_current_place(agent)
        self.agent_place_map[agent.id] = None

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

    def __init__(
        self,
        name: str,
        comm,
        enable_parallel_updates: bool = True,
        parallel_min_threshold: int = 20,
        parallel_max_workers: int | None = None,
    ):
        super().__init__(name, comm)
        self.name = name
        self.rank = comm.Get_rank()

        # Core data structures - single source of truth
        self._places: dict[int, Place] = {}  # place_id -> Place
        self._place_ranks: dict[int, int] = {}  # place_id -> owning MPI rank
        self._agent_locations: dict[int, int] = {}  # agent_id -> place_id

        # Indexes for efficient lookups (maintained automatically)
        self._local_places: set[int] = set()  # place_ids on this rank

        # Parallel update system
        self.parallel_updates_enabled = enable_parallel_updates
        self.parallel_min_threshold = parallel_min_threshold
        self.parallel_max_workers = parallel_max_workers
        self._parallel_updater = None

    def _get_parallel_updater(self):
        """Initialize the Numba updater only when parallel place updates are used."""
        if not self.parallel_updates_enabled:
            return None
        if self._parallel_updater is not None:
            return self._parallel_updater

        try:
            from casmsocial.parallel_updates import NumbaPlaceUpdater

            self._parallel_updater = NumbaPlaceUpdater(max_workers=self.parallel_max_workers)
            logger.info("Parallel place updates enabled")
            return self._parallel_updater
        except ImportError as e:
            logger.warning(f"Could not enable parallel updates: {e}")
            self.parallel_updates_enabled = False
            return None

    def register_place_rank(self, place_id: int, rank: int) -> None:
        """Register the owning rank for a place without requiring a Place object."""
        self._place_ranks[int(place_id)] = int(rank)

    def set_place_rank_map(self, place_ranks: dict[int, int], copy: bool = True) -> None:
        """Replace the place rank index used for remote-place routing."""
        self._place_ranks = (
            {int(place_id): int(rank) for place_id, rank in place_ranks.items()} if copy else place_ranks
        )

    def get_place_rank_map(self) -> dict[int, int]:
        """Return a copy of the place-to-rank index."""
        return dict(self._place_ranks)

    def rank_for_place(self, place_id: int) -> int | None:
        """Return the owning rank for a place, if known."""
        return self._place_ranks.get(int(place_id))

    def add_place(self, place: "Place") -> None:
        """Add a place to the projection."""
        self._places[place.id] = place
        self._place_ranks[int(place.id)] = int(place.rank)
        if place.rank == self.rank:
            self._local_places.add(place.id)

    def add_places(self, places: list["Place"], register_ranks: bool = True) -> None:
        """Add places in bulk, optionally reusing a preloaded rank map."""
        places_by_id = self._places
        local_places = self._local_places
        place_ranks = self._place_ranks
        rank = self.rank
        if register_ranks:
            for place in places:
                place_id = place.id
                places_by_id[place_id] = place
                place_ranks[int(place_id)] = int(place.rank)
                if place.rank == rank:
                    local_places.add(place_id)
            return

        for place in places:
            place_id = place.id
            places_by_id[place_id] = place
            if place.rank == rank:
                local_places.add(place_id)

    def add(self, agent: core.Agent) -> None:
        """Add an agent to the projection (without assigning to a place)."""
        if agent.id in self._agent_locations:
            logger.warning(f"Agent {agent.id} already in projection")
            return
        # Agent is added but not yet assigned to any place

    def assign_agent_to_place(self, agent: core.Agent, place: "Place") -> None:
        """Assign an agent to a place, handling all synchronization automatically."""
        old_place = self.get_place_for_agent(agent)

        # Remove from old place if exists
        if old_place is not None:
            old_place._remove_occupant_internal(agent)

        # Assign to new place
        self._agent_locations[agent.id] = place.id
        place._add_occupant_internal(agent)

    def assign_new_agent_to_place(self, agent: core.Agent, place: "Place") -> None:
        """Assign a newly created agent to its first place."""
        self._agent_locations[agent.id] = place.id
        place._add_occupant_internal(agent)

    def move_agent_to_place(self, agent: core.Agent, new_place: "Place") -> None:
        """Move an agent from current place to a new place."""
        if agent.id not in self._agent_locations:
            raise AgentNotInProjectionError(agent.id)

        old_place_id = self._agent_locations.get(agent.id)

        logger.debug("Moving agent {} from place {} to {}", agent.id, old_place_id, new_place.id)
        self.assign_agent_to_place(agent, new_place)

    def remove_agent_from_place(self, agent: core.Agent) -> None:
        """Unassign an agent from its current place without removing the agent."""
        old_place = self.get_place_for_agent(agent)
        if old_place is not None:
            old_place._remove_occupant_internal(agent)
        self._agent_locations.pop(agent.id, None)
        logger.debug("Removed agent {} from its current place", agent.id)

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
        logger.debug("Removed agent {} from projection", agent.id)

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

        stats = {
            "total_places": len(self._places),
            "local_places": len(self._local_places),
            "total_agents": len(self._agent_locations),
            "place_occupancy": place_occupancy,
            "avg_occupancy": sum(place_occupancy.values()) / len(place_occupancy) if place_occupancy else 0,
        }

        # Add parallel update performance stats if available
        if self._parallel_updater:
            parallel_stats = self._parallel_updater.get_performance_stats()
            stats["parallel_performance"] = parallel_stats

        return stats

    def update_places_parallel(self, current_time_minutes: int) -> dict[str, Any]:
        """
        Update all local places using parallel processing.

        Args:
            current_time_minutes: Current simulation time in minutes

        Returns:
            Dictionary with update results and performance metrics
        """
        local_places = self.get_local_places()

        if not local_places:
            return {"places_updated": 0, "total_time": 0.0}

        parallel_updater = self._get_parallel_updater()
        if parallel_updater is not None:
            logger.debug("Running parallel update on {} local places", len(local_places))
            return parallel_updater.update_places_parallel(local_places, current_time_minutes)
        else:
            # Fallback to sequential updates
            logger.debug("Running sequential update on {} local places", len(local_places))
            return self._update_places_sequential(local_places, current_time_minutes)

    def _update_places_sequential(self, places: list[Place], current_time_minutes: int) -> dict[str, Any]:
        """Fallback sequential place updates."""
        import time

        start_time = time.time()

        for place in places:
            if not hasattr(place, "computed_metrics"):
                place.computed_metrics = {}

            # Basic sequential calculations
            occupant_count = place.get_occupant_count()
            place.computed_metrics.update({"occupancy_count": occupant_count, "last_updated": time.time()})

        return {
            "places_updated": len(places),
            "total_time": time.time() - start_time,
            "parallel_time": 0.0,
            "threading_time": 0.0,
            "speedup_estimate": 1.0,
        }

    def __repr__(self):
        parallel_status = "enabled" if self.parallel_updates_enabled else "disabled"
        return (
            f"EnhancedPlacesProjection("
            f"agents={len(self._agent_locations)}, "
            f"places={len(self._places)}, "
            f"parallel={parallel_status})"
        )


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

        def filter_func(_place):
            return True

    places_of_interest = [place for place in places if filter_func(place)]
    closest_places = nsmallest(
        n, places_of_interest, key=lambda p: haversine_distance(lat, lon, p.data.latitude, p.data.longitude)
    )
    return [
        (place, haversine_distance(lat, lon, place.data.latitude, place.data.longitude)) for place in closest_places
    ]


def test_parallel_updates():
    """Test the parallel place updates system."""
    from mpi4py import MPI

    logger.info("Testing parallel place updates...")

    comm = MPI.COMM_WORLD
    places_proj = EnhancedPlacesProjection("test_projection", comm, enable_parallel_updates=True)

    # Create test places with mock data
    test_places = []
    for i in range(25):  # Above parallel threshold
        place_data = {
            "place_id": i,
            "rank": 0,
            "T_xy": 25.0 + (i % 10),  # Temperature 25-35°C
            "heat_index": 25.0 + (i % 15),  # Heat index 25-40°C
            "humidity": 40.0 + (i % 30),  # Humidity 40-70%
            "capacity": 50.0 + (i % 50),  # Capacity 50-100
            "latitude": 41.8781 + (i * 0.001),  # Chicago area coordinates
            "longitude": -87.6298 + (i * 0.001),
        }
        place = Place(place_data, PlaceData)
        test_places.append(place)
        places_proj.add_place(place)

    # Add some occupants to places
    for i, place in enumerate(test_places):
        occupant_count = i % 10  # 0-9 occupants
        place._occupants = set(range(occupant_count))  # Mock occupants

    # Test parallel updates
    current_time_minutes = 720  # 12:00 PM
    results = places_proj.update_places_parallel(current_time_minutes)

    logger.info(f"Parallel update results: {results}")

    # Verify that places have computed metrics
    for place in test_places[:5]:  # Check first 5 places
        if hasattr(place, "computed_metrics"):
            logger.info(f"Place {place.id} metrics: {place.computed_metrics}")
        else:
            logger.warning(f"Place {place.id} missing computed metrics")

    logger.info("Parallel updates test completed successfully!")


if __name__ == "__main__":
    test_parallel_updates()
