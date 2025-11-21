"""
Parallel place update system using Numba + Threading hybrid approach.

This module provides high-performance parallel updates for Place objects in casmsocial,
combining Numba's compiled parallel loops with Python threading for coordination.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from loguru import logger
from numba import njit, prange

from casmsocial.place import Place


class ParallelUpdateMetrics:
    """Metrics tracking for parallel update performance."""

    def __init__(self):
        self.update_times = []
        self.places_processed = []
        self.parallel_efficiency = []

    def record_update(self, duration: float, n_places: int, parallel_time: float, sequential_time: float):
        """Record metrics from an update cycle."""
        self.update_times.append(duration)
        self.places_processed.append(n_places)

        if sequential_time > 0:
            efficiency = sequential_time / (parallel_time * os.cpu_count())
            self.parallel_efficiency.append(min(1.0, efficiency))

    def get_average_speedup(self) -> float:
        """Get average speedup from parallelization."""
        if not self.parallel_efficiency:
            return 1.0
        return sum(self.parallel_efficiency) / len(self.parallel_efficiency)


@njit(parallel=True, fastmath=True)
def update_environmental_data_parallel(
    place_ids: np.ndarray,
    temperatures: np.ndarray,
    heat_indices: np.ndarray,
    humidity_values: np.ndarray,
    current_time_minutes: int,
) -> np.ndarray:
    """
    Update environmental data for places in parallel using Numba.

    Args:
        place_ids: Array of place IDs
        temperatures: Current temperatures for each place
        heat_indices: Heat index values for each place
        humidity_values: Humidity percentages for each place
        current_time_minutes: Current simulation time in minutes

    Returns:
        Array of computed risk scores for each place
    """
    n_places = len(place_ids)
    risk_scores = np.empty(n_places, dtype=np.float64)

    for i in prange(n_places):
        heat_idx = heat_indices[i]
        humidity = humidity_values[i]

        # Time-based risk adjustment (higher risk during peak hours 10am-4pm)
        hour_of_day = (current_time_minutes // 60) % 24
        time_factor = 1.0
        if 10 <= hour_of_day <= 16:
            time_factor = 1.5
        elif 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
            time_factor = 1.2

        # Base heat risk calculation
        base_risk = max(0.0, (heat_idx - 25.0) / 15.0)  # Normalized to 0-1+ scale

        # Humidity adjustment (higher humidity increases perceived heat)
        humidity_factor = 1.0 + (humidity - 50.0) / 100.0

        # Combined risk score
        risk_scores[i] = base_risk * humidity_factor * time_factor

    return risk_scores


@njit(parallel=True, fastmath=True)
def update_occupancy_metrics_parallel(
    place_ids: np.ndarray, occupant_counts: np.ndarray, place_capacities: np.ndarray, base_risk_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update occupancy-based metrics for places in parallel.

    Args:
        place_ids: Array of place IDs
        occupant_counts: Current number of occupants in each place
        place_capacities: Maximum capacity of each place
        base_risk_scores: Base environmental risk scores

    Returns:
        Tuple of (occupancy_ratios, adjusted_risk_scores)
    """
    n_places = len(place_ids)
    occupancy_ratios = np.empty(n_places, dtype=np.float64)
    adjusted_risk_scores = np.empty(n_places, dtype=np.float64)

    for i in prange(n_places):
        occupants = occupant_counts[i]
        capacity = max(1.0, place_capacities[i])  # Avoid division by zero
        base_risk = base_risk_scores[i]

        # Calculate occupancy ratio
        occupancy_ratio = occupants / capacity
        occupancy_ratios[i] = occupancy_ratio

        # Risk increases with crowding (non-linear relationship)
        if occupancy_ratio <= 0.7:
            crowding_factor = 1.0
        elif occupancy_ratio <= 1.0:
            crowding_factor = 1.0 + (occupancy_ratio - 0.7) * 2.0  # 1.0 to 1.6
        else:
            crowding_factor = 1.6 + (occupancy_ratio - 1.0) * 3.0  # 1.6+

        adjusted_risk_scores[i] = base_risk * crowding_factor

    return occupancy_ratios, adjusted_risk_scores


@njit(parallel=True, fastmath=True)
def calculate_place_interactions_parallel(
    place_ids: np.ndarray,
    place_coords: np.ndarray,  # Shape: (n_places, 2) - lat, lon
    occupant_counts: np.ndarray,
    interaction_radius_km: float = 2.0,
) -> np.ndarray:
    """
    Calculate interaction potential between nearby places.

    Args:
        place_ids: Array of place IDs
        place_coords: Array of (latitude, longitude) coordinates
        occupant_counts: Number of occupants in each place
        interaction_radius_km: Maximum distance for interactions in km

    Returns:
        Array of interaction scores for each place
    """
    n_places = len(place_ids)
    interaction_scores = np.zeros(n_places, dtype=np.float64)

    for i in prange(n_places):
        lat1 = place_coords[i, 0]
        lon1 = place_coords[i, 1]
        occupants_i = occupant_counts[i]

        local_interaction = 0.0

        for j in range(n_places):
            if i != j:
                lat2 = place_coords[j, 0]
                lon2 = place_coords[j, 1]
                occupants_j = occupant_counts[j]

                # Calculate haversine distance
                distance_km = _haversine_distance_numba(lat1, lon1, lat2, lon2)

                if distance_km <= interaction_radius_km:
                    # Interaction strength decreases with distance
                    distance_factor = 1.0 - (distance_km / interaction_radius_km)
                    interaction_strength = occupants_j * distance_factor
                    local_interaction += interaction_strength

        # Normalize by own occupancy
        if occupants_i > 0:
            interaction_scores[i] = local_interaction / occupants_i
        else:
            interaction_scores[i] = 0.0

    return interaction_scores


@njit(fastmath=True)
def _haversine_distance_numba(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points (Numba-compiled)."""
    R = 6371.0  # Earth radius in km

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class NumbaPlaceUpdater:
    """
    High-performance place updater using Numba compilation + Threading.

    This class coordinates parallel updates of Place objects, using compiled
    Numba kernels for CPU-intensive computations and threading for I/O and
    coordination tasks.
    """

    def __init__(self, max_workers: Optional[int] = None, enable_metrics: bool = True):
        """
        Initialize the parallel place updater.

        Args:
            max_workers: Maximum number of threads (default: CPU count)
            enable_metrics: Whether to track performance metrics
        """
        self.max_workers = max_workers or os.cpu_count()
        self.enable_metrics = enable_metrics
        self.metrics = ParallelUpdateMetrics() if enable_metrics else None

        # Performance tuning parameters
        self.min_places_for_parallel = 20  # Threshold for using parallelization
        self.chunk_size_factor = 0.25  # Chunk size as fraction of total

        logger.info(f"Initialized NumbaPlaceUpdater with {self.max_workers} workers")

    def update_places_parallel(self, places: list[Place], current_time_minutes: int) -> dict[str, Any]:
        """
        Update all places in parallel using Numba + Threading.

        Args:
            places: List of Place objects to update
            current_time_minutes: Current simulation time in minutes

        Returns:
            Dictionary with update results and performance metrics
        """
        start_time = time.time()
        n_places = len(places)

        # Use sequential processing for small numbers of places
        if n_places < self.min_places_for_parallel:
            return self._update_places_sequential(places, current_time_minutes)

        logger.debug(f"Starting parallel update of {n_places} places")

        # Extract data for parallel processing
        place_data = self._extract_place_data(places)

        # Run parallel computations
        parallel_start = time.time()
        computed_results = self._run_parallel_computations(place_data, current_time_minutes)
        parallel_time = time.time() - parallel_start

        # Apply results back to places using threading
        threading_start = time.time()
        self._apply_results_threaded(places, computed_results)
        threading_time = time.time() - threading_start

        total_time = time.time() - start_time

        # Record metrics
        if self.metrics:
            sequential_estimate = n_places * 0.001  # Rough estimate
            self.metrics.record_update(total_time, n_places, parallel_time, sequential_estimate)

        return {
            "places_updated": n_places,
            "total_time": total_time,
            "parallel_time": parallel_time,
            "threading_time": threading_time,
            "speedup_estimate": sequential_estimate / total_time if total_time > 0 else 1.0,
        }

    def _extract_place_data(self, places: list[Place]) -> dict[str, np.ndarray]:
        """Extract numerical data from places for parallel processing."""
        n_places = len(places)

        place_ids = np.zeros(n_places, dtype=np.int64)
        temperatures = np.zeros(n_places, dtype=np.float64)
        heat_indices = np.zeros(n_places, dtype=np.float64)
        humidity_values = np.zeros(n_places, dtype=np.float64)
        occupant_counts = np.zeros(n_places, dtype=np.float64)
        place_capacities = np.zeros(n_places, dtype=np.float64)
        coordinates = np.zeros((n_places, 2), dtype=np.float64)

        for i, place in enumerate(places):
            place_ids[i] = place.id

            # Extract environmental data (with defaults for missing data)
            asdict(place.data) if hasattr(place.data, "__dict__") else {}
            temperatures[i] = getattr(place.data, "T_xy", 25.0)  # Default 25°C
            heat_indices[i] = getattr(place.data, "heat_index", 25.0)
            humidity_values[i] = getattr(place.data, "humidity", 50.0)  # Default 50%

            # Occupancy data
            occupant_counts[i] = place.get_occupant_count()
            place_capacities[i] = getattr(place.data, "capacity", 50.0)  # Default capacity

            # Coordinates
            coordinates[i, 0] = getattr(place.data, "latitude", 0.0)
            coordinates[i, 1] = getattr(place.data, "longitude", 0.0)

        return {
            "place_ids": place_ids,
            "temperatures": temperatures,
            "heat_indices": heat_indices,
            "humidity_values": humidity_values,
            "occupant_counts": occupant_counts,
            "place_capacities": place_capacities,
            "coordinates": coordinates,
        }

    def _run_parallel_computations(
        self, place_data: dict[str, np.ndarray], current_time_minutes: int
    ) -> dict[str, np.ndarray]:
        """Run all parallel computations using Numba kernels."""
        # Environmental risk calculation
        environmental_risks = update_environmental_data_parallel(
            place_data["place_ids"],
            place_data["temperatures"],
            place_data["heat_indices"],
            place_data["humidity_values"],
            current_time_minutes,
        )

        # Occupancy metrics
        occupancy_ratios, adjusted_risks = update_occupancy_metrics_parallel(
            place_data["place_ids"], place_data["occupant_counts"], place_data["place_capacities"], environmental_risks
        )

        # Place interactions
        interaction_scores = calculate_place_interactions_parallel(
            place_data["place_ids"], place_data["coordinates"], place_data["occupant_counts"], interaction_radius_km=2.0
        )

        return {
            "environmental_risks": environmental_risks,
            "occupancy_ratios": occupancy_ratios,
            "adjusted_risks": adjusted_risks,
            "interaction_scores": interaction_scores,
        }

    def _apply_results_threaded(self, places: list[Place], results: dict[str, np.ndarray]):
        """Apply computed results back to places using threading."""
        chunk_size = max(1, len(places) // self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i in range(0, len(places), chunk_size):
                chunk_places = places[i : i + chunk_size]
                chunk_indices = list(range(i, min(i + chunk_size, len(places))))

                future = executor.submit(self._apply_results_to_chunk, chunk_places, chunk_indices, results)
                futures.append(future)

            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions

    def _apply_results_to_chunk(self, places: list[Place], indices: list[int], results: dict[str, np.ndarray]):
        """Apply results to a chunk of places."""
        for place, idx in zip(places, indices):
            # Create or update computed attributes on place.data
            if not hasattr(place, "computed_metrics"):
                place.computed_metrics = {}

            place.computed_metrics.update(
                {
                    "environmental_risk": float(results["environmental_risks"][idx]),
                    "occupancy_ratio": float(results["occupancy_ratios"][idx]),
                    "adjusted_risk": float(results["adjusted_risks"][idx]),
                    "interaction_score": float(results["interaction_scores"][idx]),
                    "last_updated": time.time(),
                }
            )

    def _update_places_sequential(self, places: list[Place], current_time_minutes: int) -> dict[str, Any]:
        """Fallback sequential update for small numbers of places."""
        start_time = time.time()

        for place in places:
            if not hasattr(place, "computed_metrics"):
                place.computed_metrics = {}

            # Simple sequential calculations
            heat_idx = getattr(place.data, "heat_index", 25.0)
            occupants = place.get_occupant_count()

            base_risk = max(0.0, (heat_idx - 25.0) / 15.0)
            occupancy_ratio = occupants / max(1.0, getattr(place.data, "capacity", 50.0))

            place.computed_metrics.update(
                {
                    "environmental_risk": base_risk,
                    "occupancy_ratio": occupancy_ratio,
                    "adjusted_risk": base_risk * (1.0 + occupancy_ratio),
                    "interaction_score": 0.0,  # Skip expensive calculation for small sets
                    "last_updated": time.time(),
                }
            )

        return {
            "places_updated": len(places),
            "total_time": time.time() - start_time,
            "parallel_time": 0.0,
            "threading_time": 0.0,
            "speedup_estimate": 1.0,
        }

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics from the updater."""
        if not self.metrics:
            return {}

        return {
            "average_speedup": self.metrics.get_average_speedup(),
            "total_updates": len(self.metrics.update_times),
            "average_update_time": (
                sum(self.metrics.update_times) / len(self.metrics.update_times) if self.metrics.update_times else 0.0
            ),
            "total_places_processed": sum(self.metrics.places_processed),
        }
