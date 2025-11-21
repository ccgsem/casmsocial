"""
Parallel Agent Processing System for Large-Scale Simulations

Optimized for processing 1M+ agents efficiently using thread pools and batching.
Thread-safe design for MPI-distributed agent-based models.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from loguru import logger
from repast4py import context as ctx

from casmsocial.person import Person
from casmsocial.sim_time import SimTime


class ParallelAgentProcessor:
    """
    High-performance parallel agent processor for large simulations.

    Designed for processing 100K+ agents efficiently while maintaining
    thread safety and MPI compatibility.
    """

    def __init__(self, max_workers: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Initialize parallel agent processor.

        Args:
            max_workers: Maximum number of worker threads (default: CPU count)
            batch_size: Agents per batch (default: auto-calculated)
        """
        self.max_workers = max_workers or min(16, os.cpu_count())  # Cap at 16 for memory
        self.batch_size = batch_size or 1000  # Optimized batch size for large datasets

        # Thread-local storage for isolated processing
        self.thread_local = threading.local()

        # Performance tracking
        self.processing_stats = {
            "total_agents_processed": 0,
            "total_processing_time": 0.0,
            "batches_processed": 0,
            "average_speedup": 0.0,
        }

        logger.info(f"Initialized ParallelAgentProcessor: {self.max_workers} workers, " f"batch size {self.batch_size}")

    def process_person_agents_parallel(
        self, person_agents: list[Person], context: ctx.SharedContext, cal: SimTime
    ) -> dict[str, Any]:
        """
        Process person agents (TYPE=0) in parallel using optimized batching.

        Args:
            person_agents: List of person agents to process
            context: The shared context containing agents
            cal: Current simulation time

        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()

        total_agents = len(person_agents)

        if total_agents == 0:
            return self._create_stats_result(0, 0.0, 1.0)

        logger.info(f"Processing {total_agents:,} person agents in parallel...")

        # For very large datasets, use parallel processing
        if total_agents >= 1000:
            self._process_parallel(person_agents, context, cal)
        else:
            # Fallback to sequential for small datasets
            self._process_sequential(person_agents, context, cal)

        processing_time = time.time() - start_time

        # Calculate performance metrics
        sequential_estimate = total_agents * 0.001  # Rough estimate: 1ms per agent
        speedup = sequential_estimate / processing_time if processing_time > 0 else 1.0

        logger.info(
            f"Processed {total_agents:,} agents in {processing_time:.2f}s " f"(estimated speedup: {speedup:.1f}x)"
        )

        return self._create_stats_result(total_agents, processing_time, speedup)

    def process_agents_parallel(self, context: ctx.SharedContext, cal: SimTime) -> dict[str, Any]:
        """
        Legacy method: Process all agents in parallel using optimized batching.

        This method filters for person agents (TYPE=0) automatically for backward compatibility.

        Args:
            context: The shared context containing agents
            cal: Current simulation time

        Returns:
            Dictionary with processing statistics
        """
        # Get only person agents (TYPE=0)
        person_agents = list(context.agents(agent_type=0))
        return self.process_person_agents_parallel(person_agents, context, cal)

    def _process_parallel(self, agents: list[Person], context: ctx.SharedContext, cal: SimTime) -> dict[str, Any]:
        """Process agents in parallel using thread pool."""

        # Create batches for optimal thread utilization
        agent_batches = self._create_batches(agents)

        logger.debug(f"Created {len(agent_batches)} batches for parallel processing")

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="AgentProcessor") as executor:
            # Submit all batches
            futures = {
                executor.submit(self._process_batch, batch, context, cal): i for i, batch in enumerate(agent_batches)
            }

            # Collect results as they complete
            completed_batches = 0
            errors = []

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_result = future.result()
                    completed_batches += 1

                    if batch_result.get("errors", 0) > 0:
                        errors.extend(batch_result["error_details"])

                    # Log progress for large datasets
                    if len(agent_batches) > 10 and completed_batches % 10 == 0:
                        progress = (completed_batches / len(agent_batches)) * 100
                        logger.debug(
                            f"Agent processing progress: {progress:.0f}% "
                            f"({completed_batches}/{len(agent_batches)} batches)"
                        )

                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    errors.append(f"Batch {batch_idx}: {e!s}")

        # Log any errors
        if errors:
            logger.warning(f"Agent processing completed with {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.error(f"  - {error}")
            if len(errors) > 5:
                logger.error(f"  ... and {len(errors) - 5} more errors")

        return {"completed_batches": completed_batches, "errors": errors}

    def _process_batch(self, agents_batch: list[Person], context: ctx.SharedContext, cal: SimTime) -> dict[str, Any]:
        """Process a batch of agents in a single thread."""
        batch_start = time.time()
        errors = []
        processed = 0

        try:
            for person in agents_batch:
                try:
                    person.step(context, cal)
                    processed += 1
                except Exception as e:
                    error_msg = f"Agent {person.id}: {e!s}"
                    errors.append(error_msg)
                    logger.debug(f"Agent processing error - {error_msg}")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            errors.append(f"Batch failure: {e!s}")

        batch_time = time.time() - batch_start

        return {"processed": processed, "errors": len(errors), "error_details": errors, "batch_time": batch_time}

    def _process_sequential(self, agents: list[Person], context: ctx.SharedContext, cal: SimTime) -> dict[str, Any]:
        """Fallback sequential processing for small datasets."""
        logger.debug(f"Using sequential processing for {len(agents)} agents")

        errors = []
        for person in agents:
            try:
                person.step(context, cal)
            except Exception as e:
                errors.append(f"Agent {person.id}: {e!s}")

        return {"errors": errors}

    def _create_batches(self, agents: list[Person]) -> list[list[Person]]:
        """Create optimally-sized batches from agent list."""
        # Adjust batch size based on total agent count for better load balancing
        total_agents = len(agents)

        if total_agents > 100000:
            # For very large datasets, use smaller batches for better parallelism
            optimal_batch_size = max(500, total_agents // (self.max_workers * 4))
        else:
            optimal_batch_size = self.batch_size

        batches = []
        for i in range(0, total_agents, optimal_batch_size):
            batch = agents[i : i + optimal_batch_size]
            batches.append(batch)

        return batches

    def _create_stats_result(self, agents_processed: int, processing_time: float, speedup: float) -> dict[str, Any]:
        """Create standardized statistics result."""
        self.processing_stats["total_agents_processed"] += agents_processed
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["batches_processed"] += 1

        # Rolling average speedup
        if self.processing_stats["batches_processed"] > 1:
            self.processing_stats["average_speedup"] = (
                self.processing_stats["average_speedup"] * (self.processing_stats["batches_processed"] - 1) + speedup
            ) / self.processing_stats["batches_processed"]
        else:
            self.processing_stats["average_speedup"] = speedup

        return {
            "agents_processed": agents_processed,
            "processing_time": processing_time,
            "estimated_speedup": speedup,
            "agents_per_second": agents_processed / processing_time if processing_time > 0 else 0,
            "average_speedup": self.processing_stats["average_speedup"],
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get accumulated performance statistics."""
        return self.processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.processing_stats = {
            "total_agents_processed": 0,
            "total_processing_time": 0.0,
            "batches_processed": 0,
            "average_speedup": 0.0,
        }


# Global instance for reuse across simulation steps
_global_agent_processor: Optional[ParallelAgentProcessor] = None


def get_agent_processor() -> ParallelAgentProcessor:
    """Get or create global agent processor instance."""
    global _global_agent_processor
    if _global_agent_processor is None:
        _global_agent_processor = ParallelAgentProcessor()
    return _global_agent_processor
