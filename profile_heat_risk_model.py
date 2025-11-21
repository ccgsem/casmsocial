#!/usr/bin/env python3
"""Performance profiling script for EnhancedHeatRiskModel.

This script profiles the model execution to identify performance bottlenecks.
Collects:
- CPU time per function (cProfile)
- Memory usage
- Wall-clock time per phase
- I/O statistics
"""

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

from loguru import logger
from repast4py.parameters import init_params

from casmsocial.factory import Models

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


class PerformanceProfiler:
    """Comprehensive performance profiler for the heat risk model."""

    def __init__(self, model_config: str):
        """Initialize profiler with model configuration."""
        self.model_config = model_config
        self.metrics = {"setup_time": 0, "simulation_time": 0, "total_time": 0, "peak_memory": 0, "phase_times": {}}

    def profile_model_run(self, num_steps: int = 96):
        """Profile the full model run with cProfile."""
        logger.info("Starting performance profiling...")

        # Create profiler
        profiler = cProfile.Profile()
        tracemalloc.start()

        # Setup phase
        setup_start = time.time()
        params = init_params(self.model_config)
        model_class = Models.model(params)
        model = model_class(params)
        setup_end = time.time()

        self.metrics["setup_time"] = setup_end - setup_start
        logger.info(f"Setup Time: {self.metrics['setup_time']:.2f}s")

        # Simulation phase with profiling
        sim_start = time.time()
        profiler.enable()

        try:
            model.start()
            for step in range(num_steps):
                step_start = time.time()
                model.step()
                step_time = time.time() - step_start

                if (step + 1) % 10 == 0:
                    logger.info(f"Step {step + 1}/{num_steps} completed in {step_time:.2f}s")
        finally:
            profiler.disable()
            model.end()

        sim_end = time.time()
        self.metrics["simulation_time"] = sim_end - sim_start
        self.metrics["total_time"] = sim_end - setup_start

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        self.metrics["peak_memory"] = peak / 1024 / 1024  # MB
        tracemalloc.stop()

        # Generate report
        return self._generate_report(profiler)

    def _generate_report(self, profiler):
        """Generate comprehensive performance report."""
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE PROFILING REPORT")
        logger.info("=" * 80)

        # Timing summary
        logger.info("\n### TIMING SUMMARY ###")
        logger.info(f"Setup Time:        {self.metrics['setup_time']:>10.2f}s")
        logger.info(f"Simulation Time:   {self.metrics['simulation_time']:>10.2f}s")
        logger.info(f"Total Time:        {self.metrics['total_time']:>10.2f}s")

        # Memory summary
        logger.info("\n### MEMORY USAGE ###")
        logger.info(f"Peak Memory:       {self.metrics['peak_memory']:>10.1f} MB")

        # CPU profiling stats
        logger.info("\n### TOP 20 FUNCTIONS BY CUMULATIVE TIME ###")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        logger.info(s.getvalue())

        logger.info("\n### TOP 20 FUNCTIONS BY TOTAL TIME ###")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("time")
        ps.print_stats(20)
        logger.info(s.getvalue())

        logger.info("\n### CASMSOCIAL-SPECIFIC FUNCTION TIMES ###")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_callers("casmsocial")
        logger.info(s.getvalue())

        # Calculate bottleneck analysis
        logger.info("\n### BOTTLENECK ANALYSIS ###")
        self._analyze_bottlenecks(profiler)

        logger.info("=" * 80)
        return self.metrics

    def _analyze_bottlenecks(self, profiler):
        """Analyze profiling data to identify bottlenecks."""
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")

        # Get top functions
        top_functions = []
        for func, stat in stats.stats.items():
            # stat = (primitive_call_count, num_calls, total_time, cumulative_time, callers)
            cumulative = stat[3]
            total = stat[2]
            if cumulative > 0:
                top_functions.append((func, cumulative, total))

        top_functions.sort(key=lambda x: x[1], reverse=True)

        total_time = sum(f[1] for f in top_functions)

        logger.info(f"\nTotal Time Accounted: {total_time:.2f}s")
        logger.info("\nTop Bottlenecks (>5% of total time):")

        for func, cum_time, _total_time_func in top_functions[:10]:
            filename, lineno, funcname = func
            percentage = (cum_time / total_time) * 100
            if percentage > 5:
                logger.info(f"  {percentage:>5.1f}% - {funcname} ({filename}:{lineno})")

        # Category analysis
        logger.info("\n### TIME DISTRIBUTION BY CATEGORY ###")
        categories = {"I/O": 0, "Computation": 0, "MPI": 0, "Other": 0}

        for func, cum_time, _ in top_functions:
            filename = func[0]
            if "parquet" in filename or "arrow" in filename or "polars" in filename:
                categories["I/O"] += cum_time
            elif "mpi" in filename.lower():
                categories["MPI"] += cum_time
            elif "heat_risk" in filename or "casmpop" in filename:
                categories["Computation"] += cum_time
            else:
                categories["Other"] += cum_time

        total = sum(categories.values())
        for cat, time_val in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = (time_val / total * 100) if total > 0 else 0
            logger.info(f"  {cat:>15}: {time_val:>8.2f}s ({pct:>5.1f}%)")


def main():
    """Run profiling."""
    config_file = "config/enhanced_heat_risk_example.yaml"

    if not Path(config_file).exists():
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    profiler = PerformanceProfiler(config_file)

    # Run with reduced number of steps for faster profiling (24 steps = 6 hours)
    metrics = profiler.profile_model_run(num_steps=24)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Execution Time: {metrics['total_time']:.2f} seconds")
    logger.info(f"Peak Memory Usage: {metrics['peak_memory']:.1f} MB")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
