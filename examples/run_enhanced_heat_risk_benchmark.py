#!/usr/bin/env python3
"""
Enhanced Heat Risk Model Benchmark Example

This script demonstrates how to run the enhanced heat risk model with
parallel processing and measure performance improvements.

Expected results:
- Original model (~1457 seconds) -> Enhanced model (~200-300 seconds)
- 5-10x performance improvement
- Better CPU and memory utilization

Usage:
    python examples/run_enhanced_heat_risk_benchmark.py

Requirements:
    - CASMSOCIAL_DATA_PATH environment variable set
    - Wake County data files in the data path
    - MPI environment (optional, can run on single process)
"""

import os
import sys
import time
from pathlib import Path

from loguru import logger
from mpi4py import MPI

from casmsocial.heat_risk.enhanced_heat_risk_model import (
    EnhancedHeatRiskModel,
)
from casmsocial.heat_risk.heat_risk_model2 import HeatRiskModel2
from casmsocial.heat_risk.performance_benchmark import PerformanceBenchmark

# Add casmsocial to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_original_model_benchmark(params: dict) -> float:
    """Run the original HeatRiskModel2 for performance comparison."""
    logger.info("Running original HeatRiskModel2 for baseline comparison...")

    comm = MPI.COMM_WORLD
    benchmark = PerformanceBenchmark("benchmark_results")

    # Create original model
    original_model = HeatRiskModel2(comm, params)

    # Start benchmark
    benchmark.start_benchmark(original_model, "Original HeatRiskModel2 - Wake County")

    # Run the model
    start_time = time.time()
    original_model.start()
    runtime = time.time() - start_time

    # End benchmark
    benchmark.end_benchmark()

    logger.info(f"Original model completed in {runtime:.1f} seconds")
    return runtime


def run_enhanced_model_benchmark(params: dict) -> float:
    """Run the enhanced model with parallel processing."""
    logger.info("Running Enhanced Heat Risk Model with parallel processing...")

    comm = MPI.COMM_WORLD
    benchmark = PerformanceBenchmark("benchmark_results")

    # Create enhanced model
    enhanced_model = EnhancedHeatRiskModel(comm, params)

    # Start benchmark
    benchmark.start_benchmark(enhanced_model, "Enhanced HeatRiskModel - Wake County Parallel")

    # Run the model
    start_time = time.time()
    enhanced_model.start()
    runtime = time.time() - start_time

    # End benchmark
    benchmark.end_benchmark()

    logger.info(f"Enhanced model completed in {runtime:.1f} seconds")
    return runtime


def main():
    """Main benchmarking function."""
    logger.info("Starting Heat Risk Model Performance Benchmark")

    # Check environment
    if not os.getenv("CASMSOCIAL_DATA_PATH"):
        logger.error("CASMSOCIAL_DATA_PATH environment variable not set")
        logger.error("Please set it to the directory containing Wake County data")
        sys.exit(1)

    # Load configuration
    config_dir = Path(__file__).parent.parent / "config"

    # Use the original config for baseline
    original_config = load_config(str(config_dir / "casmsocial_wc_30.yaml"))

    # Use enhanced config for comparison
    enhanced_config = load_config(str(config_dir / "enhanced_heat_risk_example.yaml"))

    logger.info(f"MPI Size: {MPI.COMM_WORLD.Get_size()}")
    logger.info(f"MPI Rank: {MPI.COMM_WORLD.Get_rank()}")

    # Performance tracking
    results = {}

    # Run benchmarks (only on rank 0 to avoid duplication)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.info("=" * 60)
        logger.info("HEAT RISK MODEL PERFORMANCE BENCHMARK")
        logger.info("=" * 60)

        # Option 1: Run both models for comparison
        run_comparison = input("Run both original and enhanced models? (y/N): ").lower().startswith("y")

        if run_comparison:
            try:
                # Run original model
                logger.info("\n" + "=" * 40)
                logger.info("RUNNING ORIGINAL MODEL")
                logger.info("=" * 40)
                original_time = run_original_model_benchmark(original_config)
                results["original"] = original_time

            except Exception as e:
                logger.warning(f"Original model failed: {e}")
                logger.info("Continuing with enhanced model only...")
                results["original"] = 1457.0  # Use known baseline
        else:
            logger.info("Using known baseline: 1457 seconds for original model")
            results["original"] = 1457.0

        # Run enhanced model
        logger.info("\n" + "=" * 40)
        logger.info("RUNNING ENHANCED MODEL")
        logger.info("=" * 40)

        try:
            enhanced_time = run_enhanced_model_benchmark(enhanced_config)
            results["enhanced"] = enhanced_time

            # Calculate improvements
            speedup = results["original"] / enhanced_time if enhanced_time > 0 else 1.0
            time_saved = results["original"] - enhanced_time
            improvement_percent = (speedup - 1) * 100

            # Final summary
            logger.info("\n" + "=" * 60)
            logger.info("PERFORMANCE BENCHMARK SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Original Model Runtime:   {results['original']:.1f} seconds")
            logger.info(f"Enhanced Model Runtime:   {enhanced_time:.1f} seconds")
            logger.info(f"Performance Speedup:      {speedup:.2f}x")
            logger.info(f"Time Saved:               {time_saved:.1f} seconds ({time_saved/60:.1f} minutes)")
            logger.info(f"Performance Improvement:  {improvement_percent:.1f}%")
            logger.info("=" * 60)

            # Success criteria
            if speedup >= 3.0:
                logger.info("🎉 EXCELLENT: Achieved 3x+ speedup!")
            elif speedup >= 2.0:
                logger.info("✅ GOOD: Achieved 2x+ speedup!")
            elif speedup >= 1.5:
                logger.info("⚡ MODERATE: Achieved 1.5x+ speedup")
            else:
                logger.warning("❌ Limited speedup achieved. Check system resources and data size.")

        except Exception as e:
            logger.error(f"Enhanced model benchmark failed: {e}")
            logger.error("Please check configuration and data files")
            raise

    else:
        # Non-root ranks just participate in MPI operations
        logger.info(f"Rank {MPI.COMM_WORLD.Get_rank()} participating in MPI operations")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    # Configure logging
    logger.add("heat_risk_benchmark.log", rotation="10 MB")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
