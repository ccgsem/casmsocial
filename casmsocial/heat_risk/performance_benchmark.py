"""
Performance benchmarking system for heat risk model parallelization.

This module provides comprehensive benchmarking tools to measure and compare
the performance improvements from parallelization enhancements.
"""

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
from loguru import logger
from mpi4py import MPI


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""

    # System information
    system_info: dict[str, Any]

    # Model configuration
    model_name: str
    simulation_duration_hours: int
    time_step_minutes: int
    num_agents: int
    num_places: int

    # Performance metrics
    total_runtime_seconds: float
    total_steps: int
    average_step_time: float
    memory_usage_mb: float
    cpu_utilization_percent: float

    # Parallelization metrics
    parallel_enabled: bool
    max_workers: int
    weather_processing_time: float
    agent_processing_time: float
    db_query_time: float
    parallel_efficiency: float
    estimated_speedup: float

    # Comparison metrics (vs baseline)
    baseline_runtime: Optional[float] = None
    actual_speedup: Optional[float] = None
    time_saved_seconds: Optional[float] = None

    # Timestamp
    benchmark_timestamp: str = ""


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking system for heat risk models.

    This class provides tools to measure, compare, and analyze performance
    improvements from parallelization enhancements.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmarking system.

        Args:
            output_dir: Directory to store benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = None
        self.end_time = None
        self.system_metrics = {}

        # Known baseline performance (original implementation)
        self.baseline_runtime = 1457.0  # seconds

        logger.info(f"Performance benchmark initialized, results will be saved to {self.output_dir}")

    def start_benchmark(self, model_instance, description: str = ""):
        """
        Start performance monitoring for a model run.

        Args:
            model_instance: The model instance being benchmarked
            description: Optional description of the benchmark
        """
        self.start_time = time.time()
        self.model_instance = model_instance
        self.description = description

        # Capture system information
        self.system_metrics = self._capture_system_info()

        # Initial memory measurement
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Benchmark started: {description}")
        logger.info(
            f"System: {self.system_metrics['cpu_count']} cores, " f"{self.system_metrics['total_memory_gb']:.1f}GB RAM"
        )

    def end_benchmark(self) -> BenchmarkResult:
        """
        End performance monitoring and generate results.

        Returns:
            BenchmarkResult with comprehensive performance metrics
        """
        if self.start_time is None:
            msg = "Benchmark not started. Call start_benchmark() first."
            raise ValueError(msg)

        self.end_time = time.time()
        total_runtime = self.end_time - self.start_time

        # Capture final system state
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = max(final_memory - self.initial_memory, final_memory)

        # Get model-specific metrics
        model_metrics = self._extract_model_metrics()

        # Calculate performance improvements
        actual_speedup = self.baseline_runtime / total_runtime if total_runtime > 0 else 1.0
        time_saved = max(0, self.baseline_runtime - total_runtime)

        # Create benchmark result
        result = BenchmarkResult(
            system_info=self.system_metrics,
            model_name=model_metrics["model_name"],
            simulation_duration_hours=model_metrics["duration_hours"],
            time_step_minutes=model_metrics["time_step_minutes"],
            num_agents=model_metrics["num_agents"],
            num_places=model_metrics["num_places"],
            total_runtime_seconds=total_runtime,
            total_steps=model_metrics["total_steps"],
            average_step_time=total_runtime / max(1, model_metrics["total_steps"]),
            memory_usage_mb=memory_usage,
            cpu_utilization_percent=psutil.cpu_percent(interval=1),
            parallel_enabled=model_metrics["parallel_enabled"],
            max_workers=model_metrics["max_workers"],
            weather_processing_time=model_metrics["weather_time"],
            agent_processing_time=model_metrics["agent_time"],
            db_query_time=model_metrics["db_time"],
            parallel_efficiency=model_metrics["parallel_efficiency"],
            estimated_speedup=model_metrics["estimated_speedup"],
            baseline_runtime=self.baseline_runtime,
            actual_speedup=actual_speedup,
            time_saved_seconds=time_saved,
            benchmark_timestamp=datetime.now().isoformat(),
        )

        # Save results
        self._save_results(result)

        # Log summary
        self._log_benchmark_summary(result)

        return result

    def compare_models(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """
        Compare performance results between different model configurations.

        Args:
            results: List of benchmark results to compare

        Returns:
            Dictionary with comparison analysis
        """
        if len(results) < 2:
            logger.warning("Need at least 2 results for comparison")
            return {}

        comparison = {
            "models_compared": len(results),
            "baseline_model": None,
            "fastest_model": None,
            "performance_ranking": [],
            "speedup_analysis": {},
            "resource_utilization": {},
        }

        # Find baseline and fastest models
        baseline_result = min(results, key=lambda x: x.actual_speedup or 1.0)
        fastest_result = max(results, key=lambda x: x.actual_speedup or 1.0)

        comparison["baseline_model"] = baseline_result.model_name
        comparison["fastest_model"] = fastest_result.model_name

        # Performance ranking
        sorted_results = sorted(results, key=lambda x: x.total_runtime_seconds)
        comparison["performance_ranking"] = [
            {
                "rank": i + 1,
                "model": result.model_name,
                "runtime": result.total_runtime_seconds,
                "speedup": result.actual_speedup or 1.0,
            }
            for i, result in enumerate(sorted_results)
        ]

        # Detailed analysis
        comparison["speedup_analysis"] = {
            "best_speedup": fastest_result.actual_speedup or 1.0,
            "average_speedup": np.mean([r.actual_speedup or 1.0 for r in results]),
            "speedup_range": {
                "min": min(r.actual_speedup or 1.0 for r in results),
                "max": max(r.actual_speedup or 1.0 for r in results),
            },
        }

        comparison["resource_utilization"] = {
            "memory_usage": {
                "min_mb": min(r.memory_usage_mb for r in results),
                "max_mb": max(r.memory_usage_mb for r in results),
                "avg_mb": np.mean([r.memory_usage_mb for r in results]),
            },
            "cpu_utilization": {
                "min_percent": min(r.cpu_utilization_percent for r in results),
                "max_percent": max(r.cpu_utilization_percent for r in results),
                "avg_percent": np.mean([r.cpu_utilization_percent for r in results]),
            },
        }

        return comparison

    def _capture_system_info(self) -> dict[str, Any]:
        """Capture comprehensive system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / 1024**3,
            "available_memory_gb": psutil.virtual_memory().available / 1024**3,
            "python_version": platform.python_version(),
            "mpi_size": MPI.COMM_WORLD.Get_size(),
            "mpi_rank": MPI.COMM_WORLD.Get_rank(),
        }

    def _extract_model_metrics(self) -> dict[str, Any]:
        """Extract performance metrics from the model instance."""
        model = self.model_instance

        # Default values
        metrics = {
            "model_name": model.__class__.__name__,
            "duration_hours": getattr(model.params, "duration.hours", 24),
            "time_step_minutes": getattr(model.params, "time.step.minutes", 60),
            "num_agents": len(list(model.context.agents())) if hasattr(model, "context") else 0,
            "num_places": 0,
            "total_steps": getattr(model.params, "ticks", 0),
            "parallel_enabled": False,
            "max_workers": 1,
            "weather_time": 0.0,
            "agent_time": 0.0,
            "db_time": 0.0,
            "parallel_efficiency": 1.0,
            "estimated_speedup": 1.0,
        }

        # Extract places count
        if hasattr(model, "places_proj"):
            metrics["num_places"] = len(model.places_proj.get_local_places())

        # Extract parallel processing metrics
        if hasattr(model, "params"):
            metrics["parallel_enabled"] = model.params.get("parallel.heat.enabled", False)
            metrics["max_workers"] = model.params.get("parallel.heat.max_workers", 1) or psutil.cpu_count()

        # Get environment performance stats
        if hasattr(model, "get_environment"):
            env = model.get_environment()
            if hasattr(env, "parallel_processor"):
                stats = env.parallel_processor.get_performance_stats()
                metrics.update(
                    {
                        "weather_time": stats.get("weather_processing_time", 0.0),
                        "agent_time": stats.get("agent_processing_time", 0.0),
                        "db_time": stats.get("db_query_time", 0.0),
                        "parallel_efficiency": stats.get("parallel_efficiency", 1.0),
                        "estimated_speedup": stats.get("estimated_speedup", 1.0),
                    }
                )

        return metrics

    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{result.model_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert to dictionary and save
        result_dict = asdict(result)

        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {filepath}")

    def _log_benchmark_summary(self, result: BenchmarkResult):
        """Log comprehensive benchmark summary."""
        logger.info(
            f"\n{'='*70}\n"
            f"PERFORMANCE BENCHMARK SUMMARY\n"
            f"{'='*70}\n"
            f"Model: {result.model_name}\n"
            f"Runtime: {result.total_runtime_seconds:.1f} seconds ({result.total_runtime_seconds/60:.1f} minutes)\n"
            f"Steps: {result.total_steps} (avg: {result.average_step_time:.3f}s per step)\n"
            f"Agents: {result.num_agents}, Places: {result.num_places}\n"
            f"\nSystem Resources:\n"
            f"  Memory Usage: {result.memory_usage_mb:.1f} MB\n"
            f"  CPU Utilization: {result.cpu_utilization_percent:.1f}%\n"
            f"  Workers: {result.max_workers}\n"
            f"\nParallelization Performance:\n"
            f"  Parallel Enabled: {result.parallel_enabled}\n"
            f"  Weather Processing: {result.weather_processing_time:.2f}s\n"
            f"  Agent Processing: {result.agent_processing_time:.2f}s\n"
            f"  Database Queries: {result.db_query_time:.2f}s\n"
            f"  Parallel Efficiency: {result.parallel_efficiency:.2f}x\n"
            f"\nPerformance vs Baseline (1457s):\n"
            f"  Actual Speedup: {result.actual_speedup:.2f}x\n"
            f"  Time Saved: {result.time_saved_seconds:.1f} seconds\n"
            f"  Improvement: {((result.actual_speedup - 1) * 100):.1f}%\n"
            f"{'='*70}\n"
        )

    def generate_performance_report(self, results_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance analysis report.

        Args:
            results_dir: Directory containing benchmark JSON files

        Returns:
            Path to generated HTML report
        """
        if results_dir is None:
            results_dir = self.output_dir

        # Load all benchmark results
        results = self._load_all_results(Path(results_dir))

        if not results:
            logger.warning("No benchmark results found for report generation")
            return ""

        # Generate HTML report
        html_content = self._generate_html_report(results)

        # Save report
        report_path = Path(results_dir) / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Performance report generated: {report_path}")
        return str(report_path)

    def _load_all_results(self, results_dir: Path) -> list[BenchmarkResult]:
        """Load all benchmark results from directory."""
        results = []

        for json_file in results_dir.glob("benchmark_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Convert back to BenchmarkResult
                result = BenchmarkResult(**data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return results

    def _generate_html_report(self, results: list[BenchmarkResult]) -> str:
        """Generate HTML performance report."""
        # This would generate a comprehensive HTML report with charts
        # For brevity, returning a simple HTML structure

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Heat Risk Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Heat Risk Model Performance Analysis</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Benchmark Results Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Runtime (s)</th>
                    <th>Speedup</th>
                    <th>Agents</th>
                    <th>Places</th>
                    <th>Parallel</th>
                </tr>
        """

        for result in sorted(results, key=lambda x: x.total_runtime_seconds):
            html += f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td>{result.total_runtime_seconds:.1f}</td>
                    <td>{result.actual_speedup:.2f}x</td>
                    <td>{result.num_agents}</td>
                    <td>{result.num_places}</td>
                    <td>{'Yes' if result.parallel_enabled else 'No'}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Performance Insights</h2>
            <div class="metric">
                <strong>Best Performance:</strong>
        """

        if results:
            best_result = min(results, key=lambda x: x.total_runtime_seconds)
            html += f"{best_result.model_name} ({best_result.total_runtime_seconds:.1f}s, {best_result.actual_speedup:.2f}x speedup)"

        html += """
            </div>
        </body>
        </html>
        """

        return html


# Convenience function for quick benchmarking
def quick_benchmark(model_instance, description: str = "") -> BenchmarkResult:
    """
    Quick benchmark function for immediate use.

    Args:
        model_instance: Model to benchmark
        description: Description of the benchmark

    Returns:
        BenchmarkResult with performance metrics
    """
    benchmark = PerformanceBenchmark()
    benchmark.start_benchmark(model_instance, description)

    # Run the model
    model_instance.start()

    return benchmark.end_benchmark()
