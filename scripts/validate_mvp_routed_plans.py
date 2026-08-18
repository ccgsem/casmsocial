"""Validate routed plan metadata for the MVP road fixture."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mpi4py import MPI
from repast4py.parameters import init_params

from casmsocial.activities import Leg
from casmsocial.casmpop import CasmPop
from casmsocial.model import Model
from casmsocial.person import Person, ScheduleBehaviorEngine

DEFAULT_CONFIG_PATH = Path("config/mvp.yaml")
DEFAULT_DUCKLAKE_PATH = Path("examples/mvp/mvp.ducklake")
DEFAULT_REPORT_PATH = Path("data/output/mvp_routed_plan_validation.json")
DEFAULT_EXPECTED_AGENTS = 2
DEFAULT_EXPECTED_LEGS = 4
DEFAULT_MODE = "drive"

EXPECTED_ROUTES: dict[tuple[int, int], tuple[int, int, float, int]] = {
    (100, 300): (1, 3, 5000.0, 12),
    (300, 100): (3, 1, 5000.0, 12),
    (200, 300): (2, 3, 4200.0, 10),
    (300, 200): (3, 2, 4200.0, 10),
}


def _route_report_key(route_key: tuple[int, int]) -> str:
    return f"{route_key[0]}->{route_key[1]}"


class MvpRoutedPlanValidationError(ValueError):
    """Raised when MVP routed plans do not match the expected fixture contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MvpRoutedPlanValidationError(message)


def _person_agents(model: Any) -> list[Any]:
    try:
        return list(model.context.agents(agent_type=Person.TYPE))
    except KeyError:
        return []


def _routed_mvp_overrides(
    *,
    roads_nodes_file: str,
    roads_edges_file: str,
    roads_place_snap_file: str,
) -> dict[str, Any]:
    return {
        "roads.enabled": True,
        "roads.nodes.file": roads_nodes_file,
        "roads.edges.file": roads_edges_file,
        "roads.place_snap.file": roads_place_snap_file,
        "behavior.engine": "schedule",
        "behavior.llm.enabled": False,
        "communication.enabled": False,
        "observers.agent_log.enabled": False,
        "observers.behavior_log.enabled": False,
    }


def validate_routed_mvp_plans(
    model: Any,
    *,
    expected_agents: int = DEFAULT_EXPECTED_AGENTS,
    expected_legs: int = DEFAULT_EXPECTED_LEGS,
    expected_routes: dict[tuple[int, int], tuple[int, int, float, int]] = EXPECTED_ROUTES,
    check_distances: bool = True,
    distance_tolerance_m: float = 0.0,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Validate routed leg metadata on locally materialized MVP people."""
    people = sorted(_person_agents(model), key=lambda person: person.id)
    _require(len(people) == expected_agents, f"Expected {expected_agents} routed MVP agents, found {len(people)}")

    route_counts: Counter[tuple[int, int]] = Counter()
    observed_routes: dict[tuple[int, int], dict[str, Any]] = {}
    leg_count = 0
    for person in people:
        _require(person.plans, f"Person {person.id} has no plans")
        plan = person.plans[0]
        legs = [element for element in plan if isinstance(element, Leg)]
        _require(legs, f"Person {person.id} has no routed legs")

        for leg in legs:
            leg_count += 1
            _require(leg.mode == mode, f"Expected routed leg mode {mode!r}, found {leg.mode!r}")
            _require(leg.origin_place_id is not None, "Routed leg is missing origin_place_id")
            _require(leg.destination_place_id is not None, "Routed leg is missing destination_place_id")
            route_key = (int(leg.origin_place_id), int(leg.destination_place_id))
            _require(route_key in expected_routes, f"Unexpected routed MVP leg: {route_key}")

            expected_origin_node, expected_destination_node, expected_distance, expected_time = expected_routes[
                route_key
            ]
            _require(
                leg.origin_node_id == expected_origin_node,
                f"Route {route_key} expected origin node {expected_origin_node}, found {leg.origin_node_id}",
            )
            _require(
                leg.destination_node_id == expected_destination_node,
                f"Route {route_key} expected destination node {expected_destination_node}, "
                f"found {leg.destination_node_id}",
            )
            if check_distances:
                _require(
                    leg.distance_m is not None
                    and math.isclose(leg.distance_m, expected_distance, rel_tol=0.0, abs_tol=distance_tolerance_m),
                    f"Route {route_key} expected distance {expected_distance}, found {leg.distance_m}",
                )
            _require(
                leg.travel_time_min == expected_time,
                f"Route {route_key} expected travel time {expected_time}, found {leg.travel_time_min}",
            )
            route_counts[route_key] += 1
            observed_routes[route_key] = {
                "origin_place_id": route_key[0],
                "destination_place_id": route_key[1],
                "origin_node_id": leg.origin_node_id,
                "destination_node_id": leg.destination_node_id,
                "distance_m": leg.distance_m,
                "travel_time_min": leg.travel_time_min,
            }

    _require(leg_count == expected_legs, f"Expected {expected_legs} routed MVP legs, found {leg_count}")
    for route_key in expected_routes:
        _require(
            route_counts[route_key] == 1, f"Expected one routed leg for {route_key}, found {route_counts[route_key]}"
        )

    routes = {
        _route_report_key(route_key): {
            **observed_routes[route_key],
            "count": route_counts[route_key],
        }
        for route_key in expected_routes
    }

    return {
        "agents": len(people),
        "legs": leg_count,
        "routes": routes,
    }


def build_routed_mvp_plan_report(
    summary: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a durable JSON report for routed MVP plan validation."""
    return {
        "version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "ducklake_path": str(ducklake_path),
        "validation": summary,
    }


def write_routed_mvp_plan_report(
    report_path: Path,
    summary: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
) -> dict[str, Any]:
    """Write routed MVP plan validation as a machine-readable artifact."""
    report = build_routed_mvp_plan_report(summary, config_path=config_path, ducklake_path=ducklake_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_routed_mvp_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    ducklake_path: Path = DEFAULT_DUCKLAKE_PATH,
    *,
    roads_nodes_file: str = "casmsocial_mvp.road_nodes",
    roads_edges_file: str = "casmsocial_mvp.road_edges",
    roads_place_snap_file: str = "casmsocial_mvp.place_road_snap",
    check_distances: bool = True,
    distance_tolerance_m: float = 0.0,
) -> dict[str, Any]:
    """Build the MVP model with road routing enabled and validate its plans."""
    previous_model = Model.get_model()
    previous_behavior_engine = Person.getBehaviorEngine()
    previous_activity_names = CasmPop.get_activity_names()
    previous_planned_activity_names = CasmPop.get_planned_activity_names()
    previous_activities_data_type = CasmPop._CasmPop__activities_data_type

    os.environ["CASMSOCIAL_DATA_PATH"] = str(ducklake_path.parent)
    os.environ["CASMSOCIAL_DUCKLAKE_PATH"] = str(ducklake_path)
    params = init_params(
        str(config_path),
        json.dumps(
            _routed_mvp_overrides(
                roads_nodes_file=roads_nodes_file,
                roads_edges_file=roads_edges_file,
                roads_place_snap_file=roads_place_snap_file,
            )
        ),
    )

    model = None
    try:
        model = CasmPop(MPI.COMM_SELF, params)
        model.build_context()
        return validate_routed_mvp_plans(
            model,
            check_distances=check_distances,
            distance_tolerance_m=distance_tolerance_m,
        )
    finally:
        if model is not None:
            model.conn.close()
        CasmPop.register_activity_names(previous_activity_names)
        CasmPop.register_planned_activity_names(previous_planned_activity_names)
        CasmPop._CasmPop__activities_data_type = previous_activities_data_type
        Person.registerBehaviorEngine(previous_behavior_engine or ScheduleBehaviorEngine)
        Model.theModel = previous_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="MVP config path.")
    parser.add_argument(
        "--ducklake-path",
        type=Path,
        default=DEFAULT_DUCKLAKE_PATH,
        help="Generated MVP DuckLake path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the routed plan validation report.",
    )
    parser.add_argument(
        "--roads-nodes-file",
        default="casmsocial_mvp.road_nodes",
        help="Road nodes table or parquet file used for validation.",
    )
    parser.add_argument(
        "--roads-edges-file",
        default="casmsocial_mvp.road_edges",
        help="Road edges table or parquet file used for validation.",
    )
    parser.add_argument(
        "--roads-place-snap-file",
        default="casmsocial_mvp.place_road_snap",
        help="Place-to-road-node snap table or parquet file used for validation.",
    )
    parser.add_argument(
        "--skip-distance-check",
        action="store_true",
        help="Validate route nodes and travel times without exact fixture distance checks.",
    )
    parser.add_argument(
        "--distance-tolerance-m",
        type=float,
        default=0.0,
        help="Absolute tolerance for route distance checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_routed_mvp_config(
            args.config,
            args.ducklake_path,
            roads_nodes_file=args.roads_nodes_file,
            roads_edges_file=args.roads_edges_file,
            roads_place_snap_file=args.roads_place_snap_file,
            check_distances=not args.skip_distance_check,
            distance_tolerance_m=args.distance_tolerance_m,
        )
    except MvpRoutedPlanValidationError as exc:
        print(f"MVP routed plan validation failed: {exc}", file=sys.stderr)
        return 1

    write_routed_mvp_plan_report(
        args.output,
        summary,
        config_path=args.config,
        ducklake_path=args.ducklake_path,
    )
    print(
        "MVP routed plans valid: "
        f"{args.ducklake_path} "
        f"(agents={summary['agents']}, legs={summary['legs']}, routes={len(summary['routes'])}, "
        f"report={args.output})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
