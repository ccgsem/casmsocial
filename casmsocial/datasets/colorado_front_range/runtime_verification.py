"""Execute and validate privacy-safe Colorado runtime smoke runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile
from casmsocial.datasets.colorado_front_range.sources import sha256_file

SUM_COLUMNS = [
    "active_person_count",
    "active_place_count",
    "co_located_person_count",
    "places_with_1_person",
    "places_with_2_to_4_people",
    "places_with_5_to_9_people",
    "places_with_10_or_more_people",
    "in_person_interaction_count",
    "remote_message_count",
]


def _runtime_manifest(runtime_dir: Path, profile: ColoradoDatasetProfile) -> dict[str, object]:
    path = runtime_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "passed"
        or manifest.get("profile_id") != profile.profile_id
        or manifest.get("profile_version") != profile.profile_version
        or not (runtime_dir / "casmsocial.yaml").is_file()
        or not (runtime_dir / "ducklake").is_dir()
    ):
        raise ValueError("Runtime product does not have a matching accepted manifest")
    return manifest


def aggregate_runtime_output(output_dir: Path) -> dict[str, object]:
    """Aggregate rank-partitioned observer datasets without exposing identifiers."""
    occupancy_path = output_dir / "schedule_occupancy.parquet"
    interactions_path = output_dir / "social_interactions.parquet"
    if not occupancy_path.exists():
        raise FileNotFoundError(occupancy_path)
    occupancy = pl.read_parquet(occupancy_path, hive_partitioning=True)
    required = {"tick", "rank", "max_place_occupancy", *SUM_COLUMNS}
    if missing := required - set(occupancy.columns):
        raise ValueError(f"Schedule occupancy output is missing columns: {', '.join(sorted(missing))}")
    expressions = [pl.col(column).sum().alias(column) for column in SUM_COLUMNS]
    expressions.append(pl.col("max_place_occupancy").max())
    ticks = occupancy.group_by("tick").agg(*expressions).sort("tick")
    interaction_rows = 0
    interaction_total = 0
    if interactions_path.exists():
        interactions = pl.read_parquet(interactions_path, hive_partitioning=True)
        if {"channel", "network_kind", "event_count"} - set(interactions.columns):
            raise ValueError("Social-interaction output has an unsupported schema")
        interaction_rows = interactions.height
        interaction_total = int(interactions["event_count"].sum())
    return {
        "ticks": ticks.height,
        "tick_min": int(ticks["tick"].min()),
        "tick_max": int(ticks["tick"].max()),
        "all_ticks_have_active_people": bool((ticks["active_person_count"] > 0).all()),
        "peak_place_occupancy": int(ticks["max_place_occupancy"].max()),
        "in_person_interaction_total": int(ticks["in_person_interaction_count"].sum()),
        "remote_message_total": int(ticks["remote_message_count"].sum()),
        "social_interaction_rows": interaction_rows,
        "social_interaction_total": interaction_total,
        "tick_metrics": ticks.to_dicts(),
    }


def validate_runtime_runs(runs: dict[int, dict[str, object]], profile: ColoradoDatasetProfile) -> dict[str, object]:
    """Apply profile smoke and cross-rank equivalence gates."""
    checks: dict[str, bool] = {}
    single = runs.get(profile.runtime.default_ranks or 1)
    if profile.validation.require_single_rank_smoke:
        if single is None:
            raise ValueError("Profile requires a single-rank smoke run")
        expected_ticks = (profile.runtime.default_duration_hours or 24) + 1
        checks.update({
            "single_rank_tick_count": single["ticks"] == expected_ticks,
            "single_rank_active_people": bool(single["all_ticks_have_active_people"]),
            "single_rank_shared_occupancy": single["peak_place_occupancy"] > 1,
            "single_rank_in_person_interactions": single["in_person_interaction_total"] > 0,
            "single_rank_remote_messages": single["remote_message_total"] > 0,
        })
    if profile.validation.require_two_rank_equivalence:
        ranks = profile.runtime.verification_ranks or 2
        parallel = runs.get(ranks)
        if single is None or parallel is None:
            raise ValueError("Profile requires single- and two-rank results")
        checks.update({
            "rank_equivalent_ticks": single["ticks"] == parallel["ticks"],
            "rank_equivalent_peak_occupancy": single["peak_place_occupancy"] == parallel["peak_place_occupancy"],
            "rank_equivalent_in_person_total": single["in_person_interaction_total"]
            == parallel["in_person_interaction_total"],
            "rank_equivalent_remote_total": single["remote_message_total"] == parallel["remote_message_total"],
        })
    status = "passed" if all(checks.values()) else "failed"
    result = {"status": status, "checks": checks}
    if status != "passed":
        raise ValueError(f"Runtime verification failed: {checks}")
    return result


def _run(runtime_dir: Path, output_dir: Path, ranks: int, profile: ColoradoDatasetProfile) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    override = {
        "observers.output_dir": str(output_dir),
        "observers.agent_log.enabled": False,
        "observers.behavior_log.enabled": False,
        "observers.delta_agent_state.enabled": False,
        "partition.table": "" if ranks == 1 else "partitions.colorado_front_range_place_partitions",
        "partition.require_full_coverage": ranks > 1,
    }
    command = [
        "mpirun",
        "--oversubscribe",
        "-n",
        str(ranks),
        sys.executable,
        "-m",
        "casmsocial",
        str(runtime_dir / "casmsocial.yaml"),
        json.dumps(override, separators=(",", ":")),
    ]
    environment = {
        **os.environ,
        "CASMSOCIAL_DATA_PATH": str(runtime_dir),
        "CASMSOCIAL_DUCKLAKE_PATH": str(runtime_dir / "ducklake"),
    }
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{ranks}-rank CASMSocial verification failed with exit code {completed.returncode}")
    return aggregate_runtime_output(output_dir)


def verify_profile_runtime(
    runtime_dir: Path,
    profile: ColoradoDatasetProfile,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run required MPI smoke configurations and publish aggregate acceptance."""
    runtime_dir, output_dir = runtime_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    runtime_manifest = _runtime_manifest(runtime_dir, profile)
    fingerprint = {
        "runtime_manifest_sha256": sha256_file(runtime_dir / "manifest.json"),
        "profile": profile.model_dump(mode="json"),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("status") == "passed" and cached.get("inputs") == fingerprint:
            return {**cached, "resumed": True}
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Verification output is not resumable; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    ranks = {profile.runtime.default_ranks or 1}
    if profile.validation.require_two_rank_equivalence:
        ranks.add(profile.runtime.verification_ranks or 2)
    runs = {
        rank_count: _run(runtime_dir, output_dir / f"rank-{rank_count}", rank_count, profile)
        for rank_count in sorted(ranks)
    }
    acceptance = validate_runtime_runs(runs, profile)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "resumed": False,
        "profile_id": profile.profile_id,
        "inputs": fingerprint,
        "runs": {str(key): value for key, value in runs.items()},
        "acceptance": acceptance,
        "privacy": "Aggregate observer outputs only; no person, tie-endpoint, place, or coordinate fields.",
        "runtime_product_status": runtime_manifest["status"],
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
