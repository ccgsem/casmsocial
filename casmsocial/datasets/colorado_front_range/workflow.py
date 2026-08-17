"""Orchestrate the complete public Colorado Front Range build workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from casmsocial.datasets.colorado_front_range.destination_supply import build_profile_destinations
from casmsocial.datasets.colorado_front_range.osf_ducklake import build_ducklake, validate_state_partitions
from casmsocial.datasets.colorado_front_range.osf_tables import build_state_tables
from casmsocial.datasets.colorado_front_range.profile_population import build_profile_population
from casmsocial.datasets.colorado_front_range.profile_runtime import build_profile_runtime
from casmsocial.datasets.colorado_front_range.profile_schedules import build_profile_schedules
from casmsocial.datasets.colorado_front_range.profiles import ColoradoDatasetProfile
from casmsocial.datasets.colorado_front_range.runtime_verification import verify_profile_runtime
from casmsocial.datasets.colorado_front_range.sources import (
    SourceArtifact,
    artifact_path,
    get_source_artifact,
    inspect_artifact,
    sha256_file,
)

REQUIRED_ARTIFACT_IDS = (
    "osf-colorado-population",
    "osf-colorado-education-sites",
    "census-2023-counties",
    "bls-atus-2024-respondents",
    "bls-atus-2024-activities",
    "bls-atus-2024-roster",
    "osm-geofabrik-colorado",
)


def profile_build_plan(
    data_dir: Path,
    profile_name: str,
    profile: ColoradoDatasetProfile,
) -> dict[str, object]:
    """Describe source readiness and canonical outputs without changing local data."""
    data_dir = data_dir.expanduser().resolve()
    artifacts = [get_source_artifact(artifact_id) for artifact_id in REQUIRED_ARTIFACT_IDS]
    source_status = [inspect_artifact(artifact, data_dir) for artifact in artifacts]
    root = data_dir / "local"
    runtime_verification_required = (
        profile.validation.require_single_rank_smoke or profile.validation.require_two_rank_equivalence
    )
    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "release_status": profile.release_status,
        "ready": all(item["status"] == "verified" for item in source_status),
        "sources": source_status,
        "stages": [
            {"name": "osf_tables", "output": str(root / "osf-synthetic-population" / "source_state=CO")},
            {
                "name": "osf_ducklake",
                "output": str(root / "osf-synthetic-ducklake" / "metadata.ducklake"),
            },
            {
                "name": "profile_population",
                "output": str(root / "colorado-front-range-populations" / profile_name),
            },
            {"name": "schedules", "output": str(root / "colorado-front-range-schedules" / profile_name)},
            {"name": "destinations", "output": str(root / "colorado-front-range-destinations" / profile_name)},
            {"name": "runtime", "output": str(root / "colorado-front-range-runtime" / profile_name)},
            {
                "name": "runtime_verification",
                "output": str(root / "colorado-front-range-verification" / profile_name),
                "required": runtime_verification_required,
            },
        ],
    }


def _verified_artifacts(data_dir: Path) -> dict[str, SourceArtifact]:
    artifacts = {artifact_id: get_source_artifact(artifact_id) for artifact_id in REQUIRED_ARTIFACT_IDS}
    unavailable = []
    for artifact in artifacts.values():
        status = inspect_artifact(artifact, data_dir)
        if status["status"] != "verified":
            action = "download manually, record, and verify" if artifact.access == "manual" else "fetch and verify"
            unavailable.append(f"{artifact.artifact_id} ({status['status']}; {action})")
    if unavailable:
        raise ValueError("Required sources are not verified:\n- " + "\n- ".join(unavailable))
    return artifacts


def _cached_state_tables(
    output_dir: Path,
    population_archive: Path,
    education_archive: Path,
) -> dict[str, object] | None:
    try:
        partitions = validate_state_partitions(output_dir)
    except (FileNotFoundError, ValueError):
        return None
    colorado = next((partition for partition in partitions if partition["state"] == "CO"), None)
    if colorado is None:
        return None
    manifest_path = Path(str(colorado["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        return None
    expected = {
        "population_archive": sha256_file(population_archive),
        "education_archive": sha256_file(education_archive),
    }
    if any(
        not isinstance(inputs.get(name), dict) or inputs[name].get("sha256") != digest
        for name, digest in expected.items()
    ):
        return None
    return {**manifest, "resumed": True}


def _state_tables(
    output_dir: Path,
    population_archive: Path,
    education_archive: Path,
    *,
    batch_size: int,
    overwrite: bool,
) -> dict[str, object]:
    if cached := _cached_state_tables(output_dir, population_archive, education_archive):
        return cached
    state_dir = output_dir / "source_state=CO"
    if state_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Colorado state tables are not resumable; use --overwrite: {state_dir}")
        shutil.rmtree(state_dir)
    return build_state_tables(
        population_archive,
        "CO",
        output_dir,
        education_archive=education_archive,
        batch_size=batch_size,
    )


def _stage_receipt(name: str, manifest_path: Path, manifest: dict[str, object]) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return {
        "name": name,
        "status": manifest.get("status", "passed"),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_colorado_profile(
    data_dir: Path,
    profile_name: str,
    profile: ColoradoDatasetProfile,
    *,
    batch_size: int = 250_000,
    source_year: int = 2024,
    minimum_routable_minutes: int = 10,
    minimum_places_per_activity_kind: int = 20,
    capacity_multiplier: float | None = None,
    full_population_capacity_multiplier: float | None = None,
    allow_planned: bool = False,
    run_runtime_verification: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build every accepted product for one profile from already staged sources."""
    if profile.release_status == "planned" and not allow_planned:
        raise ValueError(f"Profile {profile.profile_id} is planned; pass --allow-planned for an exploratory build")
    data_dir = data_dir.expanduser().resolve()
    artifacts = _verified_artifacts(data_dir)
    local = data_dir / "local"
    osf_tables = local / "osf-synthetic-population"
    lake_root = local / "osf-synthetic-ducklake"
    lake_catalog = lake_root / "metadata.ducklake"
    lake_data = lake_root / "files"
    population_dir = local / "colorado-front-range-populations" / profile_name
    schedule_dir = local / "colorado-front-range-schedules" / profile_name
    destination_dir = local / "colorado-front-range-destinations" / profile_name
    runtime_dir = local / "colorado-front-range-runtime" / profile_name
    verification_dir = local / "colorado-front-range-verification" / profile_name

    population_archive = artifact_path(data_dir, artifacts["osf-colorado-population"])
    education_archive = artifact_path(data_dir, artifacts["osf-colorado-education-sites"])
    counties = artifact_path(data_dir, artifacts["census-2023-counties"])
    respondents = artifact_path(data_dir, artifacts["bls-atus-2024-respondents"])
    activities = artifact_path(data_dir, artifacts["bls-atus-2024-activities"])
    roster = artifact_path(data_dir, artifacts["bls-atus-2024-roster"])
    osm = artifact_path(data_dir, artifacts["osm-geofabrik-colorado"])

    state_manifest = _state_tables(
        osf_tables,
        population_archive,
        education_archive,
        batch_size=batch_size,
        overwrite=overwrite,
    )
    lake_manifest = build_ducklake(osf_tables, lake_catalog, lake_data, overwrite=overwrite)
    population_manifest = build_profile_population(
        lake_catalog,
        lake_data,
        counties,
        profile,
        population_dir,
        batch_size=batch_size,
        allow_planned=allow_planned,
        overwrite=overwrite,
    )
    schedule_manifest = build_profile_schedules(
        population_dir,
        respondents,
        activities,
        roster,
        profile,
        schedule_dir,
        source_year=source_year,
        minimum_routable_minutes=minimum_routable_minutes,
        allow_planned=allow_planned,
        overwrite=overwrite,
    )
    destination_manifest = build_profile_destinations(
        population_dir,
        schedule_dir,
        osm,
        counties,
        profile,
        destination_dir,
        minimum_places_per_activity_kind=minimum_places_per_activity_kind,
        capacity_multiplier=capacity_multiplier,
        full_population_capacity_multiplier=full_population_capacity_multiplier,
        allow_planned=allow_planned,
        overwrite=overwrite,
    )
    runtime_manifest = build_profile_runtime(
        population_dir,
        destination_dir,
        profile,
        runtime_dir,
        overwrite=overwrite,
    )

    stages = [
        _stage_receipt("osf_tables", osf_tables / "source_state=CO" / "manifest.json", state_manifest),
        _stage_receipt("osf_ducklake", lake_catalog.with_suffix(".ducklake.manifest.json"), lake_manifest),
        _stage_receipt("profile_population", population_dir / "manifest.json", population_manifest),
        _stage_receipt("schedules", schedule_dir / "manifest.json", schedule_manifest),
        _stage_receipt("destinations", destination_dir / "manifest.json", destination_manifest),
        _stage_receipt("runtime", runtime_dir / "manifest.json", runtime_manifest),
    ]
    verification_required = (
        profile.validation.require_single_rank_smoke or profile.validation.require_two_rank_equivalence
    )
    if run_runtime_verification and verification_required:
        verification_manifest = verify_profile_runtime(
            runtime_dir,
            profile,
            verification_dir,
            overwrite=overwrite,
        )
        stages.append(_stage_receipt("runtime_verification", verification_dir / "manifest.json", verification_manifest))
    else:
        stages.append({
            "name": "runtime_verification",
            "status": "skipped",
            "reason": "disabled" if verification_required else "not_required_by_profile",
        })

    receipt_path = local / "colorado-front-range-builds" / profile_name / "manifest.json"
    status = "built_unverified" if verification_required and not run_runtime_verification else "passed"
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "release_status": profile.release_status,
        "stages": stages,
        "runtime": str(runtime_dir),
        "runtime_verification": str(verification_dir) if run_runtime_verification and verification_required else None,
        "governance": "Generated identifier-bearing products remain local and must not be published.",
    }
    existing = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    resumed = existing == receipt
    _write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "resumed": resumed}
