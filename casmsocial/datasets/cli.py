"""Command-line access to CASMSocial public dataset contracts."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from casmsocial.datasets.colorado_front_range import (
    artifact_path,
    build_colorado_profile,
    build_ducklake,
    build_profile_destinations,
    build_profile_population,
    build_profile_runtime,
    build_profile_schedules,
    build_state_tables,
    download_artifact,
    get_source_artifact,
    inspect_artifact,
    list_profiles,
    load_migrated_code_provenance,
    load_osm_attribution,
    load_profile,
    load_source_inventory,
    load_source_licenses,
    profile_build_plan,
    record_artifact,
    verify_profile_runtime,
)

app = typer.Typer(help="Inspect and build versioned CASMSocial datasets.", no_args_is_help=True)
colorado = typer.Typer(help="Colorado Front Range dataset contracts.", no_args_is_help=True)
app.add_typer(colorado, name="colorado")


def _render(value: object, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(value, indent=2, sort_keys=True)
    if output_format == "yaml":
        return yaml.safe_dump(value, sort_keys=False).rstrip()
    raise typer.BadParameter("format must be yaml or json", param_hint="--format")


@colorado.command("profiles")
def profiles(output_format: str = typer.Option("yaml", "--format")) -> None:
    """List bundled Colorado profiles and release states."""
    values = [
        {
            "name": name,
            "profile_id": profile.profile_id,
            "release_status": profile.release_status,
            "boundary_id": profile.geography.boundary_id,
            "person_limit": profile.population.person_limit,
        }
        for name in list_profiles()
        for profile in [load_profile(name)]
    ]
    typer.echo(_render(values, output_format))


@colorado.command("show-profile")
def show_profile(name: str, output_format: str = typer.Option("yaml", "--format")) -> None:
    """Print one validated Colorado profile."""
    try:
        profile = load_profile(name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="NAME") from error
    typer.echo(_render(profile.model_dump(mode="json"), output_format))


@colorado.command("licenses")
def licenses(output_format: str = typer.Option("yaml", "--format")) -> None:
    """Print the source-license audit and open release gates."""
    typer.echo(_render(load_source_licenses(), output_format))


@colorado.command("osm-attribution")
def osm_attribution() -> None:
    """Print the required OSM attribution and local-only distribution notice."""
    typer.echo(load_osm_attribution())


@colorado.command("provenance")
def provenance(output_format: str = typer.Option("yaml", "--format")) -> None:
    """Print the private-to-public migrated-code provenance record."""
    typer.echo(_render(load_migrated_code_provenance(), output_format))


@colorado.command("sources")
def sources(output_format: str = typer.Option("yaml", "--format")) -> None:
    """List required source artifacts and acquisition policies."""
    inventory = load_source_inventory()
    values = [
        {
            "artifact_id": artifact.artifact_id,
            "source_id": artifact.source_id,
            "access": artifact.access,
            "verification": artifact.verification,
            "destination": artifact.destination,
            "source": artifact.url or artifact.source_page,
        }
        for artifact in inventory.artifacts
    ]
    typer.echo(_render(values, output_format))


def _artifact_or_bad_parameter(artifact_id: str):
    try:
        return get_source_artifact(artifact_id)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="ARTIFACT_ID") from error


@colorado.command("fetch")
def fetch(
    artifact_id: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Download one automatic source and write a SHA-256 provenance sidecar."""
    artifact = _artifact_or_bad_parameter(artifact_id)
    try:
        status = download_artifact(artifact, data_dir, overwrite=overwrite)
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="ARTIFACT_ID") from error
    typer.echo(_render(status, output_format))


@colorado.command("record")
def record(
    artifact_id: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Record SHA-256 provenance for an already staged source file."""
    artifact = _artifact_or_bad_parameter(artifact_id)
    try:
        provenance = record_artifact(artifact, data_dir)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="ARTIFACT_ID") from error
    typer.echo(_render(provenance, output_format))


@colorado.command("verify")
def verify(
    artifact_id: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Verify one staged source against pinned or recorded SHA-256 provenance."""
    artifact = _artifact_or_bad_parameter(artifact_id)
    status = inspect_artifact(artifact, data_dir)
    typer.echo(_render(status, output_format))
    if status["status"] != "verified":
        raise typer.Exit(code=1)


@colorado.command("build-osf")
def build_osf(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    batch_size: int = typer.Option(250_000, "--batch-size", min=1),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Normalize verified Colorado OSF archives into four canonical Parquet tables."""
    population = get_source_artifact("osf-colorado-population")
    education = get_source_artifact("osf-colorado-education-sites")
    for artifact in (population, education):
        status = inspect_artifact(artifact, data_dir)
        if status["status"] != "verified":
            raise typer.BadParameter(
                f"{artifact.artifact_id} is {status['status']}; fetch and verify it before building",
                param_hint="--data-dir",
            )
    destination = output_dir or data_dir / "local" / "osf-synthetic-population"
    try:
        manifest = build_state_tables(
            artifact_path(data_dir, population),
            "CO",
            destination,
            education_archive=artifact_path(data_dir, education),
            batch_size=batch_size,
        )
    except RuntimeError as error:
        raise typer.BadParameter(str(error), param_hint="--data-dir") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-ducklake")
def build_osf_ducklake(
    input_dir: Path = typer.Option(Path("data/local/osf-synthetic-population"), "--input-dir"),
    catalog: Path = typer.Option(Path("data/local/osf-synthetic-ducklake/metadata.ducklake"), "--catalog"),
    data_path: Path = typer.Option(Path("data/local/osf-synthetic-ducklake/files"), "--data-path"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Materialize validated state partitions as one accepted local DuckLake."""
    try:
        manifest = build_ducklake(input_dir, catalog, data_path, overwrite=overwrite)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--input-dir") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-population")
def build_population(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    catalog: Path = typer.Option(Path("data/local/osf-synthetic-ducklake/metadata.ducklake"), "--catalog"),
    data_path: Path = typer.Option(Path("data/local/osf-synthetic-ducklake/files"), "--data-path"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    batch_size: int = typer.Option(250_000, "--batch-size", min=1),
    allow_planned: bool = typer.Option(False, "--allow-planned"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Build a profile-scoped, endpoint-complete Colorado population product."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    counties = get_source_artifact("census-2023-counties")
    county_status = inspect_artifact(counties, data_dir)
    if county_status["status"] != "verified":
        raise typer.BadParameter(
            f"census-2023-counties is {county_status['status']}; fetch or record and verify it before building",
            param_hint="--data-dir",
        )
    destination = output_dir or data_dir / "local" / "colorado-front-range-populations" / profile_name
    try:
        manifest = build_profile_population(
            catalog,
            data_path,
            artifact_path(data_dir, counties),
            profile,
            destination,
            batch_size=batch_size,
            allow_planned=allow_planned,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-schedules")
def build_schedules(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    profile_dir: Path | None = typer.Option(None, "--profile-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    source_year: int = typer.Option(2024, "--source-year", min=2003),
    minimum_routable_minutes: int = typer.Option(10, "--minimum-routable-minutes", min=1),
    allow_planned: bool = typer.Option(False, "--allow-planned"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Build profile-scoped ATUS weekday and weekend pre-routing schedules."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    artifacts = {
        name: get_source_artifact(name)
        for name in (
            "bls-atus-2024-respondents",
            "bls-atus-2024-activities",
            "bls-atus-2024-roster",
        )
    }
    for artifact in artifacts.values():
        status = inspect_artifact(artifact, data_dir)
        if status["status"] != "verified":
            raise typer.BadParameter(
                f"{artifact.artifact_id} is {status['status']}; record/verify before building",
                param_hint="--data-dir",
            )
    population = profile_dir or data_dir / "local" / "colorado-front-range-populations" / profile_name
    destination = output_dir or data_dir / "local" / "colorado-front-range-schedules" / profile_name
    try:
        manifest = build_profile_schedules(
            population,
            artifact_path(data_dir, artifacts["bls-atus-2024-respondents"]),
            artifact_path(data_dir, artifacts["bls-atus-2024-activities"]),
            artifact_path(data_dir, artifacts["bls-atus-2024-roster"]),
            profile,
            destination,
            source_year=source_year,
            minimum_routable_minutes=minimum_routable_minutes,
            allow_planned=allow_planned,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-destinations")
def build_destinations(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    profile_dir: Path | None = typer.Option(None, "--profile-dir"),
    schedule_dir: Path | None = typer.Option(None, "--schedule-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    minimum_places_per_activity_kind: int = typer.Option(20, "--minimum-places-per-activity-kind", min=1),
    capacity_multiplier: float | None = typer.Option(None, "--capacity-multiplier", min=0.000001),
    full_population_capacity_multiplier: float | None = typer.Option(
        None,
        "--full-population-capacity-multiplier",
        min=0.000001,
    ),
    allow_planned: bool = typer.Option(False, "--allow-planned"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Build OSM-backed supply and event-level discretionary destinations."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    artifacts = {name: get_source_artifact(name) for name in ("osm-geofabrik-colorado", "census-2023-counties")}
    for artifact in artifacts.values():
        status = inspect_artifact(artifact, data_dir)
        if status["status"] != "verified":
            action = "fetch/verify" if artifact.access == "download" else "record/verify"
            raise typer.BadParameter(
                f"{artifact.artifact_id} is {status['status']}; {action} before building",
                param_hint="--data-dir",
            )
    population = profile_dir or data_dir / "local" / "colorado-front-range-populations" / profile_name
    schedules = schedule_dir or data_dir / "local" / "colorado-front-range-schedules" / profile_name
    destination = output_dir or data_dir / "local" / "colorado-front-range-destinations" / profile_name
    try:
        manifest = build_profile_destinations(
            population,
            schedules,
            artifact_path(data_dir, artifacts["osm-geofabrik-colorado"]),
            artifact_path(data_dir, artifacts["census-2023-counties"]),
            profile,
            destination,
            minimum_places_per_activity_kind=minimum_places_per_activity_kind,
            capacity_multiplier=capacity_multiplier,
            full_population_capacity_multiplier=full_population_capacity_multiplier,
            allow_planned=allow_planned,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-runtime")
def build_runtime(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    profile_dir: Path | None = typer.Option(None, "--profile-dir"),
    destination_dir: Path | None = typer.Option(None, "--destination-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Route, validate, export, and catalog a runnable CASMSocial profile."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    population = profile_dir or data_dir / "local" / "colorado-front-range-populations" / profile_name
    destinations = destination_dir or data_dir / "local" / "colorado-front-range-destinations" / profile_name
    destination = output_dir or data_dir / "local" / "colorado-front-range-runtime" / profile_name
    try:
        manifest = build_profile_runtime(population, destinations, profile, destination, overwrite=overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("verify-runtime")
def verify_runtime(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    runtime_dir: Path | None = typer.Option(None, "--runtime-dir"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Run required MPI smoke tests and compare privacy-safe aggregates."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    runtime = runtime_dir or data_dir / "local" / "colorado-front-range-runtime" / profile_name
    destination = output_dir or data_dir / "local" / "colorado-front-range-verification" / profile_name
    try:
        manifest = verify_profile_runtime(runtime, profile, destination, overwrite=overwrite)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(manifest, output_format))


@colorado.command("build-all")
def build_all(
    profile_name: str,
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    batch_size: int = typer.Option(250_000, "--batch-size", min=1),
    source_year: int = typer.Option(2024, "--source-year", min=2003),
    minimum_routable_minutes: int = typer.Option(10, "--minimum-routable-minutes", min=1),
    minimum_places_per_activity_kind: int = typer.Option(20, "--minimum-places-per-activity-kind", min=1),
    capacity_multiplier: float | None = typer.Option(None, "--capacity-multiplier", min=0.000001),
    full_population_capacity_multiplier: float | None = typer.Option(
        None,
        "--full-population-capacity-multiplier",
        min=0.000001,
    ),
    allow_planned: bool = typer.Option(False, "--allow-planned"),
    run_runtime_verification: bool = typer.Option(
        True,
        "--verify-runtime/--skip-runtime-verification",
    ),
    plan: bool = typer.Option(False, "--plan"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_format: str = typer.Option("yaml", "--format"),
) -> None:
    """Build every Colorado profile product from explicitly staged sources."""
    try:
        profile = load_profile(profile_name)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    if plan:
        typer.echo(_render(profile_build_plan(data_dir, profile_name, profile), output_format))
        return
    try:
        receipt = build_colorado_profile(
            data_dir,
            profile_name,
            profile,
            batch_size=batch_size,
            source_year=source_year,
            minimum_routable_minutes=minimum_routable_minutes,
            minimum_places_per_activity_kind=minimum_places_per_activity_kind,
            capacity_multiplier=capacity_multiplier,
            full_population_capacity_multiplier=full_population_capacity_multiplier,
            allow_planned=allow_planned,
            run_runtime_verification=run_runtime_verification,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE_NAME") from error
    typer.echo(_render(receipt, output_format))


if __name__ == "__main__":
    app()
