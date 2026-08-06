"""Load and validate bundled Colorado Front Range dataset profiles."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Strict base for public build-contract sections."""

    model_config = ConfigDict(extra="forbid")


class Geography(ContractModel):
    boundary_id: str
    state: Literal["CO"]
    home_cbsa_codes: list[str]


class Population(ContractModel):
    mode: Literal["stratified_sample", "full_boundary_population"]
    person_limit: int | None
    seed: int
    strata: list[str] | None = None
    minimum_persons_per_cbsa: int | None = None
    require_exactly_one_weekday_home: bool
    require_endpoint_complete_social_ties: bool


class Sources(ContractModel):
    synthetic_population: str
    education_sites: str
    county_boundaries: str
    time_use: str
    destinations: str


class Schedules(ContractModel):
    day_types: list[Literal["weekday", "weekend"]]
    diary_day_start_local_time: str
    donor_assignment_seed: int


class Routing(ContractModel):
    model: str
    minimum_minutes: int
    average_speed_kph: float
    maximum_minutes: int
    destination_capacity_multiplier: float | None
    full_population_capacity_multiplier: float | None = None
    person_partitions: int
    feasibility_partitions: int
    resume_required: bool = False


class Validation(ContractModel):
    require_all_cbsas_represented: bool
    maximum_noncontiguous_transitions: int
    maximum_invalid_intervals: int
    maximum_unknown_stationary_places: int
    maximum_capacity_exceedances: int
    require_single_rank_smoke: bool = False
    require_two_rank_equivalence: bool = False
    require_partition_manifests: bool = False
    require_aggregate_acceptance: bool = False


class Runtime(ContractModel):
    model: str
    default_day_type: Literal["weekday", "weekend"]
    default_duration_hours: int | None = None
    default_ranks: int | None = None
    verification_ranks: int | None = None
    agent_log_enabled: bool
    aggregate_diagnostics_enabled: bool


class Governance(ContractModel):
    classification: str
    publish_inputs: bool
    calibrated: bool


class ColoradoDatasetProfile(ContractModel):
    """Validated, standalone Colorado dataset build contract."""

    schema_version: Literal[1]
    profile_id: str
    profile_version: str
    release_status: Literal["supported", "planned"]
    description: str
    geography: Geography
    population: Population
    sources: Sources
    schedules: Schedules
    routing: Routing
    validation: Validation
    runtime: Runtime
    governance: Governance
    release_blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> ColoradoDatasetProfile:
        codes = self.geography.home_cbsa_codes
        if len(codes) != len(set(codes)) or any(len(code) != 5 or not code.isdigit() for code in codes):
            raise ValueError("home_cbsa_codes must contain unique five-digit codes")
        if self.population.mode == "stratified_sample":
            if not self.population.person_limit or not self.population.strata:
                raise ValueError("stratified samples require person_limit and strata")
        elif self.population.person_limit is not None:
            raise ValueError("full population profiles cannot set person_limit")
        if self.release_status == "supported" and self.routing.destination_capacity_multiplier is None:
            raise ValueError("supported profiles require an accepted capacity multiplier")
        if self.release_status == "planned" and not self.release_blockers:
            raise ValueError("planned profiles require release blockers")
        if self.runtime.agent_log_enabled or self.governance.publish_inputs:
            raise ValueError("public profiles cannot enable agent logs or publish identifier-bearing inputs")
        return self


_PACKAGE = "casmsocial.datasets.colorado_front_range"


def _yaml_resource(directory: str, filename: str) -> dict:
    resource = files(_PACKAGE).joinpath(directory, filename)
    if not resource.is_file():
        raise ValueError(f"Unknown bundled Colorado resource: {filename}")
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Colorado resource must contain a mapping: {filename}")
    return content


def list_profiles() -> list[str]:
    """Return stable names for every bundled Colorado profile."""
    root = files(_PACKAGE).joinpath("profiles")
    return sorted(resource.name.removesuffix(".yaml") for resource in root.iterdir() if resource.name.endswith(".yaml"))


def load_profile(name: str) -> ColoradoDatasetProfile:
    """Load one bundled profile by filename stem."""
    if name not in list_profiles():
        raise ValueError(f"Unknown Colorado profile {name!r}; choose from {', '.join(list_profiles())}")
    return ColoradoDatasetProfile.model_validate(_yaml_resource("profiles", f"{name}.yaml"))


def load_osm_attribution() -> str:
    """Return the bundled OpenStreetMap attribution and distribution notice."""
    resource = files(_PACKAGE).joinpath("assets", "OPENSTREETMAP_ATTRIBUTION.md")
    return resource.read_text(encoding="utf-8").rstrip()


def load_migrated_code_provenance() -> dict:
    """Return the machine-readable private-to-public code migration record."""
    return _yaml_resource("assets", "migrated_code_provenance.yaml")


def load_release_review_policy() -> dict:
    """Return the machine-readable release-review and approval policy."""
    return _yaml_resource("assets", "release_review_policy.yaml")


def load_source_licenses() -> dict:
    """Return the machine-readable source-license audit."""
    return _yaml_resource("assets", "source_licenses.yaml")
