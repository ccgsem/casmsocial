"""Load the versioned Colorado Front Range metropolitan boundary."""

from __future__ import annotations

from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_PACKAGE = "casmsocial.datasets.colorado_front_range"


class BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetropolitanArea(BoundaryModel):
    cbsa_code: str
    name: str
    county_geoids: list[str]

    @field_validator("cbsa_code")
    @classmethod
    def validate_cbsa(cls, value: str) -> str:
        if len(value) != 5 or not value.isdigit():
            raise ValueError("cbsa_code must contain five digits")
        return value

    @field_validator("county_geoids")
    @classmethod
    def validate_counties(cls, values: list[str]) -> list[str]:
        if not values or any(len(value) != 5 or not value.isdigit() for value in values):
            raise ValueError("county_geoids must contain five-digit GEOIDs")
        if len(values) != len(set(values)):
            raise ValueError("county_geoids must be unique within an MSA")
        return values


class ColoradoBoundary(BoundaryModel):
    schema_version: int
    boundary_id: str
    state: str
    source: dict[str, str]
    metropolitan_statistical_areas: list[MetropolitanArea]

    @model_validator(mode="after")
    def validate_unique_geography(self) -> ColoradoBoundary:
        cbsas = [area.cbsa_code for area in self.metropolitan_statistical_areas]
        counties = [geoid for area in self.metropolitan_statistical_areas for geoid in area.county_geoids]
        if len(cbsas) != len(set(cbsas)):
            raise ValueError("CBSA codes must be unique")
        if len(counties) != len(set(counties)):
            raise ValueError("Each county must belong to exactly one bundled CBSA")
        return self


def load_boundary() -> ColoradoBoundary:
    """Load and validate the bundled 2023 six-metropolitan-area boundary."""
    resource = files(_PACKAGE).joinpath("assets", "boundary_2023.yaml")
    return ColoradoBoundary.model_validate(yaml.safe_load(resource.read_text(encoding="utf-8")))


def cbsa_by_county_geoid(boundary: ColoradoBoundary | None = None) -> dict[str, str]:
    """Map every bundled county GEOID to exactly one CBSA code."""
    selected = boundary or load_boundary()
    return {geoid: area.cbsa_code for area in selected.metropolitan_statistical_areas for geoid in area.county_geoids}
