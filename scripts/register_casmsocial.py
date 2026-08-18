"""Register the casmsocial model and its canonical scenarios in a casmdb database.

The YAML files under scenarios/casmsocial are the authoritative source for
casmsocial's canonical scenarios. Run this script once per environment (or
re-run to update) to validate those files and seed the registry.

It is intentionally casmsocial-specific — casmdb itself is model-agnostic.

Usage::

    # Dry-run: print what would be written without touching the DB
    uv run python scripts/register_casmsocial.py --db /path/to/models.db --dry-run

    # Register / update
    uv run python scripts/register_casmsocial.py --db /path/to/models.db

    # Specify version explicitly (defaults to casmsocial.__version__)
    uv run python scripts/register_casmsocial.py --db /path/to/models.db --version 2.4.0

    # REST API
    uv run python scripts/register_casmsocial.py --db http://localhost:8000

Idempotent: running multiple times is safe. Existing records are updated in-place.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

CASMSOCIAL_MODEL_NAME = "casmsocial"
CASMSOCIAL_FACTORY_KEY = "casmsocial.citysim.citysocialmodel.CitySocialModel"
SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "casmsocial"
REQUIRED_SCENARIO_FIELDS = {"scenario_name", "model_name", "description", "parameters"}
REQUIRED_SCENARIO_PARAMETER_KEYS = {
    "model.name",
    "places.table",
    "households.table",
    "persons.table",
    "activities.table",
}


def _get_version() -> str:
    """Return the installed casmsocial package version."""
    try:
        from importlib.metadata import version

        return version("casmsocial")
    except Exception:
        # Fall back to reading pyproject.toml if the package isn't installed
        import re
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text()
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                return m.group(1)
        return "0.0.0"


# ---------------------------------------------------------------------------
# Canonical scenario definitions
#
# scenario_parameters stores the canonical scenario parameter layer.
# run_scenario.py merges defaults first, so omitted keys still pick up their
# defaults, but the YAML files should explicitly include keys that identify
# data tables and model identity.
#
# model.name is always included to ensure the correct subclass is instantiated
# (CasmPop.get_default_parameters() returns "casmsocial.casmpop.CasmPop" as
# the default, which would bypass CitySocialModel).
#
# The YAML files in scenarios/casmsocial are the source of truth. This script
# only validates and registers them; do not duplicate scenario definitions here.
# ---------------------------------------------------------------------------


def _load_scenario_specs(scenario_dir: Path = SCENARIO_DIR) -> dict[str, dict[str, Any]]:
    """Load canonical scenario specs from YAML files keyed by scenario name."""
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory does not exist: {scenario_dir}")

    specs: dict[str, dict[str, Any]] = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: scenario file must contain a mapping")

        missing_fields = REQUIRED_SCENARIO_FIELDS - set(raw)
        if missing_fields:
            raise ValueError(f"{path}: missing required fields {sorted(missing_fields)}")

        name = raw["scenario_name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: scenario_name must be a non-empty string")
        if name in specs:
            raise ValueError(f"{path}: duplicate scenario_name {name!r}")
        if path.stem != name:
            raise ValueError(f"{path}: file stem must match scenario_name {name!r}")

        model_name = raw["model_name"]
        if model_name != CASMSOCIAL_MODEL_NAME:
            raise ValueError(f"{path}: model_name must be {CASMSOCIAL_MODEL_NAME!r}")

        description = raw["description"]
        if not isinstance(description, str):
            raise ValueError(f"{path}: description must be a string")

        params = raw["parameters"]
        if not isinstance(params, dict):
            raise ValueError(f"{path}: parameters must be a mapping")

        missing_params = REQUIRED_SCENARIO_PARAMETER_KEYS - set(params)
        if missing_params:
            raise ValueError(f"{path}: missing required parameters {sorted(missing_params)}")

        if params["model.name"] != CASMSOCIAL_FACTORY_KEY:
            raise ValueError(f"{path}: parameters.model.name must be {CASMSOCIAL_FACTORY_KEY!r}")

        specs[name] = {
            "description": description,
            "parameters": params,
        }

    if not specs:
        raise ValueError(f"No scenario YAML files found in {scenario_dir}")

    return specs


def _build_scenarios() -> dict[str, dict[str, Any]]:
    """Return canonical scenario parameter layers keyed by scenario name."""
    return {name: spec["parameters"] for name, spec in _load_scenario_specs().items()}


def _build_model_record(version: str, resources_uri: str) -> dict[str, Any]:
    """Build the casmdb model record dict for casmsocial."""
    return {
        "model_name": CASMSOCIAL_MODEL_NAME,
        "model_license": "MIT",
        "model_version": version,
        "model_version_date": str(date.today()),
        "model_programming_languages": ["Python"],
        "model_description": (
            "casmsocial: agent-based model of synthetic population dynamics "
            "using repast4py, MPI, and DuckDB/DuckLake."
        ),
        "model_authors": [
            {"name": "Jon C. Cline", "email": "jcline@mitre.org"},
        ],
        "model_resources_uri": resources_uri,
        "model_metadata": {
            "factory_key": CASMSOCIAL_FACTORY_KEY,
            "framework": "repast4py",
        },
        "model_parameters_schema": {},
        "model_inputs_schema": {},
        "model_outputs_schema": {},
    }


# ---------------------------------------------------------------------------
# casmdb helpers (soft dependency)
# ---------------------------------------------------------------------------


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


class _CatalogHttpError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"casmdb API request failed with HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class _CatalogHttpClient:
    """Small stdlib client for the casmdb REST API."""

    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @staticmethod
    def _segment(value: str) -> str:
        return quote(str(value), safe="")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, default=str).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                body = response.read()
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise _CatalogHttpError(exc.code, error_body) from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach casmdb API at {self.base_url}: {exc}") from exc

        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def get_model(self, model_name: str, model_version: str):
        try:
            return self._request(
                "GET",
                f"/models/{self._segment(model_name)}/{self._segment(model_version)}",
            )
        except _CatalogHttpError as exc:
            if exc.status_code == 404:
                return None
            raise

    def insert_model_from_dict(self, model_data: dict[str, Any]) -> bool:
        try:
            self._request("POST", "/models/", model_data)
            return True
        except _CatalogHttpError as exc:
            if exc.status_code == 400:
                return False
            raise

    def update_model(self, model_name: str, model_version: str, update_data: dict[str, Any]) -> bool:
        try:
            self._request(
                "PATCH",
                f"/models/{self._segment(model_name)}/{self._segment(model_version)}",
                update_data,
            )
            return True
        except _CatalogHttpError as exc:
            if exc.status_code == 404:
                return False
            raise

    def get_scenario(self, scenario_name: str, model_name: str, model_version: str):
        try:
            return self._request(
                "GET",
                (
                    f"/scenarios/{self._segment(scenario_name)}/"
                    f"{self._segment(model_name)}/{self._segment(model_version)}"
                ),
            )
        except _CatalogHttpError as exc:
            if exc.status_code == 404:
                return None
            raise

    def insert_scenario_from_dict(self, scenario_data: dict[str, Any]) -> bool:
        try:
            self._request("POST", "/scenarios/", scenario_data)
            return True
        except _CatalogHttpError as exc:
            if exc.status_code == 400:
                return False
            raise

    def update_scenario(
        self,
        scenario_name: str,
        model_name: str,
        model_version: str,
        update_data: dict[str, Any],
    ) -> bool:
        try:
            self._request(
                "PATCH",
                (
                    f"/scenarios/{self._segment(scenario_name)}/"
                    f"{self._segment(model_name)}/{self._segment(model_version)}"
                ),
                update_data,
            )
            return True
        except _CatalogHttpError as exc:
            if exc.status_code == 404:
                return False
            raise

    def close(self) -> None:
        pass


def _require_casmdb():
    try:
        from casmdb import ScenarioDB  # type: ignore[import]

        return ScenarioDB
    except ImportError:
        logger.error(
            "casmdb is not installed. Install it with:\n"
            "    pip install casmdb\n"
            "or add it to this project's optional dependencies."
        )
        sys.exit(1)


def _open_catalog(db_location: str):
    if _is_http_url(db_location):
        return _CatalogHttpClient(db_location)

    ScenarioDB = _require_casmdb()
    return ScenarioDB(db_path=db_location)


def _upsert_model(db: Any, record: dict, force_update: bool = False) -> None:
    """Insert the model record if it does not exist.

    For an existing record with the same (name, version), updates are skipped
    by default: a registered version's metadata is treated as immutable.
    Pass force_update=True (or use the casmdb API directly) to override.

    Note: DuckDB FK constraints can fire incorrectly on UPDATE when child rows
    reference the same PK, so we avoid UPDATE when not necessary.
    """
    name = record["model_name"]
    version = record["model_version"]
    existing = db.get_model(name, version)
    if existing is None:
        ok = db.insert_model_from_dict(record)
        if ok:
            logger.info("Inserted model {!r} version {}", name, version)
        else:
            logger.error("Failed to insert model {!r} version {}", name, version)
    elif force_update:
        mutable_fields = {k: v for k, v in record.items() if k not in ("model_name", "model_version")}
        ok = db.update_model(name, version, mutable_fields)
        if ok:
            logger.info("Updated model {!r} version {}", name, version)
        else:
            logger.warning("Update failed for model {!r} version {} — use casmdb API directly", name, version)
    else:
        logger.info(
            "Model {!r} version {} already registered — skipping (use --force-update-model to overwrite)", name, version
        )


def _upsert_scenario(
    db: Any,
    scenario_name: str,
    model_name: str,
    model_version: str,
    scenario_params: dict,
    description: str,
) -> None:
    """Insert or update a scenario record."""
    existing = db.get_scenario(scenario_name, model_name, model_version)
    record = {
        "scenario_name": scenario_name,
        "model_name": model_name,
        "model_version": model_version,
        "scenario_parameters": scenario_params,
        "scenario_creation_date": str(date.today()),
        "scenario_description": description,
    }
    if existing is None:
        ok = db.insert_scenario_from_dict(record)
        if ok:
            logger.info("Inserted scenario {!r} for {!r} {}", scenario_name, model_name, model_version)
        else:
            logger.error(
                "Failed to insert scenario {!r} for {!r} {}",
                scenario_name,
                model_name,
                model_version,
            )
    else:
        ok = db.update_scenario(
            scenario_name,
            model_name,
            model_version,
            {
                "scenario_parameters": scenario_params,
                "scenario_description": description,
                "scenario_creation_date": str(date.today()),
            },
        )
        if ok:
            logger.info("Updated scenario {!r} for {!r} {}", scenario_name, model_name, model_version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register casmsocial model + canonical scenarios in a casmdb database. " "Idempotent — safe to re-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="casmdb DuckDB path, s3:// URI, or http(s):// API base URL.",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="VERSION",
        help=(
            "casmsocial version string to register (default: auto-detected from "
            "installed package or pyproject.toml)."
        ),
    )
    parser.add_argument(
        "--resources-uri",
        default="",
        metavar="URI",
        help=(
            "Base URI for model resources (e.g. s3://my-bucket/models). " "Used by casmdb to derive output directories."
        ),
    )
    parser.add_argument(
        "--force-update-model",
        action="store_true",
        help=(
            "Re-write the model record even if it already exists. "
            "Off by default because a registered version's metadata is treated as immutable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching the database.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    version = args.version or _get_version()
    scenario_specs = _load_scenario_specs()
    scenarios = {name: spec["parameters"] for name, spec in scenario_specs.items()}
    descriptions = {name: spec["description"] for name, spec in scenario_specs.items()}
    model_record = _build_model_record(version, args.resources_uri)

    if args.dry_run:
        print("\n=== DRY RUN — nothing will be written ===\n")
        print(f"Model record ({args.db}):")
        print(json.dumps(model_record, indent=2, default=str))
        print(f"\nScenarios ({len(scenarios)}):")
        for name, params in scenarios.items():
            print(f"\n  [{name}]")
            print(f"  description: {descriptions.get(name, '')}")
            print(f"  parameters:  {json.dumps(params, indent=4)}")
        return 0

    db = _open_catalog(args.db)
    try:
        _upsert_model(db, model_record, force_update=args.force_update_model)
        for scenario_name, scenario_params in scenarios.items():
            _upsert_scenario(
                db,
                scenario_name=scenario_name,
                model_name=CASMSOCIAL_MODEL_NAME,
                model_version=version,
                scenario_params=scenario_params,
                description=descriptions.get(scenario_name, ""),
            )
    finally:
        db.close()

    logger.info(
        "Registration complete: model {!r} version {} with {} scenarios in {}",
        CASMSOCIAL_MODEL_NAME,
        version,
        len(scenarios),
        args.db,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
