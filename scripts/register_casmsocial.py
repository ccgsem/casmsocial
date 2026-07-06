"""Register the casmsocial model and its canonical scenarios in a casmdb database.

This script is the authoritative source for casmsocial's casmdb registration.
Run it once per environment (or re-run to update) to seed the registry.

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
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from loguru import logger

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

CASMSOCIAL_MODEL_NAME = "casmsocial"
CASMSOCIAL_FACTORY_KEY = "casmsocial.citysim.citysocialmodel.CitySocialModel"


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
# scenario_parameters stores only the keys that differ from
# CasmPop.get_default_parameters().  run_scenario.py merges defaults first,
# so omitted keys automatically pick up their defaults.
#
# model.name is always included to ensure the correct subclass is instantiated
# (CasmPop.get_default_parameters() returns "casmsocial.casmpop.CasmPop" as
# the default, which would bypass CitySocialModel).
# ---------------------------------------------------------------------------


def _build_scenarios() -> dict[str, dict[str, Any]]:
    """Return the canonical scenario parameter deltas keyed by scenario name."""
    return {
        "base": {
            # Uses all CasmPop defaults (dmv_100 tables) with CitySocialModel
            "model.name": CASMSOCIAL_FACTORY_KEY,
        },
        "dmv": {
            "model.name": CASMSOCIAL_FACTORY_KEY,
            "places.table": "rti_synth_pop_v2_dmv.places",
            "activities.table": "rti_synth_pop_v2_dmv.activities",
            "contacts.table": "rti_synth_pop_v2_dmv.contacts",
            "persons.table": "rti_synth_pop_v2_dmv.persons",
        },
        "dmv_100": {
            # Explicit — same tables as defaults, but makes intent unambiguous
            "model.name": CASMSOCIAL_FACTORY_KEY,
            "places.table": "rti_synth_pop_v2_dmv_100.places",
            "activities.table": "rti_synth_pop_v2_dmv_100.activities",
            "contacts.table": "rti_synth_pop_v2_dmv_100.contacts",
            "persons.table": "rti_synth_pop_v2_dmv_100.persons",
        },
        "dc": {
            "model.name": CASMSOCIAL_FACTORY_KEY,
            "places.table": "rti_synth_pop_v2_dc.places",
            "activities.table": "rti_synth_pop_v2_dc.activities",
            "contacts.table": "rti_synth_pop_v2_dc.contacts",
            "persons.table": "rti_synth_pop_v2_dc.persons",
        },
        "dc_5000": {
            "model.name": CASMSOCIAL_FACTORY_KEY,
            "places.table": "rti_synth_pop_v2_dc_5000.places",
            "activities.table": "rti_synth_pop_v2_dc_5000.activities",
            "contacts.table": "rti_synth_pop_v2_dc_5000.contacts",
            "persons.table": "rti_synth_pop_v2_dc_5000.persons",
        },
    }


def _build_model_record(version: str, resources_uri: str) -> dict[str, Any]:
    """Build the casmdb model record dict for casmsocial."""
    return {
        "model_name": CASMSOCIAL_MODEL_NAME,
        "model_license": "Proprietary",
        "model_version": version,
        "model_version_date": str(date.today()),
        "model_programming_languages": ["Python"],
        "model_description": (
            "casmsocial: agent-based model of synthetic population dynamics "
            "using repast4py, MPI, and DuckDB/DuckLake."
        ),
        "model_authors": [
            {"name": "Jon Cline", "email": "jon.c.cline@gmail.com"},
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


# Scenario descriptions
_DESCRIPTIONS: dict[str, str] = {
    "base": ("Baseline scenario using default dmv_100 synthetic population tables."),
    "dmv": ("Full DC-Maryland-Virginia metro area synthetic population (all ages, all zones)."),
    "dmv_100": (
        "DC-Maryland-Virginia metro area synthetic population — same tables as the default "
        "base scenario; included for explicitness."
    ),
    "dc": ("District of Columbia synthetic population (all ages, all zones)."),
    "dc_5000": ("DC synthetic population sampled to 5,000 persons — suitable for rapid iteration."),
}

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
    scenarios = _build_scenarios()
    model_record = _build_model_record(version, args.resources_uri)

    if args.dry_run:
        print("\n=== DRY RUN — nothing will be written ===\n")
        print(f"Model record ({args.db}):")
        print(json.dumps(model_record, indent=2, default=str))
        print(f"\nScenarios ({len(scenarios)}):")
        for name, params in scenarios.items():
            print(f"\n  [{name}]")
            print(f"  description: {_DESCRIPTIONS.get(name, '')}")
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
                description=_DESCRIPTIONS.get(scenario_name, ""),
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
