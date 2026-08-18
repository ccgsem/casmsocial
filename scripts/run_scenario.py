"""Run a casmsocial simulation from a casmdb scenario.

This script is the integration layer between the casmdb scenario registry and
the casmsocial simulation runner. It is intentionally casmsocial-specific: all
knowledge of default parameters, parameter validation, and model registration
lives here, not in casmdb.

Usage (single rank)::

    uv run python scripts/run_scenario.py \\
        --scenario-db http://localhost:8000 \\
        --model casmsocial \\
        --model-version 2.4.0 \\
        --scenario dmv_100

Usage (multi-rank via MPI)::

    uv run mpirun -n 8 python scripts/run_scenario.py \\
        --scenario-db /path/to/models.db \\
        --model casmsocial \\
        --model-version 2.4.0 \\
        --scenario dmv_100 \\
        --param duration.hours=48 \\
        --param behavior.engine=schedule

Usage (catalog-only check)::

    uv run python scripts/run_scenario.py \\
        --scenario-db http://localhost:8000 \\
        --model casmsocial \\
        --model-version 2.4.0 \\
        --scenario dmv_100 \\
        --resolve-only

Parameter precedence (lowest → highest):
    1. CasmPop.get_default_parameters()
    2. scenario_parameters stored in casmdb
    3. --param key=value overrides on the command line
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from loguru import logger

# ---------------------------------------------------------------------------
# casmdb import — soft dependency
# ---------------------------------------------------------------------------


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


class _CatalogHttpError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"casmdb API request failed with HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class _CatalogHttpClient:
    """Small stdlib client for reading scenarios from the casmdb REST API."""

    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @staticmethod
    def _segment(value: str) -> str:
        return quote(str(value), safe="")

    def _request(self, method: str, path: str):
        request = Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
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

    def get_scenario_parameters(self, scenario_name: str, model_name: str, model_version: str):
        try:
            scenario = self._request(
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
        return scenario["scenario_parameters"]

    def close(self) -> None:
        pass


def _require_casmdb():
    """Import ScenarioDB, raising a clear error if casmdb is not installed."""
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


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def _load_default_params() -> dict[str, Any]:
    """Return the merged defaults from all built-in casmsocial model classes."""
    from casmsocial.casmpop import CasmPop

    return CasmPop.get_default_parameters()


def _parse_override(raw: str) -> tuple[str, Any]:
    """Parse a 'key=value' override string into a (key, typed-value) pair.

    Attempts int → float → bool → str coercion in that order.
    """
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--param must be in key=value format, got: {raw!r}")
    key, _, raw_value = raw.partition("=")
    key = key.strip()

    # int
    try:
        return key, int(raw_value)
    except ValueError:
        pass
    # float
    try:
        return key, float(raw_value)
    except ValueError:
        pass
    # bool
    if raw_value.lower() in ("true", "yes", "on"):
        return key, True
    if raw_value.lower() in ("false", "no", "off"):
        return key, False
    # string
    return key, raw_value


def _merge_params(
    defaults: dict[str, Any],
    scenario_params: dict[str, Any],
    overrides: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Merge parameter layers in precedence order."""
    merged = dict(defaults)
    merged.update(scenario_params)
    for key, value in overrides:
        merged[key] = value
    return merged


def _warn_unknown_keys(
    params: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    """Warn about keys not present in the known default parameter space.

    Unknown keys are not rejected — subclass models and future parameters may
    extend the base set — but surfacing them early avoids silent typo bugs.
    """
    known = set(defaults.keys())
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        logger.warning(
            "The following parameter keys are not in CasmPop.get_default_parameters() "
            "and will be passed through unvalidated: {}",
            unknown,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run a casmsocial simulation from a named scenario stored in a casmdb database."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario-db",
        required=True,
        metavar="PATH",
        help="casmdb DuckDB path, s3:// URI, or http(s):// API base URL.",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="NAME",
        help="Model name as registered in casmdb (e.g. 'casmsocial').",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        metavar="VERSION",
        help="Exact model version string as registered in casmdb (e.g. '2.4.0').",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        metavar="NAME",
        help="Scenario name as registered in casmdb (e.g. 'dmv_100').",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="params",
        help=(
            "Override a single parameter after scenario params are loaded. "
            "May be repeated. Values are coerced to int/float/bool/str."
        ),
    )
    parser.add_argument(
        "--resolve-only",
        "--dry-run",
        action="store_true",
        help=("Resolve and validate catalog parameters, then exit without " "starting the simulation."),
    )
    parser.add_argument(
        "--print-params",
        action="store_true",
        help="With --resolve-only, print merged parameters as sorted JSON.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Parse CLI overrides before doing anything expensive.
    try:
        overrides = [_parse_override(p) for p in args.params]
    except argparse.ArgumentTypeError as exc:
        logger.error(str(exc))
        return 2

    # Fetch scenario parameters.
    db = _open_catalog(args.scenario_db)
    try:
        scenario_params = db.get_scenario_parameters(
            scenario_name=args.scenario,
            model_name=args.model,
            model_version=args.model_version,
        )
    finally:
        db.close()

    if scenario_params is None:
        logger.error(
            "Scenario {!r} not found for model {!r} version {!r} in {}",
            args.scenario,
            args.model,
            args.model_version,
            args.scenario_db,
        )
        return 1

    # Build final parameter dict.
    defaults = _load_default_params()
    params = _merge_params(defaults, scenario_params, overrides)
    _warn_unknown_keys(params, defaults)

    logger.info(
        "Loaded scenario {!r} from model {!r} {} — {} parameters",
        args.scenario,
        args.model,
        args.model_version,
        len(params),
    )

    if args.resolve_only:
        logger.info(
            "Resolve-only mode: not starting simulation for scenario {!r}.",
            args.scenario,
        )
        if args.print_params:
            print(json.dumps(params, indent=2, sort_keys=True, default=str))
        return 0

    # Delegate to the standard casmsocial runner.
    from casmsocial.__main__ import run

    return run(params)


if __name__ == "__main__":
    raise SystemExit(main())
