"""Main module."""

import sys

from loguru import logger
from mpi4py import MPI
from repast4py.parameters import create_args_parser, init_params

from casmsocial.factory import ModelNotFoundError, Models, load_models


def _param_enabled(value, default: bool = False) -> bool:
    """Parse common truthy values from config parameters."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _rank0_info_filter(rank: int):
    """Allow INFO logs only on rank 0 while preserving warnings and errors on all ranks."""
    warning_level = logger.level("WARNING").no

    def _filter(record):
        return rank == 0 or record["level"].no >= warning_level

    return _filter


def _configure_logging(params: dict, comm) -> None:
    """Configure process logging from runtime parameters."""
    logger.remove(0)
    rank0_only = _param_enabled(params.get("logging.rank0_only", False), False)
    log_filter = _rank0_info_filter(comm.Get_rank()) if rank0_only else None
    logger.add(sys.stderr, level=params.get("logging.level", "INFO"), filter=log_filter)


def load_builtin_models() -> None:
    """Register built-in models without relying on package import side effects."""
    import casmsocial.casmpop  # noqa: F401
    import casmsocial.citysim.citysocialmodel  # noqa: F401
    import casmsocial.citysim.dcmetro_schedule_model  # noqa: F401


def run(params: dict) -> int:
    """Run the model."""
    _configure_logging(params, MPI.COMM_WORLD)

    load_builtin_models()

    plugins = params.get("model.plugins", [])
    if plugins:
        logger.info(f"Loading model plugins: {plugins}")
        load_models(plugins)

    model_name = params["model.name"]

    logger.info(f"Retrieving model <{model_name}>...")

    for model in Models.get_models():
        logger.info(model)

    try:
        model_creator = Models.create_model(params["model.name"])
    except ModelNotFoundError as exc:
        logger.error(str(exc))
        return 1

    logger.info(f"Running model <{model_name}>...")

    model = model_creator(MPI.COMM_WORLD, params)
    model.start()
    return 0


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    raise SystemExit(run(params))
