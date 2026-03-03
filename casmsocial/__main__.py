"""Main module."""

import sys

from loguru import logger
from mpi4py import MPI
from repast4py.parameters import create_args_parser, init_params

from casmsocial.factory import Models, load_models


def run(params: dict):
    """Run the model."""
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    plugins = params.get("model.plugins", [])
    if plugins:
        logger.info(f"Loading model plugins: {plugins}")
        load_models(plugins)

    model_name = params["model.name"]

    logger.info(f"Retrieving model <{model_name}>...")

    for model in Models.get_models():
        logger.info(model)

    try:
        ModelCreator = Models.create_model(params["model.name"])  # get_casmsocial_model(params['model.name'])
    except ValueError as ve:
        logger.error(f"Error: {ve}")
        return

    logger.info(f"Running model <{model_name}>...")

    model = ModelCreator(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
