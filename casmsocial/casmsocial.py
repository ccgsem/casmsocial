"""Main module."""
from repast4py import parameters
from mpi4py import MPI
from typing import Dict

from casmsocial.model import (
    Model,
    get_casmsocial_model
)


def run(params: Dict):
    """Run the model."""
    model_name = params['model.name']
    
    print(f"Retrieving model <{model_name}>...")

    try:
        ModelCreator = get_casmsocial_model(params['model.name'])
    except ValueError as ve:        
        print(f"Error: {ve}")
        return

    print(f"Running model <{model_name}>...")

    model = ModelCreator(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)