"""Main module."""
from mpi4py import MPI
from repast4py.parameters import create_args_parser, init_params

from casmsocial.factory import Models


def run(params: dict):
    """Run the model."""
    model_name = params['model.name']

    print(f"Retrieving model <{model_name}>...")

    for model in Models.get_models():
        print(model)

    try:
        ModelCreator = Models.create_model(params['model.name'])  #get_casmsocial_model(params['model.name'])
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    print(f"Running model <{model_name}>...")

    model = ModelCreator(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
