# casmsocial

[![Release](https://img.shields.io/github/v/release/clinejc/casmsocial)](https://img.shields.io/github/v/release/clinejc/casmsocial)
[![Build status](https://img.shields.io/github/actions/workflow/status/clinejc/casmsocial/main.yml?branch=main)](https://github.com/clinejc/casmsocial/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/clinejc/casmsocial/branch/main/graph/badge.svg)](https://codecov.io/gh/clinejc/casmsocial)
[![Commit activity](https://img.shields.io/github/commit-activity/m/clinejc/casmsocial)](https://img.shields.io/github/commit-activity/m/clinejc/casmsocial)
[![License](https://img.shields.io/github/license/clinejc/casmsocial)](https://img.shields.io/github/license/clinejc/casmsocial)

casmsocial is a Python framework for implementing agent-based models that simulate the dynamics of a synthetic population

- **Github repository**: <https://github.com/clinejc/casmsocial/>
- **Documentation** <https://clinejc.github.io/casmsocial/>

## Installation

Install the environment with

```bash
export CC=mpicxx; export CXX=mpicxx
make install
```

To build a Docker image for `casmsocial`:

* on the MITRE network

    ```bash
    docker build -t casmsocial . -f Dockerfile-mitre
    ```

* off the MITRE network

    ```bash
    docker build -t casmsocial . -f Dockerfile
    ```

## Launch the modeling environment:

```bash
% source ./.venv/bin/activate
(casmsocial) ...
```

## Quickstart: running the model
There are three ways to run the model

1. Run from the command line using `uv run`
2. Run fromm the command line using virtualenv
3. Run from

To run (option 1):

```bash
% uv run mpirun -n 1 python -m casmsocial.runner config/casmsocial.yaml
```

To run with the virtual environment (option 2):

```bash
% source ./.venv/bin/activate
(casmsocial)
(casmsocial) mpirun -n 1 python -m casmsocial.runner config/casmsocial.yaml
....
(casmsocial) deactivate
%
```

To run in a Docker container (option 3):

```
docker run \
        -v `pwd`/config:/app/config \
        -v ~/Library/CloudStorage/Box-Box/Predicting_Population_Response/data:/app/data \
        --rm -it --entrypoint bash casmsocial
root@d332db7567f5:/app# uv run mpirun -n 1 python -m casmsocial.runner config/casmsocial_wc.yaml
```

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
