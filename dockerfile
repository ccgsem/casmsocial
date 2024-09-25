FROM library/python:3.12-bullseye

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
      apt-get install -y mpich libgdal-dev \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY poetry.lock pyproject.toml README.rst LICENSE ./

RUN python3 -m venv env && \
    ./env/bin/python3 -m pip install setuptools && \
    ./env/bin/python3 -m pip install poetry

COPY casmsocial casmsocial

ENV CC=mpicc
ENV CXX=mpicxx

RUN ./env/bin/poetry install
RUN ./env/bin/poetry build
