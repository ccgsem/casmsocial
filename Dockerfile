# Base stage with common dependencies
FROM python:3.12-slim as base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    mpich \
    libmpich-dev \
    libomp-dev \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=mpicxx CXX=mpicxx
WORKDIR /app

# Development stage
FROM base as dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3-venv \
    python3-dev \
    pylint \
    pytest \
    pytest-cov \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash vscode && \
    chown -R vscode:vscode /app

# Switch to non-root user
USER vscode

# Copy the lockfile and `pyproject.toml` into the image
COPY --chown=vscode:vscode uv.lock /app/uv.lock
COPY --chown=vscode:vscode pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --frozen --no-install-project

# Production stage
FROM base as prod
# Copy the lockfile and `pyproject.toml` into the image
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the project into the image
COPY . /app

# Sync the project
RUN uv sync --frozen

CMD ["mpirun", "-n 1", "python", "-m casmsocial.runner", "config/casmsocial.yaml"]
