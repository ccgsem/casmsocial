# Base stage with common dependencies
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
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

# Optional local build override for environments whose certificate store does
# not trust the PyTorch CPU wheel host used by the lock file.
ARG UV_INSECURE_HOST=""
ENV UV_INSECURE_HOST=${UV_INSECURE_HOST}

# Development stage
FROM base AS dev
# pytest/pytest-cov are Python (PyPI) packages, not Debian packages -- no
# such apt package exists. They're already managed via uv (pyproject.toml's
# dev dependency group), so only pylint belongs here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sudo \
    python3-venv \
    python3-dev \
    pylint \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash vscode && \
    chown -R vscode:vscode /app && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode && \
    chmod 0440 /etc/sudoers.d/vscode

# Switch to non-root user
USER vscode

# Copy the lockfile and `pyproject.toml` into the image
COPY --chown=vscode:vscode uv.lock /app/uv.lock
COPY --chown=vscode:vscode pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --frozen --no-install-project

# uv sync installs into /app/.venv, not the system Python -- put it on PATH
# so a plain `python`/terminal in this container resolves to the venv.
ENV PATH="/app/.venv/bin:${PATH}"

# Production stage
FROM base AS prod
# Copy the lockfile and `pyproject.toml` into the image
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the project into the image
COPY . /app

# Sync the project
RUN uv sync --frozen --no-dev

# uv sync installs into /app/.venv, not the system Python -- put it on PATH
# so mpirun's bare `python` resolves to the venv where casmsocial/mpi4py/
# repast4py are actually installed.
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["mpirun", "-n", "1", "python", "-m", "casmsocial", "config/casmsocial.yaml"]

# Compose MPI stage: local validation image for one-rank-per-container runs.
# MPICH Hydra launches remote ranks over SSH, so containers built from this
# stage share an image-local SSH key. Do not publish this target as a
# production image.
FROM prod AS compose-mpi
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh \
    && ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" \
    && cp /root/.ssh/id_ed25519.pub /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n" > /root/.ssh/config \
    && chmod 600 /root/.ssh/config \
    && printf "\nPermitRootLogin yes\nPasswordAuthentication no\nPubkeyAuthentication yes\n" >> /etc/ssh/sshd_config

CMD ["/usr/sbin/sshd", "-D", "-e"]

# Tools stage: adds the optional `partitioning` extra (pymetis), used only
# by casmsocial.network_partitioner_ducklake -- an offline CLI script, not
# part of the simulation runtime. Kept out of `prod` because pymetis has no
# prebuilt wheel for every platform (e.g. manylinux aarch64 at the time of
# writing) and must compile METIS from source there. Build explicitly with
# `--target tools` when the partitioner script is needed.
FROM prod AS tools
RUN uv sync --frozen --no-dev --extra partitioning
