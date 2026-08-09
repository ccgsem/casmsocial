#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

workspace="${WORKSPACE_FOLDER:-/workspace/casmsocial}"
cd "$workspace"

log "Preparing writable devcontainer volumes..."
sudo mkdir -p "$workspace/.venv" /home/vscode/.cache/uv
sudo chown -R "$(id -u):$(id -g)" "$workspace/.venv" /home/vscode/.cache/uv

export CC="${CC:-mpicxx}"
export CXX="${CXX:-mpicxx}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export VIRTUAL_ENV="$workspace/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

log "Installing project dependencies..."
uv sync --frozen

log "Installing pre-commit hooks..."
uv run pre-commit install || log "pre-commit hook installation skipped"

log "Verifying Python environment..."
python --version
which python

log "Verifying MPI installation..."
mpirun --version
mpirun -n 2 python -c 'from mpi4py import MPI; print(f"rank {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}")'

log "Development container setup completed successfully."
