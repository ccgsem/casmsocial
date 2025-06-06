#! /usr/bin/env bash
set -euo pipefail

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        log "✅ $1 completed successfully"
    else
        log "❌ $1 failed"
        exit 1
    fi
}

# Ensure proper permissions
log "Setting up workspace permissions..."
sudo chown -R vscode:vscode /workspaces/casmsocial
check_status "Setting workspace permissions"

# Initialize git if not already initialized
if [ ! -d .git ]; then
    log "Initializing git repository..."
    git init
    git config --global user.email "dev@example.com"
    git config --global user.name "Dev Container"
    check_status "Git initialization"
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d .venv ]; then
    log "Creating Python virtual environment..."
    python -m venv .venv
    check_status "Virtual environment creation"
fi

# Activate virtual environment
log "Activating virtual environment..."
source .venv/bin/activate
check_status "Virtual environment activation"

# Install Dependencies
log "Installing project dependencies..."
uv sync
check_status "Dependency installation"

# Install development tools
log "Installing development tools..."
uv pip install --upgrade pip pylint pytest pytest-cov black isort
check_status "Development tools installation"

# Install pre-commit hooks
log "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks
check_status "Pre-commit hooks installation"

# Verify Python environment
log "Verifying Python environment..."
python --version
which python
check_status "Python environment verification"

# Verify MPI installation
log "Verifying MPI installation..."
mpirun --version
check_status "MPI verification"

log "🎉 Development container setup completed successfully!"
