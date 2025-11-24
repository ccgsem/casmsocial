#!/bin/bash
#SBATCH --job-name=heat_risk_array
#SBATCH --array=1-750%50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/heat_risk_%a.log
#SBATCH --error=logs/heat_risk_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=${USER}@example.com

# Heat Risk Model Multi-Imputation and Multi-Experiment Job Array
#
# This script submits a SLURM job array for running the heat risk model
# across 30 imputations and 25 experiments (750 total jobs)
#
# Configuration:
#   - Imputation values: 1-30
#   - Experiment IDs: 1-25
#   - Total jobs: 30 * 25 = 750
#   - Job array limit: 50 concurrent jobs (adjust %50 as needed)
#
# Required Environment Variables:
#   - CASMSOCIAL_VENV_PATH: Path to Python virtual environment (required)
#   - CASMSOCIAL_DATA_PATH: Path to data directory (required)
#   - CONFIG_FILE: Path to config YAML (optional, default: config/enhanced_heat_risk_example.yaml)
#   - OUTPUT_DIR: Output directory (optional, default: hpc_outputs)
#
# Usage:
#   CASMSOCIAL_VENV_PATH=/path/to/venv sbatch submit_heat_risk_array.slurm
#
# With custom configuration:
#   CASMSOCIAL_VENV_PATH=/path/to/venv CONFIG_FILE=config/custom.yaml sbatch submit_heat_risk_array.slurm
#
# With custom output directory:
#   CASMSOCIAL_VENV_PATH=/path/to/venv OUTPUT_DIR=/scratch/results sbatch submit_heat_risk_array.slurm

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
CONFIG_FILE="${CONFIG_FILE:-config/enhanced_heat_risk_example.yaml}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/hpc_outputs}"

# Python virtual environment path (required for HPC)
# Set via: CASMSOCIAL_VENV_PATH=/path/to/venv sbatch submit_heat_risk_array.slurm
if [ -z "$CASMSOCIAL_VENV_PATH" ]; then
    echo "ERROR: CASMSOCIAL_VENV_PATH environment variable not set"
    echo "Please set it to your Python virtual environment path"
    echo "Example: CASMSOCIAL_VENV_PATH=/scratch/user/venv sbatch submit_heat_risk_array.slurm"
    exit 1
fi

if [ ! -d "$CASMSOCIAL_VENV_PATH" ]; then
    echo "ERROR: Virtual environment path does not exist: $CASMSOCIAL_VENV_PATH"
    exit 1
fi

# Array dimensions (1-based indexing)
IMPUTATION_MIN=1
IMPUTATION_MAX=30
EXPERIMENT_MIN=1
EXPERIMENT_MAX=25
NUM_EXPERIMENTS=$((EXPERIMENT_MAX - EXPERIMENT_MIN + 1))  # = 25

# Calculate values from array task ID
# Convert SLURM's 1-based array task ID to 0-based index for modulo arithmetic
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Map to imputation (cycles through experiments for each imputation)
# Imputation ranges from 1 to 30
IMPUTATION=$((ARRAY_INDEX / NUM_EXPERIMENTS + IMPUTATION_MIN))

# Map to experiment_id (cycles 1-25 for each imputation)
# experiment_id ranges from 1 to 25
EXPERIMENT_ID=$((ARRAY_INDEX % NUM_EXPERIMENTS + EXPERIMENT_MIN))

# ============================================================================
# Setup
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Load environment
cd "${PROJECT_ROOT}"

# Source environment setup if available
if [ -f .env ]; then
    set +a  # Don't export by default
    source .env
    set -a
fi

# Ensure CASMSOCIAL_DATA_PATH is set
if [ -z "$CASMSOCIAL_DATA_PATH" ]; then
    echo "ERROR: CASMSOCIAL_DATA_PATH environment variable not set"
    exit 1
fi

# ============================================================================
# Logging and Job Information
# ============================================================================

echo "=========================================="
echo "Heat Risk Model HPC Job Array"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Index: $ARRAY_INDEX"
echo "Imputation: $IMPUTATION"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Host: $HOSTNAME"
echo "CPUs Available: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time Limit: $SLURM_JOB_NUM_CPUS"
echo "Config File: $CONFIG_FILE"
echo "Data Path: $CASMSOCIAL_DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# ============================================================================
# Create JSON parameter override
# ============================================================================

# Create a JSON string with parameter overrides
PARAMS_JSON=$(cat <<EOF
{
    "Imputation": ${IMPUTATION},
    "experiment_id": ${EXPERIMENT_ID}
}
EOF
)

echo "Parameter overrides:"
echo "$PARAMS_JSON"
echo ""

# ============================================================================
# Run the Model
# ============================================================================

# Build the command
# The JSON string will override the YAML values for Imputation and experiment_id
# Using repast4py.parameters format: config_file "json_params_string"

echo "Running command:"
echo "python -m casmsocial ${CONFIG_FILE} '${PARAMS_JSON}'"
echo ""

# Load HPC modules (adjust module names for your system)
echo "Loading HPC modules..."
unset PYTHONPATH
unset LD_LIBRARY_PATH
module load ucx/default
module load python/3.12.9
module load mpich/4.0.2

# Activate Python virtual environment
echo "Activating Python virtual environment: $CASMSOCIAL_VENV_PATH"
source ${CASMSOCIAL_VENV_PATH}/bin/activate

# Set environment for UCX (MPI communication)
export UCX_TLS=tcp

# Run the model with parameter overrides
# repast4py accepts: python -m module config_file "json_params"
# The JSON string overrides values from the YAML config file
echo "Running command:"
echo "mpirun -n 1 python -m casmsocial ${CONFIG_FILE} '${PARAMS_JSON}'"
echo ""

mpirun -n 1 python -m casmsocial \
    "${CONFIG_FILE}" \
    "${PARAMS_JSON}"

RUN_EXIT_CODE=$?

# ============================================================================
# Post-processing
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $RUN_EXIT_CODE"
echo "=========================================="

# Create a marker file for job completion
if [ $RUN_EXIT_CODE -eq 0 ]; then
    touch "${OUTPUT_DIR}/imputation_${IMPUTATION}/experiment_${EXPERIMENT_ID}/.completed"
    echo "Job completed successfully"
else
    echo "Job failed with exit code $RUN_EXIT_CODE"
    exit $RUN_EXIT_CODE
fi
