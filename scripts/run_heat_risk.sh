#!/bin/bash
#SBATCH --job-name="casmsocial"
#SBATCH --nodes=1
#SBATCH --time=90:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%u-%x-job%j.out
#SBATCH --export=ALL
#SBATCH --mem=50GB

# Parse arguments for Imputation, experiment_id, and YAML parameter file
# Usage examples:
#   sbatch script.sh --imputation 2 --experiment-id 42 --yaml-file /path/to/params.yaml
#   sbatch script.sh -i 2 -e 42 -y /path/to/params.yaml
#   sbatch script.sh 2 42 /path/to/params.yaml  (positional fallback)
#   sbatch script.sh 2 42                       (uses default YAML)
IMPUTATION=1
EXPERIMENT_ID=1
YAML_PATH="/projects/ARTSOC-POLICY-MODELING/config/casmsocial_wc_30_v2_hpc.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--imputation)
      IMPUTATION="$2"
      shift 2
      ;;
    -e|--experiment-id|--experiment_id)
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    -y|--yaml|--yaml-file|--config|--config-file)
      YAML_PATH="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      if [[ -z "${POS1_SET:-}" ]]; then
        IMPUTATION="$1"
        POS1_SET=1
        shift
      elif [[ -z "${POS2_SET:-}" ]]; then
        EXPERIMENT_ID="$1"
        POS2_SET=1
        shift
      elif [[ -z "${POS3_SET:-}" ]]; then
        YAML_PATH="$1"
        POS3_SET=1
        shift
      else
        echo "Unexpected argument: $1"
        exit 1
      fi
      ;;
  esac
done

# Validate arguments are integers
if ! [[ "$IMPUTATION" =~ ^[0-9]+$ ]]; then
  echo "Error: --imputation must be an integer (got '$IMPUTATION')"
  exit 1
fi
if ! [[ "$EXPERIMENT_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --experiment-id must be an integer (got '$EXPERIMENT_ID')"
  exit 1
fi

# Validate YAML file path
if [[ -z "$YAML_PATH" ]]; then
  echo "Error: YAML parameter file path is empty."
  exit 1
fi
if [[ ! -f "$YAML_PATH" ]]; then
  echo "Error: YAML parameter file not found at '$YAML_PATH'"
  exit 1
fi

unset PYTHONPATH
unset LD_LIBRARY_PATH

module load ucx/default
module load python/3.12.9
module load mpich/4.0.2

source /projects/ARTSOC-POLICY-MODELING/python/casmsocial/bin/activate

export UCX_TLS=tcp
export CASMSOCIAL_DATA_PATH=/projects/ARTSOC-POLICY-MODELING/data

JSON_ARG=$(printf '{"Imputation":%s,"experiment_id":%s}' "$IMPUTATION" "$EXPERIMENT_ID")

echo "Running casmsocial with Imputation=$IMPUTATION, experiment_id=$EXPERIMENT_ID, yaml='$YAML_PATH'"

mpiexec -n 1 python -m casmsocial "$YAML_PATH" "$JSON_ARG"

env > slurm_casmsocial_environment.txt

echo "Job completed."
