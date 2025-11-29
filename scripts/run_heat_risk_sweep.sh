#!/bin/bash
# Submit multiple Slurm jobs over parameter sweeps of Imputation (1..30) and experiment_id (1..25)
# Optionally pass a YAML parameter file to the job script via --yaml-file

set -euo pipefail

# Path to the original Slurm job script that accepts --imputation, --experiment-id, and optionally --yaml-file
JOB_SCRIPT="script.sh"  # change to the full path if needed, e.g., /projects/ARTSOC-POLICY-MODELING/script.sh

# Optional controls (can be overridden via CLI)
I_START="${I_START:-1}"
I_END="${I_END:-30}"
E_START="${E_START:-1}"
E_END="${E_END:-25}"
DRY_RUN="${DRY_RUN:-0}"        # set to 1 to print commands without submitting
MAX_SUBMIT_RATE="${MAX_SUBMIT_RATE:-0}"  # seconds to sleep between submissions
YAML_PATH="${YAML_PATH:-}"     # optional; if empty, job script's default is used

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-script)
      JOB_SCRIPT="$2"; shift 2;;
    --imputation-start)
      I_START="$2"; shift 2;;
    --imputation-end)
      I_END="$2"; shift 2;;
    --experiment-start|--experiment-id-start)
      E_START="$2"; shift 2;;
    --experiment-end|--experiment-id-end)
      E_END="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --sleep|--rate-limit)
      MAX_SUBMIT_RATE="$2"; shift 2;;
    --yaml|--yaml-file|--config|--config-file)
      YAML_PATH="$2"; shift 2;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

# Validate job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Error: job script not found: $JOB_SCRIPT"
  exit 1
fi

# Validate YAML path if provided
if [[ -n "${YAML_PATH}" && ! -f "${YAML_PATH}" ]]; then
  echo "Error: YAML parameter file not found: ${YAML_PATH}"
  exit 1
fi

# Validate ranges
for v in "$I_START" "$I_END" "$E_START" "$E_END"; do
  if ! [[ "$v" =~ ^[0-9]+$ ]]; then
    echo "Error: range values must be integers (got '$v')"; exit 1
  fi
done
if (( I_START < 1 || I_END < I_START )); then
  echo "Error: invalid imputation range: $I_START..$I_END"; exit 1
fi
if (( E_START < 1 || E_END < E_START )); then
  echo "Error: invalid experiment_id range: $E_START..$E_END"; exit 1
fi

declare -a JOB_IDS=()

for (( i=I_START; i<=I_END; i++ )); do
  for (( e=E_START; e<=E_END; e++ )); do
    JOB_NAME="casmsocial_i${i}_e${e}"
    CMD=(sbatch --job-name="$JOB_NAME" "$JOB_SCRIPT" --imputation "$i" --experiment-id "$e")
    if [[ -n "$YAML_PATH" ]]; then
      CMD+=("--yaml-file" "$YAML_PATH")
    fi
    echo "Submitting: ${CMD[*]}"
    if (( DRY_RUN == 0 )); then
      OUTPUT="$("${CMD[@]}")"
      JOB_ID=$(awk '/Submitted batch job/ {print $4}' <<<"$OUTPUT")
      if [[ -n "$JOB_ID" ]]; then
        JOB_IDS+=("$JOB_ID")
        if [[ -n "$YAML_PATH" ]]; then
          echo "Submitted job $JOB_ID for Imputation=$i, experiment_id=$e, yaml=$YAML_PATH"
        else
          echo "Submitted job $JOB_ID for Imputation=$i, experiment_id=$e"
        fi
      else
        echo "Warning: could not parse job ID from sbatch output: $OUTPUT"
      fi
      if (( MAX_SUBMIT_RATE > 0 )); then
        sleep "$MAX_SUBMIT_RATE"
      fi
    fi
  done
done

if (( DRY_RUN == 0 )); then
  echo "Total jobs submitted: ${#JOB_IDS[@]}"
  printf "%s\n" "${JOB_IDS[@]}" > submitted_job_ids.txt
  echo "Job IDs saved to submitted_job_ids.txt"
else
  echo "Dry run complete. No jobs submitted."
fi
