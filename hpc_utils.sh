#!/bin/bash
# HPC Utility Functions for CASMSOCIAL Heat Risk Model Job Arrays
#
# This script provides helper functions for managing SLURM job arrays
# for the CASMSOCIAL heat risk model.
#
# Usage:
#   source hpc_utils.sh
#   submit_heat_risk_array [CONFIG_FILE] [OUTPUT_DIR]
#   check_array_status <JOB_ID>
#   get_failed_jobs <JOB_ID>
#   retry_failed_jobs <JOB_ID>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG="config/enhanced_heat_risk_example.yaml"
DEFAULT_OUTPUT_DIR="hpc_outputs"
LOGS_DIR="logs"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_success() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $*"
}

# ============================================================================
# Job Submission Functions
# ============================================================================

submit_heat_risk_array() {
    """
    Submit CASMSOCIAL heat risk model as SLURM job array.

    Usage:
        submit_heat_risk_array [CONFIG_FILE] [OUTPUT_DIR]

    Examples:
        submit_heat_risk_array
        submit_heat_risk_array config/custom.yaml
        submit_heat_risk_array config/custom.yaml /scratch/output
    """

    local config="${1:-$DEFAULT_CONFIG}"
    local output_dir="${2:-$DEFAULT_OUTPUT_DIR}"

    if [ ! -f "$config" ]; then
        log_error "Configuration file not found: $config"
        return 1
    fi

    log_info "Submitting job array with config: $config"
    log_info "Output directory: $output_dir"

    mkdir -p "$LOGS_DIR" "$output_dir"

    # Submit the job array
    local job_id=$(CONFIG_FILE="$config" OUTPUT_DIR="$output_dir" \
        sbatch submit_heat_risk_array.slurm 2>&1 | awk '{print $4}')

    if [ -z "$job_id" ]; then
        log_error "Failed to submit job array"
        return 1
    fi

    log_success "Job array submitted with ID: $job_id"
    log_info "To check status: squeue --job $job_id"
    log_info "To view logs: tail logs/heat_risk_*.log"

    echo "$job_id"
}

# ============================================================================
# Job Monitoring Functions
# ============================================================================

check_array_status() {
    """
    Check the status of a job array.

    Usage:
        check_array_status <JOB_ID>
    """

    local job_id="$1"

    if [ -z "$job_id" ]; then
        log_error "Job ID required"
        return 1
    fi

    log_info "Status for job array: $job_id"
    echo ""

    # Get job info
    scontrol show job "$job_id" | head -20

    echo ""
    log_info "Array task summary:"
    squeue --job "$job_id" --array --format="JobID,Name,State,ExitCode" | head -20

    echo ""
    # Count tasks by state
    local total=$(squeue --job "$job_id" --array --noheader | wc -l)
    local running=$(squeue --job "$job_id" --array --noheader --state=RUNNING | wc -l)
    local pending=$(squeue --job "$job_id" --array --noheader --state=PENDING | wc -l)
    local failed=$(squeue --job "$job_id" --array --noheader --state=FAILED | wc -l)
    local completed=$(squeue --job "$job_id" --array --noheader --state=COMPLETED | wc -l)

    log_info "Summary:"
    log_info "  Total tasks: $total"
    log_info "  Running: $running"
    log_info "  Pending: $pending"
    log_info "  Completed: $completed"
    log_info "  Failed: $failed"
}

# ============================================================================
# Job Analysis Functions
# ============================================================================

get_failed_jobs() {
    """
    List all failed jobs in an array.

    Usage:
        get_failed_jobs <JOB_ID>
    """

    local job_id="$1"

    if [ -z "$job_id" ]; then
        log_error "Job ID required"
        return 1
    fi

    log_info "Failed jobs for array: $job_id"
    echo ""

    local failed_jobs=$(squeue --job "$job_id" --array --state=FAILED --noheader -o '%i')

    if [ -z "$failed_jobs" ]; then
        log_success "No failed jobs found"
        return 0
    fi

    echo "Failed task IDs:"
    echo "$failed_jobs"

    echo ""
    log_info "To view logs for a failed task:"
    echo "  tail logs/heat_risk_<TASK_ID>.log"

    echo ""
    log_info "To resubmit failed tasks:"
    echo "  scontrol requeue $(echo $failed_jobs | tr '\n' ',')"
}

count_completed_outputs() {
    """
    Count completed output directories.

    Usage:
        count_completed_outputs [OUTPUT_DIR]
    """

    local output_dir="${1:-$DEFAULT_OUTPUT_DIR}"

    if [ ! -d "$output_dir" ]; then
        log_error "Output directory not found: $output_dir"
        return 1
    fi

    log_info "Counting completed outputs in: $output_dir"
    echo ""

    local total=$(find "$output_dir" -name ".completed" | wc -l)
    local imputation_count=$(find "$output_dir" -type d -name "imputation_*" | wc -l)
    local experiment_count=$(find "$output_dir" -type d -name "experiment_*" | wc -l)

    log_info "Completed runs: $total / 750"
    log_info "Imputations with output: $imputation_count"
    log_info "Total experiment directories: $experiment_count"

    # Show progress percentage
    if [ "$total" -gt 0 ]; then
        local percentage=$((total * 100 / 750))
        log_info "Progress: $percentage%"
    fi
}

# ============================================================================
# Log Analysis Functions
# ============================================================================

analyze_logs() {
    """
    Analyze SLURM job logs for errors and warnings.

    Usage:
        analyze_logs [LOG_DIR]
    """

    local log_dir="${1:-$LOGS_DIR}"

    if [ ! -d "$log_dir" ]; then
        log_error "Log directory not found: $log_dir"
        return 1
    fi

    log_info "Analyzing logs in: $log_dir"
    echo ""

    # Count log files
    local total_logs=$(ls "$log_dir"/heat_risk_*.log 2>/dev/null | wc -l)
    log_info "Total log files: $total_logs"

    # Find errors
    local error_count=$(grep -h "ERROR\|ERROR:" "$log_dir"/heat_risk_*.log 2>/dev/null | wc -l)
    log_info "Total ERROR messages: $error_count"

    # Find warnings
    local warning_count=$(grep -h "WARNING\|WARN:" "$log_dir"/heat_risk_*.log 2>/dev/null | wc -l)
    log_info "Total WARNING messages: $warning_count"

    # Find successful completions
    local success_count=$(grep -h "completed successfully" "$log_dir"/heat_risk_*.log 2>/dev/null | wc -l)
    log_info "Successful completions: $success_count"

    echo ""
    log_info "Common errors:"
    grep -h "ERROR\|ERROR:" "$log_dir"/heat_risk_*.log 2>/dev/null | \
        sort | uniq -c | sort -rn | head -10 || log_info "No errors found"
}

# ============================================================================
# Utility Functions
# ============================================================================

show_help() {
    """
    Display help information.

    Usage:
        show_help
    """

    cat <<EOF
CASMSOCIAL HPC Utilities

Available functions:

Job Submission:
  submit_heat_risk_array [CONFIG] [OUTPUT_DIR]
    Submit job array for heat risk model

    Examples:
      submit_heat_risk_array
      submit_heat_risk_array config/custom.yaml
      submit_heat_risk_array config/custom.yaml /scratch/output

Job Monitoring:
  check_array_status <JOB_ID>
    Check status of a job array

    Example:
      check_array_status 12345

Job Analysis:
  get_failed_jobs <JOB_ID>
    List all failed jobs in an array

    Example:
      get_failed_jobs 12345

  count_completed_outputs [OUTPUT_DIR]
    Count completed output directories

    Example:
      count_completed_outputs hpc_outputs

  analyze_logs [LOG_DIR]
    Analyze logs for errors and warnings

    Example:
      analyze_logs logs

Other:
  show_help
    Display this help message

Usage:
  source hpc_utils.sh
  submit_heat_risk_array
  check_array_status \$JOB_ID

EOF
}

# ============================================================================
# Export functions for use in shell
# ============================================================================

# Export all functions so they can be used when this script is sourced
export -f submit_heat_risk_array
export -f check_array_status
export -f get_failed_jobs
export -f count_completed_outputs
export -f analyze_logs
export -f show_help
export -f log_info
export -f log_error
export -f log_success

# If script is executed directly, show help
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    show_help
fi
