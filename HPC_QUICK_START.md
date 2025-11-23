# HPC Job Array Quick Start Guide

## One-Minute Summary

Submit 750 heat risk model jobs (30 imputations × 25 experiments) to your HPC cluster:

```bash
sbatch submit_heat_risk_array.slurm
```

That's it! The script will:
- Map array task IDs to Imputation (1-30) and experiment_id (1-25) combinations
- Override YAML config parameters using JSON
- Organize outputs by imputation and experiment
- Track completion with `.completed` marker files
- Log all output to individual job logs

## Prerequisites

1. **HPC Cluster with SLURM**: sbatch, squeue, scontrol commands available
2. **Environment Variable**: `CASMSOCIAL_DATA_PATH` must be set
3. **Configuration File**: Default is `config/enhanced_heat_risk_example.yaml`

## Basic Commands

### Submit Default Job Array (750 jobs)

```bash
sbatch submit_heat_risk_array.slurm
```

### Submit with Custom Configuration

```bash
CONFIG_FILE=config/custom.yaml sbatch submit_heat_risk_array.slurm
```

### Submit with Custom Output Directory

```bash
OUTPUT_DIR=/scratch/user/results sbatch submit_heat_risk_array.slurm
```

### Both Custom Config and Output

```bash
CONFIG_FILE=config/custom.yaml OUTPUT_DIR=/scratch/results sbatch submit_heat_risk_array.slurm
```

## Monitoring Jobs

### Check Status

```bash
# Replace <JOB_ID> with actual job ID from submission
squeue --job <JOB_ID>

# View more details
scontrol show job <JOB_ID>
```

### Watch in Real-Time

```bash
watch -n 5 'squeue --job <JOB_ID>'
```

### Check Specific Task

```bash
squeue --job <JOB_ID>_<TASK_ID>
```

## Viewing Logs

```bash
# View specific job log
tail logs/heat_risk_1.log

# Monitor in real-time
tail -f logs/heat_risk_1.log

# Check for errors
grep ERROR logs/heat_risk_*.log

# Count completed jobs
ls logs/heat_risk_*.log | wc -l
```

## Using Utility Functions

```bash
# Source the utility script
source hpc_utils.sh

# Submit job array (returns job ID)
JOB_ID=$(submit_heat_risk_array)

# Check status
check_array_status $JOB_ID

# Get failed jobs
get_failed_jobs $JOB_ID

# Count completed outputs
count_completed_outputs hpc_outputs

# Analyze logs
analyze_logs logs
```

## Output Structure

Outputs are automatically organized:

```
hpc_outputs/
├── imputation_1/
│   ├── experiment_1/
│   │   ├── agent_log.parquet
│   │   ├── run_log.parquet
│   │   └── .completed
│   ├── experiment_2/
│   ├── ...
│   └── experiment_25/
├── imputation_2/
│   └── experiment_1/
│   └── ...
└── imputation_30/
    └── experiment_25/
```

Each directory contains the outputs for that imputation-experiment pair.

## Adjusting Job Array Size

### Run Smaller Subset

Edit `submit_heat_risk_array.slurm` and change:

```bash
#SBATCH --array=1-750%50
```

to a smaller range, e.g.:

```bash
#SBATCH --array=1-250%50    # First 10 imputations only
#SBATCH --array=1-125%50    # First 5 imputations only
```

### Adjust Concurrent Job Limit

Change the `%50` to a different number:

```bash
#SBATCH --array=1-750%25    # Run 25 jobs concurrently (more conservative)
#SBATCH --array=1-750%100   # Run 100 jobs concurrently (if cluster allows)
```

## Understanding the Mapping

Each job's array task ID maps to parameters:

```
Task ID 1  → Imputation=1, experiment_id=1
Task ID 2  → Imputation=1, experiment_id=2
...
Task ID 25 → Imputation=1, experiment_id=25
Task ID 26 → Imputation=2, experiment_id=1
Task ID 50 → Imputation=2, experiment_id=25
...
Task ID 750 → Imputation=30, experiment_id=25
```

Formula:
- **Imputation** = (Task ID - 1) ÷ 25 + 1
- **experiment_id** = (Task ID - 1) mod 25 + 1

## Troubleshooting

### Job Fails Immediately

1. Check error log: `cat logs/heat_risk_1.err`
2. Verify `CASMSOCIAL_DATA_PATH` is set: `echo $CASMSOCIAL_DATA_PATH`
3. Check config file exists: `ls config/enhanced_heat_risk_example.yaml`

### Out of Memory

Increase in `submit_heat_risk_array.slurm`:
```bash
#SBATCH --mem=128GB    # Increase from 64GB to 128GB
```

### Job Timeout

Increase in `submit_heat_risk_array.slurm`:
```bash
#SBATCH --time=04:00:00    # Increase from 2:00:00 to 4:00:00
```

### Job Array Too Large

Reduce concurrent limit:
```bash
#SBATCH --array=1-750%25   # Run only 25 concurrent jobs
```

## Advanced: Resubmit Failed Jobs

```bash
# Get list of failed tasks
source hpc_utils.sh
get_failed_jobs <JOB_ID>

# Resubmit specific failed tasks
sbatch --array=<FAILED_TASK_IDS> submit_heat_risk_array.slurm
```

## Performance Notes

- **Expected runtime**: 200-300 seconds per job with EnhancedHeatRiskModel
- **Total estimated time**: ~750 jobs × 250 seconds = ~52 hours of compute
- **With 50 concurrent jobs**: ~1 hour wall-clock time
- **Memory usage**: ~64GB per job (adjustable based on data size)

## Configuration Parameters

Key parameters in `enhanced_heat_risk_example.yaml`:

```yaml
Imputation: 1                    # Overridden by job array
experiment_id: 1                 # Overridden by job array
heat_threshold_cooling_center: 90
heat_threshold_health_effect: 100
duration.hours: 24               # Simulation duration
random.seed: 42                  # For reproducibility
```

## Next Steps

1. **First run**: Submit a small test with `--array=1-10`
2. **Monitor**: Use `check_array_status` to track progress
3. **Full run**: Once confident, submit full `--array=1-750`
4. **Analysis**: Post-process results from `hpc_outputs/`

## See Also

- **Full Guide**: `HPC_SUBMISSION_GUIDE.md`
- **SLURM Script**: `submit_heat_risk_array.slurm`
- **Utilities**: `hpc_utils.sh`
- **Config**: `config/enhanced_heat_risk_example.yaml`

## Support

For issues:
1. Check logs: `tail logs/heat_risk_<TASK_ID>.err`
2. Run diagnostics: `source hpc_utils.sh && analyze_logs`
3. Review `HPC_SUBMISSION_GUIDE.md` troubleshooting section
