# HPC Job Array Submission Guide

This guide explains how to submit the CASMSOCIAL heat risk model to an HPC cluster using SLURM job arrays with multiple imputations and experiments.

## Overview

The `submit_heat_risk_array.slurm` script automates running the heat risk model across:
- **30 imputations** (Imputation=1 to 30)
- **25 experiments** (experiment_id=1 to 25)
- **Total: 750 jobs** submitted as a single array job

Each job runs independently with its own imputation and experiment configuration.

## Quick Start

### Basic Submission

```bash
sbatch submit_heat_risk_array.slurm
```

### Custom Configuration File

```bash
CONFIG_FILE=config/my_custom_config.yaml sbatch submit_heat_risk_array.slurm
```

### Custom Output Directory

```bash
OUTPUT_DIR=/scratch/my_user/heat_risk_output sbatch submit_heat_risk_array.slurm
```

### Combined Options

```bash
CONFIG_FILE=config/custom.yaml OUTPUT_DIR=/scratch/output sbatch submit_heat_risk_array.slurm
```

## Job Array Details

### Array Configuration

```
--array=1-750%50
```

- **Array range**: 1 to 750 (30 imputations × 25 experiments)
- **Concurrent job limit**: 50 (adjust `%50` as needed for your cluster's policies)

### Mapping Job Array Index to Parameters

Each job's array index (`SLURM_ARRAY_TASK_ID`) maps to a unique imputation and experiment combination:

```bash
ARRAY_INDEX = SLURM_ARRAY_TASK_ID - 1
IMPUTATION = ARRAY_INDEX / 25 + 1       # Range: 1-30
EXPERIMENT_ID = ARRAY_INDEX % 25 + 1    # Range: 1-25
```

**Example Mappings:**
- Job 1: Imputation=1, experiment_id=1
- Job 26: Imputation=2, experiment_id=1
- Job 50: Imputation=2, experiment_id=25
- Job 750: Imputation=30, experiment_id=25

## SLURM Configuration

### Resource Requirements

```
--ntasks=1              # Single MPI task per job
--cpus-per-task=16      # 16 CPU cores
--mem=64GB              # 64 GB memory
--time=02:00:00         # 2 hour time limit
```

Adjust these based on:
- Your cluster's available resources
- Your data size (1M agents uses ~64GB)
- Expected runtime for your configuration

### Output Logging

```
--output=logs/heat_risk_%a.log     # Standard output (one per job)
--error=logs/heat_risk_%a.err      # Standard error (one per job)
```

The `%a` placeholder expands to the array task ID, creating individual log files for each job.

## Parameter Override Format

The script uses repast4py's parameter override system. Parameters are passed as a JSON string:

```json
{
    "Imputation": 1,
    "experiment_id": 1
}
```

These values override the corresponding parameters in the YAML configuration file.

## Output Organization

By default, outputs are organized by imputation and experiment:

```
hpc_outputs/
├── imputation_1/
│   ├── experiment_1/
│   │   ├── agent_log.parquet
│   │   ├── run_log.parquet
│   │   └── .completed
│   ├── experiment_2/
│   └── ...
├── imputation_2/
│   └── ...
└── imputation_30/
    └── experiment_25/
```

Each subdirectory contains the outputs for that specific imputation-experiment combination.

## Monitoring Jobs

### Check Job Status

```bash
# View all jobs in the array
squeue --job <JOB_ID>

# View specific array task
squeue --array <JOB_ID>_1-100

# Watch updates
watch -n 5 'squeue --job <JOB_ID>'
```

### Check Job Details

```bash
# Show array job info
scontrol show job <JOB_ID>

# Get job array statistics
squeue --job <JOB_ID> --array
```

### View Logs

```bash
# Check a specific job's log
tail logs/heat_risk_1.log
tail logs/heat_risk_1.err

# Check multiple logs
ls -lh logs/heat_risk_*.log | wc -l

# Monitor in real-time
tail -f logs/heat_risk_<TASK_ID>.log
```

## Troubleshooting

### Job Fails Immediately

**Check:**
- `CASMSOCIAL_DATA_PATH` environment variable is set
- Configuration file path is correct
- Data files exist at the specified paths
- Sufficient disk space for output

**Debug:**
```bash
cat logs/heat_risk_1.log
cat logs/heat_risk_1.err
```

### Out of Memory

**Solution:**
- Increase `--mem` in the SLURM script
- Reduce data size or agent count
- Reduce `--cpus-per-task` if other jobs can run

### Timeout

**Solution:**
- Increase `--time` in the SLURM script
- Profile the model to identify bottlenecks
- Use the enhanced heat risk model for better performance

### Job Array Too Large

**Solution:**
- Reduce the max concurrent jobs: Change `%50` to a smaller number
- Submit multiple job arrays with ranges (e.g., `--array=1-375` twice)

## Advanced Usage

### Reduce Array Size (Subset of Parameters)

Edit `submit_heat_risk_array.slurm` and change the array configuration:

```bash
# Only run first 5 imputations (125 jobs)
#SBATCH --array=1-125%50

# Only run 10 experiments for all imputations (300 jobs)
# Requires changes to mapping logic
```

### Custom Parameter Combinations

Create multiple SLURM scripts with different parameter ranges if you need non-rectangular parameter spaces.

### Resume Failed Jobs

```bash
# Get list of failed jobs
squeue --job <JOB_ID> --array --state=FAILED

# Resubmit only failed tasks
scontrol requeue <FAILED_TASK_ID>

# Or manually submit specific tasks
sbatch --array=50,100,150 submit_heat_risk_array.slurm
```

## Performance Optimization

### Parallel Processing

The model supports parallel processing. The enhanced heat risk model provides significant speedups:
- Original: ~1457 seconds
- Enhanced: ~200-300 seconds
- Speedup: 5-10x

### Data Pre-staging

For better performance on large HPC systems:

```bash
# Pre-stage data to local scratch
cp /path/to/data /scratch/$USER/data
export CASMSOCIAL_DATA_PATH=/scratch/$USER/data

sbatch submit_heat_risk_array.slurm
```

### Job Dependency Chains

To process results after all jobs complete:

```bash
# Submit main array job
JOB_ID=$(sbatch submit_heat_risk_array.slurm | awk '{print $4}')

# Submit post-processing job that depends on array completion
sbatch --dependency=afterok:${JOB_ID} post_processing.slurm
```

## Example Commands

```bash
# Submit with defaults
sbatch submit_heat_risk_array.slurm

# Submit subset for testing
sbatch --array=1-100%20 submit_heat_risk_array.slurm

# Submit with custom paths
CASMSOCIAL_DATA_PATH=/data/heat_risk \
CONFIG_FILE=config/production.yaml \
OUTPUT_DIR=/scratch/results \
sbatch submit_heat_risk_array.slurm

# Check status while running
watch -n 10 'squeue --job $(squeue -u $USER -O jobid --noheader | head -1)'
```

## File Locations

- **SLURM Script**: `submit_heat_risk_array.slurm`
- **Configuration**: `config/enhanced_heat_risk_example.yaml`
- **Logs**: `logs/heat_risk_*.log` and `logs/heat_risk_*.err`
- **Outputs**: `hpc_outputs/imputation_*/experiment_*/`

## Support

For issues or questions:
1. Check job logs: `cat logs/heat_risk_<TASK_ID>.log`
2. Review SLURM error output: `cat logs/heat_risk_<TASK_ID>.err`
3. Check cluster-specific documentation
4. Review CASMSOCIAL documentation
