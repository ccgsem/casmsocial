#!/bin/bash
#SBATCH --job-name="casmsocial"
#SBATCH --nodes=1
#SBATCH --time=90:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%u-%x-job%j.out
#SBATCH --export=ALL
#SBATCH --mem=50GB

unset PYTHONPATH
unset LD_LIBRARY_PATH

module load ucx/default
module load python/3.12.9
module load mpich/4.0.2

source /projects/ARTSOC-POLICY-MODELING/python/casmsocial/bin/activate

export UCX_TLS=tcp
export CASMSOCIAL_DATA_PATH=/projects/ARTSOC-POLICY-MODELING/data
mpiexec -n 1 python -m casmsocial /projects/ARTSOC-POLICY-MODELING/config/casmsocial_wc_30_v2_hpc.yaml '{"Imputation":1,"experiment_id":1}'

env > slurm_casmsocial_environment.txt

echo "Job completed."

