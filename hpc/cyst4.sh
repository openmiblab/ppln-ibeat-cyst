#!/bin/bash   
#SBATCH --mem=16G         
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=s.sourbron@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=cyst4
#SBATCH --output=logs/cyst4.out
#SBATCH --error=logs/cyst4.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2025.06-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/ibeat_cyst"
CODE="/mnt/parscratch/users/$(whoami)/scripts/ppln-ibeat-cyst/src/ibeat_cyst"
LOCAL="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build"
ARCHIVE="loginb:/shared/abdominal_imaging/Archive/iBEAt_Build"

# srun runs your program on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/stage_4_measurements.py" --build="$LOCAL"
rsync -av --no-group --no-perms "$LOCAL/cyst/stage_4_measurements" "$ARCHIVE/cyst"