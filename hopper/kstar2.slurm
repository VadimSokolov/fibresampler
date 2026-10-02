#!/bin/bash
#SBATCH --job-name=kstar2
#SBATCH --partition=normal
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/kstar2_%A_%a.out
cd "$SLURM_SUBMIT_DIR"
/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin/python3 experiments/hopper_kstar2.py $SLURM_ARRAY_TASK_ID 4000
