#!/bin/bash
#SBATCH --job-name=rim3way
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/rim3way_%j.out
cd "$SLURM_SUBMIT_DIR"
/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin/python3 experiments/ref_3x3x2_rim.py
