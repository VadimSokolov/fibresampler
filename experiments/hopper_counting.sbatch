#!/bin/bash
#SBATCH --job-name=count_est
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/counting_%j.log

# Tier C counting / rare-event estimators, cross-checked against exact enumeration.
cd "$SLURM_SUBMIT_DIR"
# interpreter with numpy and scipy (override with PYTHON=...; on GMU Hopper this was the miniconda3 python3)
PY="${PYTHON:-python3}"
echo "host $(hostname)  start $(date)  numpy $($PY -c 'import numpy;print(numpy.__version__)')"
$PY -u experiments/counting_estimators.py
echo "done $(date)"
