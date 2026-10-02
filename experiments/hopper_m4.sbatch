#!/bin/bash
#SBATCH --job-name=m4_ess
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/m4_%j.log

# M4 real-data ESS/second benchmark on P2 (Auckland SH16), full 10^6-iteration grid.
cd "$SLURM_SUBMIT_DIR"
# interpreter with numpy and scipy (override with PYTHON=...; on GMU Hopper this was the miniconda3 python3)
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "host $(hostname)  start $(date)  numpy $($PY -c 'import numpy;print(numpy.__version__)')"
$PY -u experiments/m4_realdata_ess.py --iters 1000000 --burn 50000 --seeds 5
echo "done $(date)"
