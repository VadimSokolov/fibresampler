#!/bin/bash
#SBATCH --job-name=q1_sh16_load
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/q1_sh16_loading_%A_%a.out

# Hold-fixed recombination versus the PLB on the real SH16 loading regime (unfloored max-entropy Poisson).
# Array index = seed - 1.
cd "$SLURM_SUBMIT_DIR"
# interpreter with numpy and scipy (override with PYTHON=...; on GMU Hopper this was the miniconda3 python3)
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SEED=$(( SLURM_ARRAY_TASK_ID + 1 ))
mkdir -p results/q1_sh16_loading
$PY -u experiments/refQ1_sh16_loading.py --seed $SEED --out results/q1_sh16_loading/seed${SEED}.json
