#!/bin/bash
#SBATCH --job-name=q1_sh16
#SBATCH --partition=normal
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/q1_sh16_%A_%a.out

# Change of basis versus reflection on the real Auckland SH16 corridor (Martin Hazelton's question).
# Array index: seed = idx % 5 + 1, target = uniform (idx < 5) or poiswc (idx >= 5).
cd "$SLURM_SUBMIT_DIR"
# interpreter with numpy and scipy (override with PYTHON=...; on GMU Hopper this was the miniconda3 python3)
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
IDX=$SLURM_ARRAY_TASK_ID
SEED=$(( IDX % 5 + 1 ))
if [ "$IDX" -lt 5 ]; then TARGET=uniform; else TARGET=poiswc; fi
mkdir -p results/q1_sh16
$PY -u experiments/refQ1_sh16_basis.py --target $TARGET --seed $SEED \
    --out results/q1_sh16/${TARGET}_seed${SEED}.json
