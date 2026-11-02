#!/bin/bash
#SBATCH --job-name=p2p_combined
#SBATCH --array=0-59%7
#SBATCH --output=logs/p2p_%A_%a.out
#SBATCH --error=logs/p2p_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --nodelist=gpu-[001-007]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00

# Task layout (60 runs total):
#   0-47  : norm grid, 4 gen norms x 4 disc norms x 3 seeds
#   48-59 : spectral norm, 4 gen norms x disc=none (+ --disc-sn) x 3 seeds
#
# Usage (create the log directory before submitting):
#   mkdir -p logs
#   sbatch run_combined.sbatch        -> full set, 200 epochs
#   sbatch run_combined.sbatch 2      -> full set, 2 epochs (smoke test)
#
# %7 caps concurrency at 7 jobs. Adjust to your queue.

set -euo pipefail

EPOCHS=${1:-200}
BATCH_SIZE=1
CODE_DIR=/users/sabitijoel/workspace/translation/bce/normalization_tests/train_norm_test

NORMS=(bn in ln gn)
SEEDS=(42 1337 2024)
NN=${#NORMS[@]}
NS=${#SEEDS[@]}
GRID_RUNS=$(( NN * NN * NS ))   # 48
SN_RUNS=$(( NN * NS ))          # 12

IDX=$SLURM_ARRAY_TASK_ID
EXTRA_ARGS=()

if (( IDX < GRID_RUNS )); then
    SEED=${SEEDS[$(( IDX % NS ))]}
    CELL=$(( IDX / NS ))
    DISC_NORM=${NORMS[$(( CELL % NN ))]}
    GEN_NORM=${NORMS[$(( CELL / NN ))]}
    MODE="grid"
else
    J=$(( IDX - GRID_RUNS ))
    SEED=${SEEDS[$(( J % NS ))]}
    GEN_NORM=${NORMS[$(( J / NS ))]}
    DISC_NORM=none
    EXTRA_ARGS=(--disc-sn --save-every 1)
    MODE="spectral-norm"
fi

echo "Task $IDX | mode=$MODE | gen=$GEN_NORM | disc=$DISC_NORM | seed=$SEED | epochs=$EPOCHS | $(hostname)"

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate
cd "$CODE_DIR"

python -u p2p.py \
    --seed "$SEED" \
    --gen-norm "$GEN_NORM" \
    --disc-norm "$DISC_NORM" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}