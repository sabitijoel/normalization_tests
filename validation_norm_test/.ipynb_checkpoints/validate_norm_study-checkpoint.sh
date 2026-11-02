#!/bin/bash
#SBATCH --job-name=val_norm
#SBATCH --output=logs/val_norm_%A_%a.out
#SBATCH --error=logs/val_norm_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

# One array task per checkpoint folder listed in norm_study_folders.txt
# (norm_study/<cell>/seed_<N>). Submit with submit_validation.sh.

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?must be submitted as an array job via submit_validation.sh}"

# ---- Edit these two ----
CODE_DIR=/users/sabitijoel/workspace/translation/bce/normalization_tests
GEN_MODULE=p2p

OUT_ROOT=/users/sabitijoel/workspace/translation/bce/normalization_tests/norm_study
EPOCH_STRIDE=${EPOCH_STRIDE:-1}

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate
cd "$CODE_DIR"

FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" norm_study_folders.txt)
[[ -n "$FOLDER" ]] || { echo "No folder for task $SLURM_ARRAY_TASK_ID"; exit 1; }

CELL=$(basename "$(dirname "$FOLDER")")
SEED=$(basename "$FOLDER")
GEN_NORM=${CELL#g}
GEN_NORM=${GEN_NORM%%_d*}

echo "Task $SLURM_ARRAY_TASK_ID | $CELL | $SEED | generator norm: $GEN_NORM | $(hostname)"

python -u validation.py \
    --checkpoint-folder "$FOLDER" \
    --gen-norm "$GEN_NORM" \
    --generator-module "$GEN_MODULE" \
    --output-dir "$OUT_ROOT/$CELL/$SEED" \
    --epoch-stride "$EPOCH_STRIDE"