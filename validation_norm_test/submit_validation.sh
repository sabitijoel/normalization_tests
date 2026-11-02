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

# Usage:  bash validate_norm_study.sh
#   Run directly (no SLURM_ARRAY_TASK_ID): counts the folders in the list
#   and resubmits this same script as an array job, max 4 tasks at once.
#   Run by Slurm as an array task: validates one checkpoint folder.

set -euo pipefail

SCRIPT_DIR=/users/sabitijoel/workspace/translation/bce/normalization_tests/validation_norm_test
LIST=/users/sabitijoel/workspace/translation/bce/normalization_tests/norm_study_folders.txt
MAX_CONCURRENT=4

# ---- Submission step ----
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    cd "$SCRIPT_DIR"
    mkdir -p logs
    N=$(grep -c . "$LIST")
    if (( N == 0 )); then
        echo "No folders in $LIST" >&2
        exit 1
    fi
    echo "Submitting $N validation tasks (max $MAX_CONCURRENT concurrent)"
    sbatch --array=0-$((N - 1))%$MAX_CONCURRENT "$(readlink -f "$0")"
    exit 0
fi

# ---- Task step (runs under Slurm) ----
CODE_DIR=$SCRIPT_DIR
GEN_MODULE=p2p
OUT_ROOT=/users/sabitijoel/workspace/translation/bce/normalization_tests/validation_norm_test
EPOCH_STRIDE=${EPOCH_STRIDE:-1}

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate
cd "$CODE_DIR"

FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
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