#!/bin/bash
#SBATCH --job-name=test_norm
#SBATCH --output=logs/test_norm_%A_%a.out
#SBATCH --error=logs/test_norm_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# One array task per line of test_list.txt: "<cell> <seed> <epoch> <gen_norm> <checkpoint>".
# Submit with submit_test.sh, not directly.

set -euo pipefail

CODE_DIR=/users/sabitijoel/workspace/translation/bce/normalization_tests/test/testing
GEN_MODULE=p2p
OUT_ROOT=$CODE_DIR/results

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate
cd "$CODE_DIR"

read -r CELL SEED EPOCH GEN_NORM CKPT < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" test_list.txt)
echo "Task $SLURM_ARRAY_TASK_ID | $CELL | seed_$SEED | epoch $EPOCH | generator norm: $GEN_NORM | $(hostname)"

python -u test_checkpoint.py "$CKPT" \
    --gen-norm "$GEN_NORM" \
    --generator-module "$GEN_MODULE" \
    --model-name "$CELL" \
    --dataset-name ISIC_TEST \
    --output-dir "$OUT_ROOT/$CELL/seed_$SEED"