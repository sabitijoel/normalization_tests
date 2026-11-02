#!/bin/bash
#SBATCH --job-name=select_best
#SBATCH --output=logs/select_best_%j.out
#SBATCH --error=logs/select_best_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Best-epoch selection for every norm-study validation CSV (CPU only, no GPU).
#
# Run directly:                 bash select_all_runs.sh
# Or as a job after validation: sbatch --dependency=afterany:<validation_array_jobid> select_all_runs.sh
#
# EXPECTED_EPOCHS = epochs each CSV should contain (200 for every epoch of a
# 200-epoch run; 40 if validated with EPOCH_STRIDE=5). Override when calling:
#   EXPECTED_EPOCHS=40 bash select_all_runs.sh

set -euo pipefail


module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate



python -u select_all_runs.py \
    --out-dir "/users/sabitijoel/workspace/translation/bce/normalization_tests/test/selecting_best_epoch" \
