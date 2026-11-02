#!/bin/bash
#SBATCH --job-name=queue_test
#SBATCH --output=logs/queue_test_%j.out
#SBATCH --error=logs/queue_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Runs submit_test.sh once best-epoch selection has finished, so the test array
# (and its summary) are submitted without you being there.
# Queue with:  sbatch --dependency=afterok:<select_best_jobid> queue_test.sh

set -euo pipefail

bash submit_test.sh