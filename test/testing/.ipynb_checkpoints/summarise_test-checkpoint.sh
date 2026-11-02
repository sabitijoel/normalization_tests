#!/bin/bash
#SBATCH --job-name=summarise_test
#SBATCH --output=/users/sabitijoel/workspace/translation/bce/normalization_tests/test/testing/logs/summarise_test_%j.out
#SBATCH --error=/users/sabitijoel/workspace/translation/bce/normalization_tests/test/testing/logs/summarise_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

BASE=/users/sabitijoel/workspace/translation/bce/normalization_tests
CODE_DIR="$BASE/test/testing/"
TEST_ROOT="$CODE_DIR"

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate

mkdir -p "$CODE_DIR/logs"
cd "$CODE_DIR"

python -u "$CODE_DIR/summarise_test.py" --test-root "$TEST_ROOT"
