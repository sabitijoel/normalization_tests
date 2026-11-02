#!/bin/bash
# Lists every norm_study/<cell>/seed_<N> folder and submits one validation task per folder.
# Usage:  bash submit_validation.sh            (every epoch)
#         EPOCH_STRIDE=5 bash submit_validation.sh   (every 5th epoch + the last)

set -euo pipefail

CODE_DIR=/users/sabitijoel/workspace/translation/bce/normalization_tests
CKPT_ROOT=/cbio/projects/037/jsabiti/datasets/ISIC/checkpoints/norm_study
MAX_PARALLEL=4

cd "$CODE_DIR"
mkdir -p logs

find "$CKPT_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed_*' | sort > norm_study_folders.txt
N=$(wc -l < norm_study_folders.txt)
if [ "$N" -eq 0 ]; then
    echo "No checkpoint folders found under $CKPT_ROOT"; exit 1
fi

echo "Found $N checkpoint folders:"
nl -v0 norm_study_folders.txt

sbatch --array=0-$((N - 1))%${MAX_PARALLEL} \
       --export=ALL,EPOCH_STRIDE=${EPOCH_STRIDE:-1} \
       validate_norm_study.sh