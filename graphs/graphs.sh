#!/bin/bash
# Copies every graph (.png) saved by the norm-study training runs into one folder.
# Source:  <CKPT_ROOT>/<cell>/seed_<N>/*.png   (e.g. gbn_dbn_seed42.png)
# Target:  <DEST>/<cell>_seed<N>.png           (flat, one file per run)
#
# Usage:  bash collect_graphs.sh
# Safe to re-run: existing copies are overwritten with the latest version.

set -euo pipefail

CKPT_ROOT=/cbio/projects/037/jsabiti/datasets/ISIC/checkpoints/norm_study
DEST=/users/sabitijoel/workspace/translation/bce/normalization_tests/graphs/all_plots

mkdir -p "$DEST"

count=0
while IFS= read -r -d '' png; do
    seed_dir=$(basename "$(dirname "$png")")                # seed_42
    cell=$(basename "$(dirname "$(dirname "$png")")")      # gbn_dbn
    name=$(basename "$png")

    # Training names plots <cell>_seed<N>.png; keep that. Anything else gets
    # the cell and seed prefixed so files from different runs never collide.
    if [[ "$name" == "${cell}_seed${seed_dir#seed_}.png" ]]; then
        target="$DEST/$name"
    else
        target="$DEST/${cell}_${seed_dir}_${name}"
    fi

    cp -p "$png" "$target"
    count=$((count + 1))
done < <(find "$CKPT_ROOT" -mindepth 3 -maxdepth 3 -type f -name '*.png' -print0 | sort -z)

echo "Copied $count graphs to $DEST"
ls "$DEST" | head -5
if [ "$count" -gt 5 ]; then echo "..."; fi