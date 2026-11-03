#!/bin/bash
# Build the test list, submit one evaluation per cell/seed,
# and queue the summary job after the evaluation array.
# Each evaluation task tests its checkpoint on every test dataset
# registered in test_checkpoint.py (ISIC_TEST, PH2, SD-198).

set -euo pipefail

BASE="/users/sabitijoel/workspace/translation/bce/normalization_tests"
CODE_DIR="$BASE/test/testing"
CKPT_ROOT="/cbio/projects/037/jsabiti/datasets/ISIC/checkpoints/norm_study"
SELECTION="$BASE/test/selecting_best_epoch/BALANCED_LR/best_epoch_per_run.csv"
TEST_SCRIPT="$CODE_DIR/test_norm_study.sh"
TEST_PY="$CODE_DIR/test_checkpoint.py"
SUMMARY_SCRIPT="$CODE_DIR/summarise_test.sh"
SUMMARY_PY="$CODE_DIR/summarise_test.py"
TEST_LIST="$CODE_DIR/test_list.txt"
MAX_PARALLEL=4

cd "$CODE_DIR"
mkdir -p logs

module load python/3.11
source /users/sabitijoel/workspace/myenv/bin/activate

# Check required files and directories.
for file in "$SELECTION" "$TEST_SCRIPT" "$TEST_PY" "$SUMMARY_SCRIPT" "$SUMMARY_PY"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file not found: $file" >&2
        exit 1
    fi
done

if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_ROOT" >&2
    exit 1
fi

# Pair every test dataset with the same code the array tasks use, so a missing
# folder or .bmp files that pair_images skips stop the submission here instead
# of failing in every task.
echo "Checking test datasets:"
python - <<'PY'
import test_checkpoint as tc

total = 0
for name, (gray_dir, rgb_dir) in tc.DATASETS.items():
    pairs = tc.build_manifest(name, gray_dir, rgb_dir)
    total += len(pairs)
    print(f"  {name:<10} {len(pairs):>6} image pairs")
print(f"  {'total':<10} {total:>6} image pairs per task")
PY

# Generate test_list.txt.
# Format: cell seed epoch gen_norm checkpoint_path
python - "$SELECTION" "$CKPT_ROOT" > "$TEST_LIST" <<'PY'
import sys
from pathlib import Path

import pandas as pd

selection = Path(sys.argv[1])
ckpt_root = Path(sys.argv[2])

runs = pd.read_csv(selection)

required = {"cell", "seed", "best_epoch"}
missing_columns = required - set(runs.columns)

if missing_columns:
    sys.exit(f"Missing CSV columns: {sorted(missing_columns)}")

if runs.empty:
    sys.exit(f"Selection CSV contains no rows: {selection}")

entries = []
errors = []

valid_norms = {"bn", "in", "ln", "gn", "none"}

for row in runs.itertuples(index=False):
    cell = str(row.cell).strip()

    # Derive generator normalization from the cell name.
    # Examples: gbn_dbn -> bn; ggn_dnone_sn -> gn.
    if not cell.startswith("g") or "_d" not in cell:
        sys.exit(f"Cannot parse cell name: {cell}")

    gen_norm = cell[1:].split("_d", 1)[0]

    if gen_norm not in valid_norms:
        sys.exit(
            f"Invalid generator normalization '{gen_norm}' "
            f"for cell '{cell}'"
        )

    try:
        seed = int(row.seed)
        epoch = int(row.best_epoch)
    except (TypeError, ValueError):
        sys.exit(f"Invalid seed or best_epoch for cell '{cell}'")

    folder = ckpt_root / cell / f"seed_{seed}"
    matches = sorted(folder.glob(f"*_epoch_{epoch}.pth"))

    if len(matches) != 1:
        errors.append(
            f"{cell}/seed_{seed}, epoch {epoch}: "
            f"expected one checkpoint, found {len(matches)}"
        )
        continue

    entries.append(
        f"{cell} {seed} {epoch} {gen_norm} {matches[0]}"
    )

if errors:
    sys.exit("Checkpoint validation failed:\n  " + "\n  ".join(errors))

print("\n".join(entries))
PY

# Stop if list generation failed or no entries were produced.
N=$(wc -l < "$TEST_LIST")

if (( N == 0 )); then
    echo "ERROR: No test evaluations found; no jobs submitted." >&2
    exit 1
fi

# Verify that the generator normalization column contains no NaNs.
if awk '$4 !~ /^(bn|in|ln|gn|none)$/ { bad=1; print "Invalid entry:", $0 > "/dev/stderr" } END { exit bad ? 0 : 1 }' "$TEST_LIST"; then
    echo "ERROR: Invalid generator normalization in test_list.txt." >&2
    exit 1
fi

echo "Test evaluations to run: $N"
echo "Generated test list:"
nl -v0 "$TEST_LIST"

# Submit the test array, with at most four tasks running concurrently.
ARRAY_ID=$(sbatch --parsable \
    --array="0-$((N - 1))%${MAX_PARALLEL}" \
    "$TEST_SCRIPT")
ARRAY_ID=${ARRAY_ID%%;*}

if [[ ! "$ARRAY_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid array job ID: $ARRAY_ID" >&2
    exit 1
fi

echo "Submitted test array: $ARRAY_ID"

# Run the summary after all array tasks finish, even if some fail.
SUMMARY_ID=$(sbatch --parsable \
    --dependency="afterany:${ARRAY_ID}" \
    "$SUMMARY_SCRIPT")
SUMMARY_ID=${SUMMARY_ID%%;*}

if [[ ! "$SUMMARY_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid summary job ID: $SUMMARY_ID" >&2
    exit 1
fi

echo "Queued summary job: $SUMMARY_ID"
echo "The summary will run after test array $ARRAY_ID finishes."