#!/usr/bin/env python3
"""
Collect the test and OOD results of every norm-study run and summarise them.

Reads  <test_root>/results/<cell>/seed_<N>/checkpoint_evaluation_metrics.csv
(one row per dataset: the validation-selected checkpoint evaluated on
ISIC_TEST, PH2 and SD-198) and writes to <test_root>, for each dataset <D>:

  test_per_run_<D>.csv     one row per cell x seed (epoch + all metrics)
  test_per_cell_<D>.csv    mean and std over seeds per cell, sorted by Delta E
  test_pivot_<m>_<D>.csv   generator-norm x discriminator-norm grids of the seed mean

and, when more than one dataset is summarised:

  test_cross_dataset_<m>.csv   cell x dataset table of the seed mean,
                               sorted by the first dataset (ISIC_TEST)

Usage:  python summarise_test.py [--test-root ...] [--datasets ISIC_TEST PH2 SD-198]
        (default: every dataset found in the CSVs)
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("/users/sabitijoel/workspace/translation/bce/normalization_tests/test/testing")

CELL_RE = re.compile(
    r"^g(bn|in|ln|gn|none)_d(bn|in|ln|gn|none)(?:_sn)?(?:_bs(\d+))?$"
)

METRICS = {  # name: higher is better
    "delta_e": False, "uv_ssim": True, "uv_psnr": True, "psnr": True, "ssim": True,
    "rmse": False, "mae": False, "lpips": False, "kid": False, "fid": False,
}
GRID_METRICS = ["delta_e", "uv_ssim", "uv_psnr"]
NORM_ORDER = ["bn", "in", "ln", "gn", "none"]
DATASET_ORDER = ["ISIC_TEST", "PH2", "SD-198"]  # any other dataset follows, alphabetically


def load_runs(test_root: Path) -> pd.DataFrame:
    """One row per cell x seed x dataset."""
    files = sorted(test_root.glob("results/*/seed_*/checkpoint_evaluation_metrics.csv"))
    if not files:
        raise SystemExit(f"No test CSVs under {test_root}/results/<cell>/seed_*/")

    rows, stale = [], []
    for f in files:
        cell, seed = f.parent.parent.name, int(f.parent.name.split("_", 1)[1])
        df = pd.read_csv(f)
        if df.empty:
            continue
        # The last row written belongs to the run's current checkpoint. Rows left
        # from an earlier best-epoch selection are ignored, so every dataset of a
        # run comes from the same checkpoint and a failed dataset shows up as
        # missing instead of silently reusing an old result.
        current = df["checkpoint"].iloc[-1]
        if (df["checkpoint"] != current).any():
            stale.append(f"{cell}/seed_{seed}")
        df = df[df["checkpoint"] == current]

        m = CELL_RE.match(cell)
        for dataset, group in df.groupby("dataset", sort=False):
            r = group.iloc[-1]
            row = {"dataset": dataset, "cell": cell,
                   "gen_norm": m.group(1) if m else None,
                   "disc_norm": m.group(2) if m else None,
                   "batch_size": int(m.group(3)) if m and m.group(3) else 8,
                   "seed": seed, "epoch": int(r["epoch"]), "n_images": int(r["n_images"])}
            row.update({k: float(r[k]) for k in METRICS if k in r})
            rows.append(row)

    if stale:
        print("Ignored rows from older checkpoints in: " + ", ".join(stale) + "\n")
    return pd.DataFrame(rows)


def summarise_dataset(per_run: pd.DataFrame, dataset: str, test_root: Path) -> pd.DataFrame:
    per_run = per_run.sort_values(["cell", "seed"]).reset_index(drop=True)
    per_run.to_csv(test_root / f"test_per_run_{dataset}.csv", index=False)

    metric_cols = [k for k in METRICS if k in per_run.columns]
    per_cell = (per_run.groupby(["cell", "gen_norm", "disc_norm", "batch_size"])
                .agg(n_seeds=("seed", "nunique"),
                     **{f"{k}_mean": (k, "mean") for k in metric_cols},
                     **{f"{k}_std": (k, "std") for k in metric_cols})
                .reset_index().sort_values("delta_e_mean"))
    per_cell.insert(0, "dataset", dataset)
    per_cell.to_csv(test_root / f"test_per_cell_{dataset}.csv", index=False)

    sizes = sorted(per_run["n_images"].unique())
    size_note = (f"{sizes[0]} images" if len(sizes) == 1
                 else f"WARNING: image counts differ between runs {sizes}")
    print("=" * 78)
    print(f"{dataset}: {len(per_run)} runs, {per_cell.shape[0]} cells, {size_note}")
    print("=" * 78)
    print("Per cell, mean ± std over seeds (sorted by Delta E, lower is better):")
    for _, r in per_cell.iterrows():
        parts = [f"{k}={r[f'{k}_mean']:.4f}±{(r[f'{k}_std'] if pd.notna(r[f'{k}_std']) else 0):.4f}"
                 for k in GRID_METRICS + [k for k in ("lpips", "fid") if k in metric_cols]]
        print(f"  {r['cell']:<14} seeds={r['n_seeds']}  " + "  ".join(parts))

    grid_src = per_run[per_run["batch_size"] == 8].dropna(subset=["gen_norm"])
    for k in GRID_METRICS:
        pivot = grid_src.pivot_table(index="gen_norm", columns="disc_norm", values=k, aggfunc="mean")
        pivot = pivot.reindex(index=[n for n in NORM_ORDER if n in pivot.index],
                              columns=[n for n in NORM_ORDER if n in pivot.columns])
        pivot.index.name, pivot.columns.name = "generator \\ discriminator", None
        pivot.to_csv(test_root / f"test_pivot_{k}_{dataset}.csv")
        print(f"\n{k} on {dataset} (seed mean, {'higher' if METRICS[k] else 'lower'} is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    incomplete = per_cell[per_cell["n_seeds"] < per_cell["n_seeds"].max()]
    if not incomplete.empty:
        print(f"\nCells missing seeds on {dataset}: " + ", ".join(incomplete["cell"]))
    print()
    return per_cell


def cross_dataset_tables(per_cell_all: pd.DataFrame, datasets: list, test_root: Path) -> None:
    """cell x dataset tables of the seed mean, sorted by the first dataset."""
    reference = datasets[0]
    print("=" * 78)
    print(f"ACROSS DATASETS (seed mean per cell, sorted by {reference})")
    print("=" * 78)
    for k in GRID_METRICS:
        table = per_cell_all.pivot_table(index="cell", columns="dataset",
                                         values=f"{k}_mean", aggfunc="first")
        table = table.reindex(columns=[d for d in datasets if d in table.columns])
        if reference in table.columns:
            table = table.sort_values(reference, ascending=not METRICS[k])
        table.columns.name = None
        table.to_csv(test_root / f"test_cross_dataset_{k}.csv")
        print(f"\n{k} ({'higher' if METRICS[k] else 'lower'} is better):")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--datasets", "--dataset", nargs="+", default=None,
                   help="datasets to summarise (default: every dataset found)")
    args = p.parse_known_args()[0]

    runs = load_runs(args.test_root)
    if runs.empty:
        raise SystemExit(f"Every test CSV under {args.test_root}/results/ is empty.")

    found = list(runs["dataset"].unique())
    if args.datasets:
        missing = [d for d in args.datasets if d not in found]
        if missing:
            print(f"No rows for: {', '.join(missing)} (found: {', '.join(found)})\n")
        datasets = [d for d in dict.fromkeys(args.datasets) if d in found]
    else:
        rank = {d: i for i, d in enumerate(DATASET_ORDER)}
        datasets = sorted(found, key=lambda d: (rank.get(d, len(rank)), d))
    if not datasets:
        raise SystemExit("No rows for the requested datasets.")

    per_cells = [summarise_dataset(runs[runs["dataset"] == d], d, args.test_root)
                 for d in datasets]
    written = "test_per_run_<D>.csv, test_per_cell_<D>.csv, test_pivot_<metric>_<D>.csv"
    if len(datasets) > 1:
        cross_dataset_tables(pd.concat(per_cells, ignore_index=True), datasets, args.test_root)
        written += ", test_cross_dataset_<metric>.csv"
    print(f"\nSaved to {args.test_root} for {', '.join(datasets)}: {written}")


if __name__ == "__main__":
    main()