#!/usr/bin/env python3
"""
Collect the test-set results of every norm-study run and summarise them.

Reads  <test_root>/<cell>/seed_<N>/checkpoint_evaluation_metrics.csv
(one row each: the validation-selected checkpoint evaluated on the test set)
and writes to <test_root>:

  test_per_run.csv     one row per cell x seed (epoch + all test metrics)
  test_per_cell.csv    mean and std over seeds per cell, sorted by Delta E
  test_pivot_<m>.csv   generator-norm x discriminator-norm grids of the seed mean

Usage:  python summarise_test.py [--test-root ...] [--dataset ISIC_TEST]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--dataset", default="ISIC_TEST")
    args = p.parse_known_args()[0]

    files = sorted(
        args.test_root.glob("results/*/seed_*/checkpoint_evaluation_metrics.csv")
    )
    if not files:
        raise SystemExit(f"No test CSVs under {args.test_root}/<cell>/seed_*/")

    rows = []
    for f in files:
        cell, seed = f.parent.parent.name, int(f.parent.name.split("_", 1)[1])
        df = pd.read_csv(f)
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            continue
        r = df.iloc[-1]
        m = CELL_RE.match(cell)
        row = {"cell": cell, "gen_norm": m.group(1) if m else None,
               "disc_norm": m.group(2) if m else None,
               "batch_size": int(m.group(3)) if m and m.group(3) else 8,
               "seed": seed, "epoch": int(r["epoch"]), "n_images": int(r["n_images"])}
        row.update({k: float(r[k]) for k in METRICS if k in r})
        rows.append(row)

    per_run = pd.DataFrame(rows).sort_values(["cell", "seed"]).reset_index(drop=True)
    per_run.to_csv(args.test_root / "test_per_run.csv", index=False)

    metric_cols = [k for k in METRICS if k in per_run.columns]
    per_cell = (per_run.groupby(["cell", "gen_norm", "disc_norm", "batch_size"])
                .agg(n_seeds=("seed", "nunique"),
                     **{f"{k}_mean": (k, "mean") for k in metric_cols},
                     **{f"{k}_std": (k, "std") for k in metric_cols})
                .reset_index().sort_values("delta_e_mean"))
    per_cell.to_csv(args.test_root / "test_per_cell.csv", index=False)

    print(f"{len(per_run)} runs, {per_cell.shape[0]} cells, dataset {args.dataset} "
          f"({per_run['n_images'].iloc[0]} images)\n")
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
        pivot.to_csv(args.test_root / f"test_pivot_{k}.csv")
        print(f"\n{k} on test (seed mean, {'higher' if METRICS[k] else 'lower'} is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    incomplete = per_cell[per_cell["n_seeds"] < per_cell["n_seeds"].max()]
    if not incomplete.empty:
        print("\nCells missing seeds: " + ", ".join(incomplete["cell"]))
    print(f"\nSaved to {args.test_root}: test_per_run.csv, test_per_cell.csv, test_pivot_<metric>.csv")


if __name__ == "__main__":
    main()