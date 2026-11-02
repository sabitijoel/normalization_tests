#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd

from select_best_epoch_csv import REQUIRED, select_best_epoch

DEFAULT_ROOT = Path("/users/sabitijoel/workspace/translation/bce/normalization_tests/validation_norm_test/norm_study")
CSV_NAME = "validation_checkpoint_metrics.csv"
CELL_RE = re.compile(r"^g(bn|in|ln|gn|none)_d(bn|in|ln|gn|none)(?:_bs(\d+))?$")

# Metrics carried into the summary (whichever the CSVs contain), and whether higher is better
METRICS = {
    "delta_e": False, "uv_ssim": True, "uv_psnr": True,
    "psnr": True, "ssim": True, "rmse": False, "mae": False,
    "lpips": False, "kid": False, "fid": False,
}
NORM_ORDER = ["bn", "in", "ln", "gn", "none"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=None, help="default: --root")
    p.add_argument("--expected-epochs", type=int, default=None,
                   help="warn about runs whose CSV has fewer epochs than this")
    p.add_argument("--tol-de", type=float, default=0.1)
    p.add_argument("--tol-ssim", type=float, default=0.002)
    p.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.3, 0.2),
                   metavar=("W_DE", "W_SSIM", "W_PSNR"))
    # parse_known_args so %run in Jupyter doesn't choke on kernel arguments
    return p.parse_known_args()[0]


def main():
    args = parse_args()
    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(args.root.glob(f"*/seed_*/{CSV_NAME}"))
    if not csvs:
        raise SystemExit(f"No {CSV_NAME} found under {args.root}/<cell>/seed_*/")
    print(f"Found {len(csvs)} validation CSVs under {args.root}\n")

    rows, problems = [], []
    for csv_path in csvs:
        cell = csv_path.parent.parent.name
        seed = int(csv_path.parent.name.split("_", 1)[1])
        m = CELL_RE.match(cell)
        gen_norm, disc_norm, batch = (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)

        try:
            best, df, top = select_best_epoch(
                str(csv_path), tol_de=args.tol_de, tol_ssim=args.tol_ssim,
                weights=tuple(args.weights), verbose=False)
        except (ValueError, KeyError) as e:
            problems.append(f"{cell}/seed_{seed}: {e}")
            continue

        rank_pick = top.index[0]
        row = {
            "cell": cell, "gen_norm": gen_norm, "disc_norm": disc_norm,
            "batch_size": int(batch) if batch else 8, "seed": seed,
            "best_epoch": best, "rank_pick": rank_pick, "rules_agree": best == rank_pick,
            "n_epochs": len(df), "last_epoch": int(df.index.max()),
        }
        for metric in METRICS:
            if metric in df.columns:
                row[metric] = float(df.loc[best, metric])
        rows.append(row)

        if args.expected_epochs and len(df) < args.expected_epochs:
            problems.append(f"{cell}/seed_{seed}: only {len(df)} epochs "
                            f"(expected {args.expected_epochs}) — CSV may be incomplete")

    if not rows:
        raise SystemExit("No CSV could be processed:\n  " + "\n  ".join(problems))

    per_run = pd.DataFrame(rows).sort_values(["cell", "seed"]).reset_index(drop=True)
    per_run.to_csv(out_dir / "best_epoch_per_run.csv", index=False)

    metric_cols = [m for m in METRICS if m in per_run.columns]
    per_cell = (per_run.groupby(["cell", "gen_norm", "disc_norm", "batch_size"])
                .agg(n_seeds=("seed", "nunique"),
                     best_epochs=("best_epoch", lambda s: ",".join(str(int(x)) for x in s)),
                     **{f"{m}_mean": (m, "mean") for m in metric_cols},
                     **{f"{m}_std": (m, "std") for m in metric_cols})
                .reset_index()
                .sort_values("delta_e_mean"))
    per_cell.to_csv(out_dir / "best_epoch_per_cell.csv", index=False)

    # ---- Print summary ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("Per run (selected epoch and its metrics):")
    print(per_run[["cell", "seed", "best_epoch", "rank_pick", "n_epochs"] + REQUIRED]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPer cell, mean ± std over seeds (sorted by Delta E, lower is better):")
    for _, r in per_cell.iterrows():
        parts = [f"{m}={r[f'{m}_mean']:.4f}±{(r[f'{m}_std'] if pd.notna(r[f'{m}_std']) else 0):.4f}"
                 for m in REQUIRED]
        print(f"  {r['cell']:<14} seeds={r['n_seeds']}  epochs=[{r['best_epochs']}]  " + "  ".join(parts))

    # Generator x discriminator grids (default batch size only, so cells line up)
    grid_src = per_run[per_run["batch_size"] == 8].dropna(subset=["gen_norm"])
    for metric in REQUIRED:
        pivot = grid_src.pivot_table(index="gen_norm", columns="disc_norm", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=[n for n in NORM_ORDER if n in pivot.index],
                              columns=[n for n in NORM_ORDER if n in pivot.columns])
        pivot.index.name, pivot.columns.name = "generator \\ discriminator", None
        pivot.to_csv(out_dir / f"pivot_{metric}.csv")
        direction = "higher" if METRICS[metric] else "lower"
        print(f"\n{metric} (seed mean, {direction} is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    disagree = int((~per_run["rules_agree"]).sum())
    print(f"\nLexicographic and rank-weighted rules disagree in {disagree}/{len(per_run)} runs "
          f"(lexicographic pick used, as in select_best_epoch).")
    if problems:
        print("\nWarnings:")
        for msg in problems:
            print(f"  - {msg}")
    print(f"\nSaved: {out_dir / 'best_epoch_per_run.csv'}, {out_dir / 'best_epoch_per_cell.csv'}, "
          f"pivot_<metric>.csv")


if __name__ == "__main__":
    main()