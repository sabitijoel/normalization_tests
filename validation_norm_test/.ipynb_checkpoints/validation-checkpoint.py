#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import importlib
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from checkpoint_evaluation_core import (
    configure_reproducibility,
    evaluate_checkpoint,
    extract_epoch_from_checkpoint,
    resolve_device,
)
from comparable_metrics import DeepMetricModels, MetricResult, pair_images

DEFAULT_CHECKPOINT_FOLDER = Path(
    "/cbio/projects/037/jsabiti/datasets/ISIC/"
    "ISIC2018_Task3_Validation_Input/vanilla/BN_BN"
)
DEFAULT_OUTPUT_DIR = Path(
    "/users/sabitijoel/workspace/translation/bce/normalization_tests/vanilla"
)
DEFAULT_GRAY_DIR = Path(
    "/cbio/projects/037/jsabiti/datasets/ISIC/"
    "ISIC2018_Task3_Validation_Input/gray"
)
DEFAULT_RGB_DIR = Path(
    "/cbio/projects/037/jsabiti/datasets/ISIC/"
    "ISIC2018_Task3_Validation_Input/rgb"
)

DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16

METRIC_DIRECTIONS = {
    "psnr": "max",
    "ssim": "max",
    "uv_psnr": "max",
    "uv_ssim": "max",
    "rmse": "min",
    "mae": "min",
    "delta_e": "min",
    "lpips": "min",
    "kid": "min",
    "fid": "min",
}

CSV_COLUMNS = (
    "checkpoint",
    "epoch",
    "n_images",
    "psnr",
    "ssim",
    "uv_psnr",
    "uv_ssim",
    "u_ssim",
    "v_ssim",
    "rmse",
    "mae",
    "delta_e",
    "lpips",
    "kid",
    "fid",
)


NORM_CHOICES = ("bn", "in", "ln", "gn", "none")


def make_generator_factory(module_name: str, gen_norm: str):
    """Build the generator with the SAME normalisation it was trained with.

    The module must define Generator(norm_kind) -- i.e. the norm-study training
    script. A checkpoint loaded into a generator with a different norm would
    either fail the strict key check or silently compute the wrong function.
    """
    module = importlib.import_module(module_name)
    generator_cls = module.Generator

    def factory():
        return generator_cls(gen_norm)

    return factory


def epoch_of(checkpoint: Path) -> int:
    """Epoch from '..._epoch_<N>' names; falls back to the shared helper.

    Norm-study names also contain the seed (e.g. gbn_dbn_seed42_epoch_5), so a
    generic 'first number' parser could return the seed instead of the epoch.
    """
    match = re.search(r"epoch[_-]?(\d+)", checkpoint.stem)
    if match:
        return int(match.group(1))
    return extract_epoch_from_checkpoint(checkpoint.stem)


def checkpoint_files(folder: Path) -> List[Path]:
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.name.lower().endswith((".pth", ".pth.tar", ".pt"))
    ]
    # Numeric epoch order (plain sorting would put epoch_10 before epoch_2)
    return sorted(files, key=lambda path: (epoch_of(path), path.name))


def result_row(checkpoint: Path, epoch: int, result: MetricResult) -> Dict[str, object]:
    values = result.as_dict()
    row: Dict[str, object] = {
        "checkpoint": checkpoint.stem,
        "epoch": epoch,
        "n_images": result.n_images,
    }
    for name in CSV_COLUMNS[3:]:
        row[name] = f"{float(values[name]):.10g}"
    return row


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def print_result(checkpoint: Path, epoch: int, result: MetricResult) -> None:
    print("\n" + "=" * 74)
    print(f"Checkpoint : {checkpoint.name}")
    print(f"Epoch      : {epoch}")
    print(f"Images     : {result.n_images}")
    print("=" * 74)
    print(f"Y-PSNR     : {result.psnr:.6f}")
    print(f"Y-SSIM     : {result.ssim:.6f}")
    print(f"UV-PSNR    : {result.uv_psnr:.6f}")
    print(f"UV-SSIM    : {result.uv_ssim:.6f}")
    print(f"  U-SSIM   : {result.u_ssim:.6f}")
    print(f"  V-SSIM   : {result.v_ssim:.6f}")
    print(f"RGB-RMSE   : {result.rmse:.6f}")
    print(f"RGB-MAE    : {result.mae:.6f}")
    print(f"CIEDE2000  : {result.delta_e:.6f}")
    print(f"LPIPS-VGG  : {result.lpips:.6f}")
    print(f"KID        : {result.kid:.6f}")
    print(f"FID        : {result.fid:.6f}")


def best_checkpoints(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    best: Dict[str, Dict[str, object]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        candidates = []
        for row in rows:
            value = float(row[metric])
            if np.isfinite(value):
                candidates.append((value, row))
        if not candidates:
            continue
        selected = max(candidates, key=lambda item: item[0]) if direction == "max" else min(
            candidates, key=lambda item: item[0]
        )
        best[metric] = {
            "value": selected[0],
            "checkpoint": selected[1]["checkpoint"],
            "epoch": selected[1]["epoch"],
        }
    return best


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints with metrics comparable to the corrected baseline."
    )
    parser.add_argument("--checkpoint-folder", type=Path, default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument("--gen-norm", choices=NORM_CHOICES, default="bn",
                        help="normalisation the generator was trained with")
    parser.add_argument("--generator-module", default="p2p",
                        help="module that defines Generator(norm_kind), e.g. the norm-study training script")
    parser.add_argument("--epoch-stride", type=int, default=1,
                        help="evaluate every N-th epoch (the final epoch is always included)")
    parser.add_argument("--gray-dir", type=Path, default=DEFAULT_GRAY_DIR)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Permit missing/unexpected checkpoint keys, but print them.",
    )
    args = parser.parse_args(argv)

    if args.image_size < 11:
        parser.error("--image-size must be at least 11.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.epoch_stride < 1:
        parser.error("--epoch-stride must be at least 1.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    configure_reproducibility(args.seed)
    device = resolve_device(args.device)

    if not args.checkpoint_folder.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {args.checkpoint_folder}")

    checkpoints = checkpoint_files(args.checkpoint_folder)
    if not checkpoints:
        raise RuntimeError(f"No .pth/.pth.tar/.pt checkpoints found in {args.checkpoint_folder}")
    if args.epoch_stride > 1:
        last = checkpoints[-1]
        checkpoints = [c for c in checkpoints if epoch_of(c) % args.epoch_stride == 0]
        if last not in checkpoints:
            checkpoints.append(last)

    generator_factory = make_generator_factory(args.generator_module, args.gen_norm)

    manifest = pair_images(args.gray_dir, args.rgb_dir)
    if not manifest:
        raise RuntimeError("No matched grayscale/RGB validation pairs were found.")

    print("=" * 78)
    print("COMPARABLE CHECKPOINT VALIDATION")
    print(f"Checkpoint folder : {args.checkpoint_folder}")
    print(f"Generator norm    : {args.gen_norm}  (from {args.generator_module}.Generator)")
    print(f"Checkpoints       : {len(checkpoints)}")
    print(f"Image manifest    : {len(manifest)} matched pairs")
    print(f"Image size        : {args.image_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Device            : {device}")
    print("Metric contract   : Y-PSNR/Y-SSIM, zero-centred UV, shared LPIPS/Inception")
    print("=" * 78)

    print("\nLoading LPIPS-VGG and torchvision Inception-v3...")
    deep_models = DeepMetricModels(device)

    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        epoch = epoch_of(checkpoint)
        try:
            result = evaluate_checkpoint(
                checkpoint,
                generator_factory=generator_factory,
                gray_dir=args.gray_dir,
                rgb_dir=args.rgb_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
                deep_models=deep_models,
                strict_checkpoint=not args.non_strict_checkpoint,
                pairs=manifest,
            )
        except Exception as error:
            print(f"\n!! Checkpoint failed: {checkpoint.name}: {error}")
            continue

        print_result(checkpoint, epoch, result)
        rows.append(result_row(checkpoint, epoch, result))

        # Persist after every successful checkpoint so an interrupted long run
        # retains all completed results.
        write_csv(rows, args.output_dir / "validation_checkpoint_metrics.csv")

    if not rows:
        print("\nNo checkpoint was evaluated successfully.")
        return 1

    best = best_checkpoints(rows)
    print("\n" + "=" * 78)
    print("BEST CHECKPOINT BY METRIC")
    print("=" * 78)
    for metric in METRIC_DIRECTIONS:
        if metric not in best:
            continue
        item = best[metric]
        direction = "higher" if METRIC_DIRECTIONS[metric] == "max" else "lower"
        print(
            f"{metric.upper():10} ({direction}): {item['value']:.6f}  "
            f"epoch={item['epoch']}  checkpoint={item['checkpoint']}"
        )

    csv_path = args.output_dir / "validation_checkpoint_metrics.csv"
    print(f"\nCSV saved to: {csv_path}")
    print(
        "Compare these rows only with a grayscale baseline run on the same "
        "validation directories, image size, and matched-pair manifest."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())