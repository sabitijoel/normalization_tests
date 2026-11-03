#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one selected norm-study generator checkpoint on test and OOD datasets.

Same metric contract as validation.py (and the corrected grayscale baseline):
metric definitions, Inception model, preprocessing, image pairing, resizing and
aggregation all come from checkpoint_evaluation_core / comparable_metrics.

Differences from the single-model evaluator:
  --gen-norm / --generator-module  build Generator(norm_kind) with the norm the
                                   checkpoint was trained with
  --model-name                     written to the 'model' column (e.g. the cell)
  --datasets                       evaluate several registered datasets in one
                                   run (ISIC test, PH2, SD-198 by default); the
                                   metric models load once, each dataset gets
                                   its own CSV row
  --gray-dir / --rgb-dir           evaluate a single custom dataset instead
  epoch parsed from '..._epoch_<N>' so the seed in the name is not mistaken for it

Every dataset starts from the same RNG state (the state right after the metric
models load), so its numbers do not depend on which other datasets ran first
and ISIC_TEST matches a single-dataset run of the previous version.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import argparse
import csv
import importlib
import random
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from checkpoint_evaluation_core import (
    configure_reproducibility,
    evaluate_checkpoint,
    extract_epoch_from_checkpoint,
    resolve_device,
)
from comparable_metrics import DeepMetricModels, MetricResult, pair_images


PROJECT_ROOT = Path("/cbio/projects/037/jsabiti")
ISIC_TEST_ROOT = PROJECT_ROOT / "datasets" / "ISIC" / "ISIC2018_Task3_Test_Input"
OOD_ROOT = PROJECT_ROOT / "other" / "validation_dataset"

# dataset name -> (gray_dir, rgb_dir); the name goes in the 'dataset' column.
# PH2 and SD-198 images are .bmp.
DATASETS: Dict[str, Tuple[Path, Path]] = {
    "ISIC_TEST": (ISIC_TEST_ROOT / "gray", ISIC_TEST_ROOT / "rgb"),
    "PH2": (OOD_ROOT / "gray" / "PH2Dataset", OOD_ROOT / "rgb" / "PH2Dataset"),
    "SD-198": (OOD_ROOT / "gray" / "SD-198", OOD_ROOT / "rgb" / "SD-198"),
}

DEFAULT_OUTPUT_DIR = Path("/users/sabitijoel/workspace/translation/bce/normalization_tests/test/testing")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
NORM_CHOICES = ("bn", "in", "ln", "gn", "none")
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

CSV_COLUMNS = (
    "model",
    "dataset",
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


def make_generator_factory(module_name: str, gen_norm: str):
    generator_cls = importlib.import_module(module_name).Generator

    def factory():
        return generator_cls(gen_norm)

    return factory


def epoch_of(checkpoint: Path) -> int:
    match = re.search(r"epoch[_-]?(\d+)", checkpoint.stem)
    if match:
        return int(match.group(1))
    return extract_epoch_from_checkpoint(checkpoint.stem)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def selected_datasets(args: argparse.Namespace) -> List[Tuple[str, Path, Path]]:
    if args.gray_dir is not None:
        return [(args.dataset_name or "CUSTOM", args.gray_dir, args.rgb_dir)]
    # dict.fromkeys drops repeats while keeping the requested order
    return [(name, *DATASETS[name]) for name in dict.fromkeys(args.datasets)]


def count_images(directory: Path) -> Counter:
    return Counter(
        path.suffix.lower()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_manifest(name: str, gray_dir: Path, rgb_dir: Path):
    """Pair images for one dataset, failing loudly if .bmp files are skipped."""
    for directory in (gray_dir, rgb_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"[{name}] directory not found: {directory}")

    on_disk = count_images(gray_dir)
    n_on_disk = sum(on_disk.values())
    found = ", ".join(f"{n} {ext}" for ext, n in sorted(on_disk.items())) or "no images"

    manifest = pair_images(gray_dir, rgb_dir)
    if not manifest:
        raise RuntimeError(
            f"[{name}] no matched grayscale/RGB pairs ({found} under {gray_dir}). "
            "Check that pair_images in comparable_metrics accepts the .bmp suffix "
            "and that grayscale and RGB file names match."
        )
    if len(manifest) < n_on_disk:
        print(
            f"WARNING [{name}]: {len(manifest)} pairs from {n_on_disk} grayscale "
            f"images ({found}); the rest are unmatched, nested in subfolders, or "
            "filtered out by pair_images."
        )
    return manifest


# --------------------------------------------------------------------------- #
# RNG snapshot so every dataset starts from the same state
# --------------------------------------------------------------------------- #
def capture_rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state) -> None:
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def result_row(
    checkpoint: Path,
    model_name: str,
    dataset_name: str,
    result: MetricResult,
) -> Dict[str, object]:
    values = result.as_dict()
    row: Dict[str, object] = {
        "model": model_name,
        "dataset": dataset_name,
        "checkpoint": checkpoint.stem,
        "epoch": epoch_of(checkpoint),
        "n_images": result.n_images,
    }
    for name in CSV_COLUMNS[5:]:
        row[name] = f"{float(values[name]):.10g}"
    return row


def append_csv(row: Dict[str, object], output_path: Path) -> None:
    """Append or replace the same checkpoint/dataset row atomically."""
    existing: list[Dict[str, str]] = []
    if output_path.is_file():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    key = (str(row["dataset"]), str(row["checkpoint"]))
    retained = [
        item for item in existing
        if (item.get("dataset", ""), item.get("checkpoint", "")) != key
    ]
    retained.append({column: str(row[column]) for column in CSV_COLUMNS})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(output_path)


def print_result(checkpoint: Path, dataset_name: str, result: MetricResult) -> None:
    print("\n" + "=" * 74)
    print(f"Dataset    : {dataset_name}")
    print(f"Checkpoint : {checkpoint.name}")
    print(f"Epoch      : {epoch_of(checkpoint)}")
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


def print_summary(
    results: Dict[str, MetricResult],
    failures: Dict[str, str],
) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'dataset':<10} {'n':>6} {'Y-PSNR':>9} {'Y-SSIM':>8} {'UV-SSIM':>8} "
          f"{'dE00':>8} {'LPIPS':>8} {'KID':>8} {'FID':>9}")
    for name, r in results.items():
        print(f"{name:<10} {r.n_images:>6d} {r.psnr:>9.4f} {r.ssim:>8.4f} {r.uv_ssim:>8.4f} "
              f"{r.delta_e:>8.4f} {r.lpips:>8.4f} {r.kid:>8.5f} {r.fid:>9.3f}")
    for name, reason in failures.items():
        print(f"{name:<10} FAILED: {reason}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one norm-study checkpoint with the comparable metric contract."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the generator checkpoint.")
    parser.add_argument("--gen-norm", choices=NORM_CHOICES, required=True,
                        help="normalisation the generator was trained with")
    parser.add_argument("--generator-module", default="p2p",
                        help="module defining Generator(norm_kind)")
    parser.add_argument("--model-name", default="GENERATOR",
                        help="value for the 'model' column, e.g. the cell name")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS),
                        default=list(DATASETS),
                        help="registered datasets to evaluate (default: all)")
    parser.add_argument("--gray-dir", type=Path, default=None,
                        help="custom grayscale dir; with --rgb-dir, evaluates only this dataset")
    parser.add_argument("--rgb-dir", type=Path, default=None,
                        help="custom RGB dir; requires --gray-dir")
    parser.add_argument("--dataset-name", default=None,
                        help="'dataset' column value for --gray-dir/--rgb-dir (default CUSTOM)")
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
    if (args.gray_dir is None) != (args.rgb_dir is None):
        parser.error("--gray-dir and --rgb-dir must be given together.")
    if args.dataset_name is not None and args.gray_dir is None:
        parser.error("--dataset-name only applies with --gray-dir/--rgb-dir; "
                     "registered datasets use their own names.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    configure_reproducibility(args.seed)
    device = resolve_device(args.device)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Pair every dataset before loading any model, so a bad path or an
    # unsupported extension fails in seconds rather than after ISIC finishes.
    datasets = selected_datasets(args)
    manifests = {name: build_manifest(name, gray, rgb) for name, gray, rgb in datasets}

    print("=" * 78)
    print("COMPARABLE SINGLE-CHECKPOINT EVALUATION")
    print(f"Model             : {args.model_name}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Generator norm    : {args.gen_norm}  (from {args.generator_module}.Generator)")
    print("Datasets          :")
    for name, gray_dir, _ in datasets:
        print(f"  {name:<10} {len(manifests[name]):>6} pairs  ({gray_dir})")
    print(f"Image size        : {args.image_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Device            : {device}")
    print("Metric contract   : Y-PSNR/Y-SSIM, zero-centred UV, shared LPIPS/Inception")
    print("=" * 78)

    print("\nLoading LPIPS-VGG and torchvision Inception-v3...")
    deep_models = DeepMetricModels(device)
    rng_state = capture_rng_state()

    generator_factory = make_generator_factory(args.generator_module, args.gen_norm)
    csv_path = args.output_dir / "checkpoint_evaluation_metrics.csv"
    results: Dict[str, MetricResult] = {}
    failures: Dict[str, str] = {}

    for name, gray_dir, rgb_dir in datasets:
        restore_rng_state(rng_state)
        print(f"\n>>> Evaluating {name} ({len(manifests[name])} pairs)")
        try:
            result = evaluate_checkpoint(
                args.checkpoint,
                generator_factory=generator_factory,
                gray_dir=gray_dir,
                rgb_dir=rgb_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
                deep_models=deep_models,
                strict_checkpoint=not args.non_strict_checkpoint,
                pairs=manifests[name],
            )
        except Exception as error:  # keep going so the other datasets still get rows
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"\nERROR [{name}]: {failures[name]}", file=sys.stderr)
            traceback.print_exc()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        print_result(args.checkpoint, name, result)
        append_csv(result_row(args.checkpoint, args.model_name, name, result), csv_path)
        results[name] = result

    print_summary(results, failures)
    if results:
        print(f"\nCSV saved to: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())