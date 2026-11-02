#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared generator-checkpoint evaluation engine.

This module intentionally contains no dataset-specific defaults and no command-
line interface. Both validation.py and evaluation.py call the same evaluator,
which guarantees identical image loading, model-output conversion, metric
aggregation, LPIPS preprocessing, and Inception preprocessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from comparable_metrics import (
    DeepMetricModels,
    MetricAccumulator,
    MetricResult,
    pair_images,
)


@dataclass(frozen=True)
class LoadedPair:
    """One resized grayscale input and corresponding ground-truth RGB image."""

    key: str
    gray_tensor: torch.Tensor  # (1,H,W), normalised to [-1,1], on CPU
    real_rgb: np.ndarray       # (H,W,3), uint8


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def configure_reproducibility(seed: int = 0) -> None:
    """Use deterministic inference settings where PyTorch supports them."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def extract_epoch_from_checkpoint(checkpoint_name: str) -> int:
    patterns = (
        r"epoch_(\d+)",
        r"(\d+)_epoch",
        r"generator_(\d+)",
        r"checkpoint_(\d+)",
        r"epoch(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, checkpoint_name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    numbers = re.findall(r"\d+", checkpoint_name)
    return int(numbers[-1]) if numbers else -1


def _normalise_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip a DataParallel 'module.' prefix when present on every key."""
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract a generator state dict from common checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint).__name__}")

    candidate_keys = (
        "model_state_dict",
        "generator_state_dict",
        "gen_state_dict",
        "state_dict",
        "generator",
        "model",
    )
    for key in candidate_keys:
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate:
            if all(torch.is_tensor(value) for value in candidate.values()):
                return _normalise_state_dict_keys(candidate)

    # A raw state dict is itself a mapping from names to tensors.
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _normalise_state_dict_keys(checkpoint)  # type: ignore[arg-type]

    raise ValueError(
        "Could not find a generator state dict. Supported keys include "
        "model_state_dict, generator_state_dict, state_dict, generator, and model."
    )


def load_generator_checkpoint(
    generator: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> None:
    """Load one checkpoint into an existing generator instance."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    incompatible = generator.load_state_dict(state_dict, strict=strict)

    # In strict mode PyTorch raises before reaching here. In non-strict mode,
    # make missing/unexpected keys visible instead of silently accepting them.
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Warning: non-strict checkpoint load for {checkpoint_path.name}")
        if incompatible.missing_keys:
            print(f"  Missing keys   : {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {incompatible.unexpected_keys}")


def load_pair(
    gray_path: Path,
    rgb_path: Path,
    image_size: int,
) -> LoadedPair:
    """Load a pair using the same PIL bilinear resize as the baseline script."""
    target_size = (image_size, image_size)

    with Image.open(gray_path) as image:
        gray_image = image.convert("L").resize(
            target_size,
            resample=Image.Resampling.BILINEAR,
        )
        gray = np.asarray(gray_image, dtype=np.uint8)

    with Image.open(rgb_path) as image:
        real_image = image.convert("RGB").resize(
            target_size,
            resample=Image.Resampling.BILINEAR,
        )
        real = np.asarray(real_image, dtype=np.uint8)

    gray_tensor = torch.from_numpy(gray.copy()).unsqueeze(0).float() / 255.0
    gray_tensor = gray_tensor * 2.0 - 1.0

    return LoadedPair(
        key=gray_path.stem.casefold(),
        gray_tensor=gray_tensor,
        real_rgb=real,
    )


def generator_output_to_uint8(
    output: torch.Tensor,
    image_size: int,
) -> List[np.ndarray]:
    """Convert a BCHW generator output in [-1,1] to resized uint8 RGB arrays."""
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError(
            "Generator output must have shape (B, 3, H, W); "
            f"received {tuple(output.shape)}."
        )
    if not torch.isfinite(output).all():
        raise ValueError("Generator output contains NaN or infinite values.")

    unit = (output.detach().cpu().float() * 0.5 + 0.5).clamp(0.0, 1.0)
    arrays = (unit.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)

    results: List[np.ndarray] = []
    for array in arrays:
        if array.shape[:2] != (image_size, image_size):
            image = Image.fromarray(array).resize(
                (image_size, image_size),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.uint8)
        results.append(array)
    return results


def _chunks(sequence: Sequence[Tuple[Path, Path]], size: int):
    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]


def evaluate_loaded_generator(
    generator: nn.Module,
    *,
    pairs: Sequence[Tuple[Path, Path]],
    image_size: int,
    batch_size: int,
    device: torch.device,
    deep_models: DeepMetricModels,
    description: str,
) -> MetricResult:
    """Evaluate an already-loaded generator on one fixed image manifest."""
    if image_size < 11:
        raise ValueError("image_size must be at least 11 for Gaussian-window SSIM.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not pairs:
        raise RuntimeError("The image manifest is empty.")

    generator.eval()
    accumulator = MetricAccumulator.empty()
    failed: List[str] = []

    progress = tqdm(total=len(pairs), desc=description)
    try:
        for pair_batch in _chunks(pairs, batch_size):
            loaded: List[LoadedPair] = []
            for gray_path, rgb_path in pair_batch:
                try:
                    loaded.append(load_pair(gray_path, rgb_path, image_size))
                except Exception as error:
                    failed.append(f"{gray_path.name}: {error}")
                    progress.update(1)

            if not loaded:
                continue

            input_batch = torch.stack([item.gray_tensor for item in loaded], dim=0).to(device)
            with torch.inference_mode():
                output_batch = generator(input_batch)
            fake_images = generator_output_to_uint8(output_batch, image_size)
            if len(fake_images) != len(loaded):
                raise RuntimeError(
                    "Generator output batch length differs from the input batch: "
                    f"inputs={len(loaded)}, outputs={len(fake_images)}."
                )

            valid_real: List[np.ndarray] = []
            valid_fake: List[np.ndarray] = []
            for item, fake in zip(loaded, fake_images):
                try:
                    # Classical metrics are appended atomically. If any one
                    # fails, the pair is excluded from every deep metric too.
                    accumulator.append_classical(item.real_rgb, fake)
                    valid_real.append(item.real_rgb)
                    valid_fake.append(fake)
                except Exception as error:
                    failed.append(f"{item.key}: {error}")
                finally:
                    progress.update(1)

            if valid_real:
                lpips_values, real_features, fake_features = deep_models.compute_batch(
                    valid_real,
                    valid_fake,
                )
                accumulator.append_deep_batch(lpips_values, real_features, fake_features)
    finally:
        progress.close()

    if failed:
        print(f"\nFailed image pairs: {len(failed)}")
        for message in failed[:10]:
            print(f"  - {message}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    result = accumulator.finalise()
    if result.n_images != len(pairs) - len(failed):
        # This is a diagnostic rather than an error because multiple failures
        # cannot occur for one pair in the current pipeline.
        print(
            "Warning: evaluated count differs from manifest-minus-failures: "
            f"evaluated={result.n_images}, manifest={len(pairs)}, failures={len(failed)}."
        )
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    generator_factory: Callable[[], nn.Module],
    gray_dir: Path,
    rgb_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    deep_models: DeepMetricModels,
    strict_checkpoint: bool = True,
    pairs: Sequence[Tuple[Path, Path]] | None = None,
) -> MetricResult:
    """Build, load, and evaluate one generator checkpoint."""
    manifest = list(pairs) if pairs is not None else pair_images(gray_dir, rgb_dir)
    if not manifest:
        raise RuntimeError("No matched grayscale/RGB image pairs were found.")

    generator = generator_factory().to(device)
    load_generator_checkpoint(
        generator,
        checkpoint_path,
        device,
        strict=strict_checkpoint,
    )
    return evaluate_loaded_generator(
        generator,
        pairs=manifest,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        deep_models=deep_models,
        description=checkpoint_path.stem,
    )
