#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metric contract for grayscale-to-RGB reconstruction evaluation.

Import this module from the baseline, validation, and final-evaluation scripts.
Using one implementation prevents silent drift in colour conversion, SSIM
settings, Inception preprocessing, and dataset-level aggregation.

Metric contract
---------------
* RGB arrays are uint8-like values on [0, 255].
* PSNR and SSIM are computed on analog BT.601 luminance Y.
* UV-PSNR averages independently range-normalised U/V mean squared errors.
* UV-SSIM is the arithmetic mean of zero-centred U-SSIM and V-SSIM.
* LPIPS uses the VGG backbone and inputs mapped to [-1, 1].
* Inception features use torchvision Inception-v3 IMAGENET1K_V1 weights,
  the preprocessing attached to those weights, and transform_input=False.
* KID is the unbiased full-sample polynomial-kernel estimate.
* FID and KID are internally comparable only when every script uses this exact
  model, weights, preprocessing, image manifest, and aggregation code.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableSequence, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# Analog ITU-R BT.601 YUV coefficients. Rows: Y, U, V; columns: R, G, B.
BT601_YUV_MATRIX = np.array(
    [
        [0.29900, 0.58700, 0.11400],
        [-0.14713, -0.28886, 0.43600],
        [0.61500, -0.51499, -0.10001],
    ],
    dtype=np.float64,
)

Y_DATA_RANGE = 255.0

# Full theoretical channel range for RGB values in [0, 255].
def _linear_channel_range(coefficients: np.ndarray) -> float:
    positive = np.clip(coefficients, 0.0, None).sum()
    negative = np.clip(coefficients, None, 0.0).sum()
    return float(255.0 * (positive - negative))


U_DATA_RANGE = _linear_channel_range(BT601_YUV_MATRIX[1])  # 222.35745
V_DATA_RANGE = _linear_channel_range(BT601_YUV_MATRIX[2])  # 313.65000

# Settings recommended by scikit-image for reproducing the Wang et al. SSIM
# formulation. The data_range is supplied separately for Y, U, and V.
SSIM_OPTIONS = {
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
}


@dataclass(frozen=True)
class UVMetrics:
    """Chrominance metrics for one image pair."""

    uv_psnr: float
    uv_ssim: float
    u_ssim: float
    v_ssim: float


@dataclass(frozen=True)
class MetricResult:
    """Aggregated metrics for one evaluated model and image manifest."""

    n_images: int
    psnr: float
    ssim: float
    uv_psnr: float
    uv_ssim: float
    u_ssim: float
    v_ssim: float
    rmse: float
    mae: float
    delta_e: float
    lpips: float
    kid: float
    fid: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "n_images": self.n_images,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "uv_psnr": self.uv_psnr,
            "uv_ssim": self.uv_ssim,
            "u_ssim": self.u_ssim,
            "v_ssim": self.v_ssim,
            "rmse": self.rmse,
            "mae": self.mae,
            "delta_e": self.delta_e,
            "lpips": self.lpips,
            "kid": self.kid,
            "fid": self.fid,
        }


@dataclass
class MetricAccumulator:
    """Synchronised storage for per-image and distribution-level metrics."""

    psnr: List[float]
    ssim: List[float]
    uv_psnr: List[float]
    uv_ssim: List[float]
    u_ssim: List[float]
    v_ssim: List[float]
    rmse: List[float]
    mae: List[float]
    delta_e: List[float]
    lpips: List[float]
    real_feature_batches: List[np.ndarray]
    fake_feature_batches: List[np.ndarray]

    @classmethod
    def empty(cls) -> "MetricAccumulator":
        return cls(
            psnr=[],
            ssim=[],
            uv_psnr=[],
            uv_ssim=[],
            u_ssim=[],
            v_ssim=[],
            rmse=[],
            mae=[],
            delta_e=[],
            lpips=[],
            real_feature_batches=[],
            fake_feature_batches=[],
        )

    @property
    def n_images(self) -> int:
        return len(self.psnr)

    def append_classical(self, real: np.ndarray, fake: np.ndarray) -> None:
        """Compute all classical metrics first, then append them atomically."""
        psnr_value, ssim_value = compute_luminance_metrics(real, fake)
        uv = compute_uv_metrics(real, fake)
        rmse_value = compute_rmse(real, fake)
        mae_value = compute_mae(real, fake)
        delta_e_value = compute_delta_e(real, fake)

        self.psnr.append(psnr_value)
        self.ssim.append(ssim_value)
        self.uv_psnr.append(uv.uv_psnr)
        self.uv_ssim.append(uv.uv_ssim)
        self.u_ssim.append(uv.u_ssim)
        self.v_ssim.append(uv.v_ssim)
        self.rmse.append(rmse_value)
        self.mae.append(mae_value)
        self.delta_e.append(delta_e_value)

    def append_deep_batch(
        self,
        lpips_values: np.ndarray,
        real_features: np.ndarray,
        fake_features: np.ndarray,
    ) -> None:
        self.lpips.extend(float(value) for value in np.asarray(lpips_values).reshape(-1))
        self.real_feature_batches.append(np.asarray(real_features, dtype=np.float64))
        self.fake_feature_batches.append(np.asarray(fake_features, dtype=np.float64))

    def finalise(self) -> MetricResult:
        if self.n_images == 0:
            raise RuntimeError("No valid image pairs were accumulated.")

        if len(self.lpips) != self.n_images:
            raise RuntimeError(
                "Metric count mismatch: "
                f"classical={self.n_images}, LPIPS={len(self.lpips)}."
            )

        if not self.real_feature_batches or not self.fake_feature_batches:
            raise RuntimeError("No Inception features were accumulated.")

        real_features = np.concatenate(self.real_feature_batches, axis=0)
        fake_features = np.concatenate(self.fake_feature_batches, axis=0)

        if real_features.shape[0] != self.n_images or fake_features.shape[0] != self.n_images:
            raise RuntimeError(
                "Feature count mismatch: "
                f"images={self.n_images}, real={real_features.shape[0]}, "
                f"fake={fake_features.shape[0]}."
            )

        return MetricResult(
            n_images=self.n_images,
            psnr=float(np.mean(self.psnr)),
            ssim=float(np.mean(self.ssim)),
            uv_psnr=float(np.mean(self.uv_psnr)),
            uv_ssim=float(np.mean(self.uv_ssim)),
            u_ssim=float(np.mean(self.u_ssim)),
            v_ssim=float(np.mean(self.v_ssim)),
            rmse=float(np.mean(self.rmse)),
            mae=float(np.mean(self.mae)),
            delta_e=float(np.mean(self.delta_e)),
            lpips=float(np.mean(self.lpips)),
            kid=compute_kid(real_features, fake_features),
            fid=calculate_fid(real_features, fake_features),
        )


# =============================================================================
# Validation and colour conversion
# =============================================================================


def validate_rgb_image(image: np.ndarray, name: str) -> np.ndarray:
    """Validate an RGB image and return float64 without changing its scale."""
    array = np.asarray(image)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3); received {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric; received dtype {array.dtype}.")

    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            f"{name} must use the [0, 255] RGB scale; observed "
            f"minimum={minimum:.6f}, maximum={maximum:.6f}."
        )

    return array


def validate_rgb_pair(real: np.ndarray, fake: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    real_f = validate_rgb_image(real, "real")
    fake_f = validate_rgb_image(fake, "fake")
    if real_f.shape != fake_f.shape:
        raise ValueError(f"Image shape mismatch: real={real_f.shape}, fake={fake_f.shape}.")
    return real_f, fake_f


def rgb_to_yuv_bt601(image: np.ndarray) -> np.ndarray:
    """Convert RGB [0, 255] to zero-centred analog BT.601 YUV."""
    rgb = validate_rgb_image(image, "image")
    return rgb @ BT601_YUV_MATRIX.T


# =============================================================================
# Per-image metrics
# =============================================================================


def compute_luminance_metrics(real: np.ndarray, fake: np.ndarray) -> Tuple[float, float]:
    """Compute BT.601 luminance PSNR and SSIM."""
    real_f, fake_f = validate_rgb_pair(real, fake)
    real_y = real_f @ BT601_YUV_MATRIX[0]
    fake_y = fake_f @ BT601_YUV_MATRIX[0]

    psnr = peak_signal_noise_ratio(real_y, fake_y, data_range=Y_DATA_RANGE)
    ssim = structural_similarity(
        real_y,
        fake_y,
        data_range=Y_DATA_RANGE,
        **SSIM_OPTIONS,
    )
    return float(psnr), float(ssim)


def compute_uv_metrics(real: np.ndarray, fake: np.ndarray) -> UVMetrics:
    """Compute range-normalised UV-PSNR and zero-centred UV-SSIM."""
    real_f, fake_f = validate_rgb_pair(real, fake)
    real_yuv = real_f @ BT601_YUV_MATRIX.T
    fake_yuv = fake_f @ BT601_YUV_MATRIX.T

    real_u, real_v = real_yuv[..., 1], real_yuv[..., 2]
    fake_u, fake_v = fake_yuv[..., 1], fake_yuv[..., 2]

    mse_u = float(np.mean(np.square(real_u - fake_u)))
    mse_v = float(np.mean(np.square(real_v - fake_v)))

    normalised_mse = 0.5 * (
        mse_u / (U_DATA_RANGE**2) + mse_v / (V_DATA_RANGE**2)
    )
    uv_psnr = math.inf if normalised_mse == 0.0 else -10.0 * math.log10(normalised_mse)

    u_ssim = structural_similarity(
        real_u,
        fake_u,
        data_range=U_DATA_RANGE,
        **SSIM_OPTIONS,
    )
    v_ssim = structural_similarity(
        real_v,
        fake_v,
        data_range=V_DATA_RANGE,
        **SSIM_OPTIONS,
    )

    return UVMetrics(
        uv_psnr=float(uv_psnr),
        uv_ssim=0.5 * (float(u_ssim) + float(v_ssim)),
        u_ssim=float(u_ssim),
        v_ssim=float(v_ssim),
    )


def compute_rmse(real: np.ndarray, fake: np.ndarray) -> float:
    """RGB RMSE after scaling both images to [0, 1]."""
    real_f, fake_f = validate_rgb_pair(real, fake)
    difference = (real_f - fake_f) / 255.0
    return float(np.sqrt(np.mean(np.square(difference))))


def compute_mae(real: np.ndarray, fake: np.ndarray) -> float:
    """RGB MAE on [0, 255]."""
    real_f, fake_f = validate_rgb_pair(real, fake)
    return float(np.mean(np.abs(real_f - fake_f)))


def compute_delta_e(real: np.ndarray, fake: np.ndarray) -> float:
    """Mean CIEDE2000 colour difference."""
    real_f, fake_f = validate_rgb_pair(real, fake)
    real_lab = rgb2lab(real_f / 255.0)
    fake_lab = rgb2lab(fake_f / 255.0)
    return float(np.mean(deltaE_ciede2000(real_lab, fake_lab)))


# =============================================================================
# LPIPS and Inception features
# =============================================================================


class DeepMetricModels:
    """Cached LPIPS-VGG and torchvision Inception-v3 feature models."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

        try:
            import lpips  # Imported lazily for a clearer missing-package error.
        except ImportError as error:
            raise RuntimeError(
                "The 'lpips' package is required. Install it in the evaluation "
                "environment before running this script."
            ) from error

        self.lpips_model = lpips.LPIPS(net="vgg").to(device)
        self.lpips_model.eval()

        self.inception_weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.inception_preprocess = self.inception_weights.transforms()
        self.inception_model = models.inception_v3(
            weights=self.inception_weights,
            aux_logits=True,
            transform_input=False,
        )
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.inception_model.to(device)

    @staticmethod
    def _images_to_unit_tensor(images: Sequence[np.ndarray]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required.")

        validated = [validate_rgb_image(image, "batch image") for image in images]
        first_shape = validated[0].shape
        if any(image.shape != first_shape for image in validated):
            raise ValueError("All images in a deep-metric batch must have the same shape.")

        batch = np.stack(validated, axis=0)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0

    @torch.inference_mode()
    def compute_batch(
        self,
        real_images: Sequence[np.ndarray],
        fake_images: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return LPIPS values and 2048-d Inception features for a batch."""
        if len(real_images) != len(fake_images):
            raise ValueError(
                f"Batch length mismatch: real={len(real_images)}, fake={len(fake_images)}."
            )

        real_unit = self._images_to_unit_tensor(real_images)
        fake_unit = self._images_to_unit_tensor(fake_images)
        if real_unit.shape != fake_unit.shape:
            raise ValueError(
                f"Deep metric tensor mismatch: real={tuple(real_unit.shape)}, "
                f"fake={tuple(fake_unit.shape)}."
            )

        real_lpips = (2.0 * real_unit - 1.0).to(self.device)
        fake_lpips = (2.0 * fake_unit - 1.0).to(self.device)
        lpips_values = self.lpips_model(real_lpips, fake_lpips)
        lpips_np = lpips_values.reshape(-1).detach().cpu().numpy().astype(np.float64)

        # The weight transform resizes to 342, centre-crops to 299, and applies
        # the exact ImageNet normalisation associated with these weights.
        real_inception = self.inception_preprocess(real_unit).to(self.device)
        fake_inception = self.inception_preprocess(fake_unit).to(self.device)

        real_features = self.inception_model(real_inception)
        fake_features = self.inception_model(fake_inception)

        # Defensive handling for torchvision variants returning a named tuple.
        if not isinstance(real_features, torch.Tensor):
            real_features = real_features.logits
        if not isinstance(fake_features, torch.Tensor):
            fake_features = fake_features.logits

        return (
            lpips_np,
            real_features.detach().cpu().numpy().astype(np.float64),
            fake_features.detach().cpu().numpy().astype(np.float64),
        )


# =============================================================================
# KID and FID
# =============================================================================


def _polynomial_kernel_block(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    gamma = 1.0 / x.shape[1]
    return np.power(gamma * (x @ y.T) + coef0, degree)


def _kernel_sum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    exclude_diagonal: bool,
    block_size: int = 512,
) -> float:
    total = 0.0
    same_object = x is y

    for row_start in range(0, x.shape[0], block_size):
        row_end = min(row_start + block_size, x.shape[0])
        x_block = x[row_start:row_end]

        for col_start in range(0, y.shape[0], block_size):
            col_end = min(col_start + block_size, y.shape[0])
            y_block = y[col_start:col_end]
            kernel = _polynomial_kernel_block(x_block, y_block)
            total += float(kernel.sum(dtype=np.float64))

            if exclude_diagonal and same_object:
                overlap_start = max(row_start, col_start)
                overlap_end = min(row_end, col_end)
                if overlap_start < overlap_end:
                    rows = np.arange(overlap_start, overlap_end) - row_start
                    cols = np.arange(overlap_start, overlap_end) - col_start
                    total -= float(kernel[rows, cols].sum(dtype=np.float64))

    return total


def compute_kid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Unbiased full-sample polynomial-kernel KID estimate."""
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)

    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("KID features must be two-dimensional arrays.")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(f"KID feature dimensions differ: {real.shape} vs {fake.shape}.")

    m, n = real.shape[0], fake.shape[0]
    if m < 2 or n < 2:
        return float("nan")

    sum_xx = _kernel_sum(real, real, exclude_diagonal=True) / (m * (m - 1))
    sum_yy = _kernel_sum(fake, fake, exclude_diagonal=True) / (n * (n - 1))
    sum_xy = _kernel_sum(real, fake, exclude_diagonal=False) / (m * n)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Numerically guarded Fréchet distance between two feature sets."""
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)

    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("FID features must be two-dimensional arrays.")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(f"FID feature dimensions differ: {real.shape} vs {fake.shape}.")
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return float("nan")

    mu_real = np.mean(real, axis=0)
    mu_fake = np.mean(fake, axis=0)
    covariance_real = np.atleast_2d(np.cov(real, rowvar=False))
    covariance_fake = np.atleast_2d(np.cov(fake, rowvar=False))

    mean_difference = mu_real - mu_fake
    covariance_mean = linalg.sqrtm(covariance_real @ covariance_fake)

    if not np.all(np.isfinite(covariance_mean)):
        warnings.warn(
            "FID covariance product was singular; adding a small diagonal offset.",
            RuntimeWarning,
            stacklevel=2,
        )
        identity = np.eye(covariance_real.shape[0], dtype=np.float64)
        covariance_mean = linalg.sqrtm(
            (covariance_real + epsilon * identity)
            @ (covariance_fake + epsilon * identity)
        )

    if np.iscomplexobj(covariance_mean):
        maximum_imaginary = float(np.max(np.abs(covariance_mean.imag)))
        if maximum_imaginary > 1e-3:
            warnings.warn(
                "FID matrix square root has a non-negligible imaginary component "
                f"({maximum_imaginary:.3e}); using the real part.",
                RuntimeWarning,
                stacklevel=2,
            )
        covariance_mean = covariance_mean.real

    fid = (
        float(mean_difference @ mean_difference)
        + float(np.trace(covariance_real))
        + float(np.trace(covariance_fake))
        - 2.0 * float(np.trace(covariance_mean))
    )
    if fid < 0.0 and abs(fid) < 1e-8:
        fid = 0.0
    return float(fid)


# =============================================================================
# Image manifest helpers
# =============================================================================


def _image_index(directory: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in index:
            raise ValueError(
                f"Ambiguous image stem {path.stem!r} in {directory}: "
                f"{index[key].name!r} and {path.name!r}."
            )
        index[key] = path
    return index


def pair_images(gray_dir: Path, rgb_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair grayscale and RGB images by case-insensitive filename stem."""
    if not gray_dir.is_dir():
        raise FileNotFoundError(f"Grayscale directory not found: {gray_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    gray_index = _image_index(gray_dir)
    rgb_index = _image_index(rgb_dir)
    shared_keys = sorted(gray_index.keys() & rgb_index.keys())

    missing_rgb = sorted(gray_index.keys() - rgb_index.keys())
    missing_gray = sorted(rgb_index.keys() - gray_index.keys())
    if missing_rgb:
        print(f"Warning: {len(missing_rgb)} grayscale image(s) have no RGB match.")
    if missing_gray:
        print(f"Warning: {len(missing_gray)} RGB image(s) have no grayscale match.")

    return [(gray_index[key], rgb_index[key]) for key in shared_keys]
