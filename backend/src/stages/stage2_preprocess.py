"""
Stage 2 — Acoustic Signal Pre-processing
=========================================
Applies the following pipeline to the raw sonar waterfall array:

  1. Lee Filter          — speckle suppression (preserves edges)
  2. CLAHE               — contrast enhancement
  3. Gamma correction    — adjust brightness curve
  4. 8-bit normalization — ready for AI inference

The Lee filter is implemented manually using scipy.ndimage so there
is no dependency on any paid or proprietary DSP library.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lee Filter
# ---------------------------------------------------------------------------
def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Lee adaptive speckle filter.

    For each pixel neighbourhood of `size × size`:
      - Compute local mean μ and local variance σ²
      - Estimate noise variance from overall image variance
      - Output = μ + k × (pixel - μ)   where k = σ² / (σ² + noise_var)

    This preserves strong edges (high local variance) while smoothing
    flat water background regions (low local variance).

    Args:
        img:  2-D float32 array
        size: Neighbourhood window size (must be odd; default 7)

    Returns:
        Filtered float32 array of same shape.
    """
    img = img.astype(np.float32)
    size = size if size % 2 == 1 else size + 1   # ensure odd

    # Local mean and mean of squares
    local_mean = uniform_filter(img, size=size)
    local_mean_sq = uniform_filter(img ** 2, size=size)
    local_var = local_mean_sq - local_mean ** 2

    # Noise variance estimated from the overall image
    # (Lee's original paper uses the mean of local variances as noise estimate)
    noise_var = float(np.mean(local_var))

    # Weighting coefficient k
    k = local_var / (local_var + noise_var + 1e-8)

    filtered = local_mean + k * (img - local_mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Kuan Filter (alternative to Lee — slightly more accurate noise model)
# ---------------------------------------------------------------------------
def kuan_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Kuan adaptive filter — similar to Lee but uses a multiplicative noise model.
    Slightly better at preserving fine texture in the acoustic shadow region.
    """
    img = img.astype(np.float32)
    size = size if size % 2 == 1 else size + 1

    local_mean = uniform_filter(img, size=size)
    local_sq   = uniform_filter(img ** 2, size=size)
    local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)

    # Coefficient of variation squared (noise model parameter)
    cu2 = float(np.var(img) / (np.mean(img) ** 2 + 1e-8))
    ci2 = local_var / (local_mean ** 2 + 1e-8)

    w = (1 - cu2 / (ci2 + 1e-8)).clip(0.0, 1.0)
    filtered = local_mean + w * (img - local_mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# CLAHE contrast enhancement
# ---------------------------------------------------------------------------
def apply_clahe(img_u8: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation to an 8-bit image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_u8)


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------
def gamma_correction(img_u8: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction: out = (in / 255)^(1/gamma) × 255."""
    lut = np.array(
        [min(255, int(((i / 255.0) ** (1.0 / gamma)) * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return lut[img_u8]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def preprocess(
    raw_array: np.ndarray,
    filter_type: str = "lee",
    filter_size: int | None = None,
    clahe_clip: float = 2.0,
    gamma: float = 1.2,
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        raw_array:   2-D float array from Stage 1
        filter_type: "lee" | "kuan" — which speckle filter to apply
        filter_size: Override the default filter window size from config
        clahe_clip:  CLAHE clip limit
        gamma:       Gamma correction exponent

    Returns:
        Processed uint8 grayscale NumPy array ready for AI inference.
    """
    logger.info("Stage 2 — pre-processing (filter=%s).", filter_type)

    arr = raw_array.astype(np.float32)
    size = filter_size or settings.LEE_FILTER_SIZE

    # --- 1. Speckle suppression ---
    if filter_type == "kuan":
        arr = kuan_filter(arr, size=size)
    else:
        arr = lee_filter(arr, size=size)

    # --- 2. Normalize to 8-bit ---
    arr_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 3. CLAHE contrast enhancement ---
    arr_u8 = apply_clahe(arr_u8, clip_limit=clahe_clip)

    # --- 4. Gamma correction ---
    arr_u8 = gamma_correction(arr_u8, gamma=gamma)

    logger.info("Stage 2 complete — output shape %s, dtype %s.", arr_u8.shape, arr_u8.dtype)
    return arr_u8
