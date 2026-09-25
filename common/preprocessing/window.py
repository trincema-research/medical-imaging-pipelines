"""Percentile window → 8-bit. Used by challenge crop pipelines."""

from __future__ import annotations

import numpy as np


def window_to_uint8(
    arr: np.ndarray,
    *,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    lo = float(np.percentile(arr, lower_pct))
    hi = float(np.percentile(arr, upper_pct))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(arr.astype(np.float32), lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return (scaled * 255.0).astype(np.uint8)
