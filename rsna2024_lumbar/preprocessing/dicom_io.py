"""Read a DICOM slice, window it, return a cropped RGB PIL image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from common.preprocessing.window import window_to_uint8
from rsna2024_lumbar.preprocessing.crops import (
    CropSettings,
    crop_bounds,
    crop_hw,
    extract_patch,
)


def rescale_pixels(ds: pydicom.Dataset, pixels: np.ndarray) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels.astype(np.float32) * slope + intercept


def read_gray_uint8(dicom_path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array
    if pixels.ndim > 2:
        pixels = pixels[..., 0]
    return window_to_uint8(rescale_pixels(ds, pixels))


def crop_dicom(
    dicom_path: str | Path,
    x: float,
    y: float,
    settings: CropSettings,
) -> Image.Image:
    gray = read_gray_uint8(dicom_path)
    h, w = gray.shape
    crop_h, crop_w = crop_hw(h, w, settings)
    x0, y0, x1, y1 = crop_bounds(x, y, crop_h, crop_w, h, w, settings)
    patch = extract_patch(gray, x0, y0, x1, y1, crop_h, crop_w)
    rgb = np.stack([patch, patch, patch], axis=-1)
    return Image.fromarray(rgb).convert("RGB")
