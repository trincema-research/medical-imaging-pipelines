"""Read a DICOM slice, window it, return a cropped RGB PIL image."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SliceGeometry:
    ipp: tuple[float, float, float]
    iop: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]
    rows: int
    columns: int
    instance_number: int


def rescale_pixels(ds: pydicom.Dataset, pixels: np.ndarray) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels.astype(np.float32) * slope + intercept


def _as_floats(values, n: int) -> tuple[float, ...] | None:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if len(out) != n:
        return None
    return out


def geometry_from_dataset(ds: pydicom.Dataset) -> SliceGeometry | None:
    ipp = _as_floats(getattr(ds, "ImagePositionPatient", None), 3)
    iop = _as_floats(getattr(ds, "ImageOrientationPatient", None), 6)
    spacing = _as_floats(getattr(ds, "PixelSpacing", None), 2)
    if ipp is None or iop is None or spacing is None:
        return None
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        return None
    instance = getattr(ds, "InstanceNumber", None)
    try:
        instance_number = int(instance) if instance is not None else 0
    except (TypeError, ValueError):
        instance_number = 0
    return SliceGeometry(
        ipp=(ipp[0], ipp[1], ipp[2]),
        iop=(iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]),
        pixel_spacing=(spacing[0], spacing[1]),
        rows=int(getattr(ds, "Rows", 0) or 0),
        columns=int(getattr(ds, "Columns", 0) or 0),
        instance_number=instance_number,
    )


def read_gray_uint8(dicom_path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array
    if pixels.ndim > 2:
        pixels = pixels[..., 0]
    return window_to_uint8(rescale_pixels(ds, pixels))


def read_gray_and_geometry(dicom_path: str | Path) -> tuple[np.ndarray, SliceGeometry] | None:
    ds = pydicom.dcmread(str(dicom_path))
    geometry = geometry_from_dataset(ds)
    if geometry is None:
        return None
    pixels = ds.pixel_array
    if pixels.ndim > 2:
        pixels = pixels[..., 0]
    return window_to_uint8(rescale_pixels(ds, pixels)), geometry


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
