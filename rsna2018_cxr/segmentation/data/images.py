"""Image IO for the severity split.

Every study is already 1024x1024 grayscale PNG or JPEG. Segmentation reads
those pixels and writes masks on the same grid. A different size is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rsna2018_cxr.segmentation.constants import IMAGE_SUFFIXES, NATIVE_HW


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def image_hw(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` without decoding pixel values."""
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def read_grayscale(path: Path) -> np.ndarray:
    """Return a uint8 grayscale image at its file resolution."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def to_unit_float(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0


def assert_native_resolution(image: np.ndarray, shape: tuple[int, int] = NATIVE_HW) -> None:
    """Reject a resample. Cohort tensors stay at ``shape`` (height, width)."""
    actual = (int(image.shape[0]), int(image.shape[1]))
    if actual != tuple(shape):
        raise ValueError(
            f"Expected original resolution {shape[0]}x{shape[1]}, got {actual[0]}x{actual[1]}."
        )


def transform_record(patient_id: str, original_shape: tuple[int, int]) -> dict:
    """Provenance for a study that was segmented on its own 1024 grid."""
    actual = (int(original_shape[0]), int(original_shape[1]))
    if actual != NATIVE_HW:
        raise ValueError(
            f"Expected original resolution {NATIVE_HW[0]}x{NATIVE_HW[1]}, got {actual[0]}x{actual[1]}."
        )
    return {
        "patient_id": patient_id,
        "original_shape": [actual[0], actual[1]],
        "model_input_shape": [actual[0], actual[1]],
        "crop": None,
        "resize_method": None,
        "mask_resize_method": None,
        "inverse_transform_available": True,
    }


def dumps_transform(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"))
