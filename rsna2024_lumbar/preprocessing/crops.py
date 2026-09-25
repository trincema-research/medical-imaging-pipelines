"""Crop geometry. Pixel math only — no DICOM I/O."""

from __future__ import annotations

from dataclasses import dataclass

from rsna2024_lumbar.preprocessing.constants import (
    BASE_CROP_FRACTION,
    CONDITIONS,
    CROP_POLICIES,
    CROP_POLICY_CENTERED,
    EXTEND50_EXTRA_OF_BASE,
    EXTEND50_SIDE_BY_CONDITION,
)

MIN_CROP_PX = 16


@dataclass(frozen=True)
class CropSettings:
    """One of: centered rectangle, or base square + extra width on one side."""

    crop_width_fraction: float | None = None
    crop_height_fraction: float | None = None
    crop_base_width_fraction: float | None = None
    crop_width_extra_left_fraction: float | None = None
    crop_width_extra_right_fraction: float | None = None

    def is_asymmetric(self) -> bool:
        return self.crop_base_width_fraction is not None


def crop_settings_for(condition: str, crop_policy: str) -> CropSettings:
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition}")
    if crop_policy not in CROP_POLICIES:
        raise ValueError(f"Unknown crop policy {crop_policy!r}.")
    if crop_policy == CROP_POLICY_CENTERED:
        return CropSettings(
            crop_width_fraction=BASE_CROP_FRACTION,
            crop_height_fraction=BASE_CROP_FRACTION,
        )
    extra = BASE_CROP_FRACTION * EXTEND50_EXTRA_OF_BASE
    if EXTEND50_SIDE_BY_CONDITION[condition] == "left":
        return CropSettings(
            crop_base_width_fraction=BASE_CROP_FRACTION,
            crop_height_fraction=BASE_CROP_FRACTION,
            crop_width_extra_left_fraction=extra,
            crop_width_extra_right_fraction=0.0,
        )
    return CropSettings(
        crop_base_width_fraction=BASE_CROP_FRACTION,
        crop_height_fraction=BASE_CROP_FRACTION,
        crop_width_extra_left_fraction=0.0,
        crop_width_extra_right_fraction=extra,
    )


def crop_hw(height: int, width: int, settings: CropSettings) -> tuple[int, int]:
    """Return (crop_h, crop_w) in pixels."""
    if settings.is_asymmetric():
        base_w = max(MIN_CROP_PX, int(round(width * float(settings.crop_base_width_fraction))))
        extra = int(round(width * (settings.crop_width_extra_left_fraction or 0.0)))
        extra += int(round(width * (settings.crop_width_extra_right_fraction or 0.0)))
        crop_w = max(MIN_CROP_PX, base_w + extra)
        crop_h = max(MIN_CROP_PX, int(round(height * float(settings.crop_height_fraction))))
        return crop_h, crop_w
    crop_w = max(MIN_CROP_PX, int(round(width * float(settings.crop_width_fraction))))
    crop_h = max(MIN_CROP_PX, int(round(height * float(settings.crop_height_fraction))))
    return crop_h, crop_w


def crop_bounds(
    x: float,
    y: float,
    crop_h: int,
    crop_w: int,
    height: int,
    width: int,
    settings: CropSettings,
) -> tuple[int, int, int, int]:
    """Inclusive-exclusive box (x0, y0, x1, y1), clamped to the slice."""
    cx = int(round(x))
    cy = int(round(y))
    y0 = max(0, cy - crop_h // 2)
    y1 = min(height, y0 + crop_h)

    if settings.is_asymmetric():
        base_w = max(MIN_CROP_PX, int(round(width * float(settings.crop_base_width_fraction))))
        x0_base = cx - base_w // 2
        x1_base = x0_base + base_w
        extra_left = (settings.crop_width_extra_left_fraction or 0.0) > 0.0
        if extra_left:
            x1 = min(width, x1_base)
            x0 = max(0, x1 - crop_w)
        else:
            x0 = max(0, x0_base)
            x1 = min(width, x0 + crop_w)
        return x0, y0, x1, y1

    x0 = max(0, cx - crop_w // 2)
    x1 = min(width, x0 + crop_w)
    return x0, y0, x1, y1


def extract_patch(
    gray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    crop_h: int,
    crop_w: int,
):
    """Zero-pad if the box hit a border (same as the training exporter)."""
    import numpy as np

    patch = gray[y0:y1, x0:x1]
    if patch.shape[0] == crop_h and patch.shape[1] == crop_w:
        return patch
    padded = np.zeros((crop_h, crop_w), dtype=gray.dtype)
    padded[: patch.shape[0], : patch.shape[1]] = patch
    return padded


def describe_settings(settings: CropSettings) -> str:
    if settings.is_asymmetric():
        h = float(settings.crop_height_fraction) * 100.0
        base = float(settings.crop_base_width_fraction) * 100.0
        if (settings.crop_width_extra_left_fraction or 0.0) > 0.0:
            extra = float(settings.crop_width_extra_left_fraction) * 100.0
            side = "left"
        else:
            extra = float(settings.crop_width_extra_right_fraction or 0.0) * 100.0
            side = "right"
        return f"{base:g}%×{h:g}% + {extra:g}% {side} (total W {base + extra:g}%)"
    w = float(settings.crop_width_fraction) * 100.0
    h = float(settings.crop_height_fraction) * 100.0
    return f"{w:g}%×{h:g}% centered"
