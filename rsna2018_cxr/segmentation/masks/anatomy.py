"""Anatomical plausibility checks.

Absolute ranges below are wide starting hypotheses. When a reference set of
high-confidence area fractions is available, :func:`thresholds_from_percentiles`
replaces them with percentile alerts instead of rejecting on a narrow constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsna2018_cxr.segmentation.masks.components import label_components


@dataclass
class QcThresholds:
    min_lung_area_fraction: float = 0.08
    max_lung_area_fraction: float = 0.80
    hard_max_lung_area_fraction: float = 0.95
    max_overlap_fraction: float = 0.05
    warning_asymmetry_ratio: float = 3.5
    hard_asymmetry_ratio: float = 8.0
    min_bbox_aspect: float = 0.25
    max_bbox_aspect: float = 2.2
    thorax_margin: float = 0.02
    min_component_area_fraction: float = 0.01


def thresholds_from_percentiles(area_fractions: np.ndarray) -> dict[str, float]:
    """Soft band is P1–P99. Hard band is P0.1–P99.9."""
    values = np.asarray(area_fractions, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ValueError("Need at least 20 reference area fractions to fit percentiles.")
    return {
        "soft_low": float(np.percentile(values, 1)),
        "soft_high": float(np.percentile(values, 99)),
        "hard_low": float(np.percentile(values, 0.1)),
        "hard_high": float(np.percentile(values, 99.9)),
    }


def anatomy_qc(
    merged: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    thresholds: QcThresholds | None = None,
) -> dict:
    """Return hard failures, soft warnings, and whether the mask may be silver."""
    limits = thresholds or QcThresholds()
    region = np.asarray(merged) > 0
    left_region = np.asarray(left) > 0
    right_region = np.asarray(right) > 0
    height, width = region.shape
    area = int(region.sum())
    fraction = area / float(region.size)
    labels, count = label_components(region)
    warnings: list[str] = []
    hard: list[str] = []
    counts = np.bincount(labels.ravel(), minlength=count + 1) if count else np.zeros(1, dtype=np.int64)
    minimum = limits.min_component_area_fraction * region.size
    large = int(np.count_nonzero(counts[1:] >= minimum)) if count else 0

    if area == 0:
        hard.append("empty")
    if fraction >= limits.hard_max_lung_area_fraction:
        hard.append("covers_almost_whole_image")
    elif fraction > limits.max_lung_area_fraction or (area > 0 and fraction < limits.min_lung_area_fraction):
        warnings.append("lung_area_fraction")
    if area > 0 and large != 2:
        warnings.append("principal_component_count")

    overlap = np.logical_and(left_region, right_region).sum(dtype=np.float64)
    smaller = max(1.0, float(min(left_region.sum(), right_region.sum())))
    if left_region.any() and right_region.any() and overlap / smaller > limits.max_overlap_fraction:
        hard.append("left_right_overlap")

    if left_region.any() and right_region.any():
        left_area = float(left_region.sum())
        right_area = float(right_region.sum())
        ratio = max(left_area, right_area) / max(1.0, min(left_area, right_area))
        if ratio >= limits.hard_asymmetry_ratio:
            hard.append("extreme_asymmetry")
        elif ratio >= limits.warning_asymmetry_ratio:
            warnings.append("asymmetry")
        left_c = _centroid_x(left_region)
        right_c = _centroid_x(right_region)
        # Patient left should sit on the image right under standard PA orientation.
        if left_c <= right_c:
            warnings.append("left_right_order")
        margin = limits.thorax_margin
        for center in (_centroid(left_region), _centroid(right_region)):
            y, x = center
            if y < margin * height or y > (1 - margin) * height or x < margin * width or x > (1 - margin) * width:
                warnings.append("centroid_outside_thorax")
                break
    elif area > 0:
        warnings.append("missing_side")

    if area > 0:
        ys, xs = np.nonzero(region)
        bbox_h = max(1, int(ys.max() - ys.min() + 1))
        bbox_w = max(1, int(xs.max() - xs.min() + 1))
        aspect = bbox_w / bbox_h
        if aspect < limits.min_bbox_aspect or aspect > limits.max_bbox_aspect:
            warnings.append("bbox_aspect")
        if ys.min() < 0 or xs.min() < 0 or ys.max() >= height or xs.max() >= width:
            hard.append("contour_outside_image")

    return {
        "pass": not hard,
        "hard": hard,
        "warnings": warnings,
        "area_fraction": fraction,
        "component_count": int(count),
        "principal_components": int(large),
    }


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def _centroid_x(mask: np.ndarray) -> float:
    return _centroid(mask)[1]
