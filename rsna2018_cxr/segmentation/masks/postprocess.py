"""Deterministic lung-mask cleanup shared by every candidate."""

from __future__ import annotations

import numpy as np

from rsna2018_cxr.segmentation.masks.components import label_components


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(mask) >= threshold


def fill_holes(mask: np.ndarray) -> np.ndarray:
    region = np.asarray(mask) > 0
    try:
        from scipy.ndimage import binary_fill_holes

        return binary_fill_holes(region)
    except ImportError:
        return region


def remove_small_islands(mask: np.ndarray, min_area_fraction: float = 0.002) -> np.ndarray:
    region = np.asarray(mask) > 0
    labels, count = label_components(region)
    if count == 0:
        return region
    minimum = max(1, int(round(min_area_fraction * region.size)))
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum
    keep[0] = False
    return keep[labels]


def split_lungs(mask: np.ndarray, radiological: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Split a merged lung field into patient left and patient right.

    On a standard PA display the patient's left lung is on the image right.
    ``radiological=True`` uses that convention. Components do not overlap.
    """
    region = np.asarray(mask) > 0
    labels, count = label_components(region)
    empty = np.zeros_like(region)
    if count == 0:
        return empty, empty
    areas = [(int((labels == index).sum()), index) for index in range(1, count + 1)]
    areas.sort(reverse=True)
    chosen = [labels == index for _, index in areas[:2]]
    if len(chosen) == 1:
        component = chosen[0]
        ys, xs = np.nonzero(component)
        image_right = float(xs.mean()) >= (region.shape[1] / 2.0)
        patient_left = image_right if radiological else not image_right
        return (component, empty) if patient_left else (empty, component)
    centroids = []
    for component in chosen:
        _, xs = np.nonzero(component)
        centroids.append(float(xs.mean()))
    order = np.argsort(centroids)
    image_left = chosen[int(order[0])]
    image_right = chosen[int(order[1])]
    if radiological:
        return image_right, image_left
    return image_left, image_right


def postprocess_merged(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area_fraction: float = 0.002,
) -> np.ndarray:
    binary = binarize(mask, threshold)
    binary = remove_small_islands(binary, min_area_fraction=min_area_fraction)
    return fill_holes(binary)
