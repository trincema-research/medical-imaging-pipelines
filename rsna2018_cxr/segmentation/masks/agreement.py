"""Mask agreement metrics.

Dice, IoU, area difference, and centroid distance are pure NumPy.
HD95, ASSD, and boundary F1 use a distance transform when SciPy is installed
and a boundary-to-boundary fallback otherwise.
"""

from __future__ import annotations

import numpy as np

_BOUNDARY_TOLERANCE = 2


def as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    left = as_bool(a)
    right = as_bool(b)
    intersection = np.logical_and(left, right).sum(dtype=np.float64)
    total = left.sum(dtype=np.float64) + right.sum(dtype=np.float64)
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left = as_bool(a)
    right = as_bool(b)
    intersection = np.logical_and(left, right).sum(dtype=np.float64)
    union = np.logical_or(left, right).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    return float(intersection / union)


def area_difference(a: np.ndarray, b: np.ndarray) -> float:
    left = float(as_bool(a).sum())
    right = float(as_bool(b).sum())
    baseline = max(left, right, 1.0)
    return abs(left - right) / baseline


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(as_bool(mask))
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = centroid(a)
    cb = centroid(b)
    if ca is None or cb is None:
        return float("nan")
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel contour via a 4-neighbor erosion."""
    region = as_bool(mask)
    if not region.any():
        return region
    up = np.ones_like(region)
    down = np.ones_like(region)
    left = np.ones_like(region)
    right = np.ones_like(region)
    up[1:] = region[:-1]
    down[:-1] = region[1:]
    left[:, 1:] = region[:, :-1]
    right[:, :-1] = region[:, 1:]
    eroded = region & up & down & left & right
    return region & ~eroded


def _pairwise_min_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.empty((0,), dtype=np.float64)
    # Chunk so a long contour does not allocate a full pairwise matrix.
    distances = np.empty(source.shape[0], dtype=np.float64)
    block = 512
    for start in range(0, source.shape[0], block):
        chunk = source[start : start + block]
        delta = chunk[:, None, :] - target[None, :, :]
        distances[start : start + chunk.shape[0]] = np.sqrt((delta ** 2).sum(axis=2)).min(axis=1)
    return distances


def surface_distances(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    left = as_bool(a)
    right = as_bool(b)
    if not left.any() or not right.any():
        return None
    try:
        from scipy.ndimage import distance_transform_edt

        bound_a = boundary(left)
        bound_b = boundary(right)
        dist_to_a = distance_transform_edt(~left)
        dist_to_b = distance_transform_edt(~right)
        return dist_to_a[bound_b], dist_to_b[bound_a]
    except ImportError:
        points_a = np.argwhere(boundary(left))
        points_b = np.argwhere(boundary(right))
        return _pairwise_min_distances(points_b, points_a), _pairwise_min_distances(points_a, points_b)


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    distances = surface_distances(a, b)
    if distances is None:
        return float("nan")
    pooled = np.concatenate(distances)
    if pooled.size == 0:
        return float("nan")
    return float(np.percentile(pooled, 95))


def assd(a: np.ndarray, b: np.ndarray) -> float:
    distances = surface_distances(a, b)
    if distances is None:
        return float("nan")
    left, right = distances
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float((left.mean() + right.mean()) / 2.0)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    region = as_bool(mask)
    if radius <= 0 or not region.any():
        return region
    height, width = region.shape
    ys, xs = np.nonzero(region)
    out = np.zeros_like(region)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            yy = ys + dy
            xx = xs + dx
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            out[yy[valid], xx[valid]] = True
    return out


def boundary_f1(a: np.ndarray, b: np.ndarray, tolerance: int = _BOUNDARY_TOLERANCE) -> float:
    bound_a = boundary(a)
    bound_b = boundary(b)
    if not bound_a.any() and not bound_b.any():
        return 1.0
    if not bound_a.any() or not bound_b.any():
        return 0.0
    near_b = _dilate(bound_b, tolerance)
    near_a = _dilate(bound_a, tolerance)
    precision = np.logical_and(bound_a, near_b).sum(dtype=np.float64) / bound_a.sum(dtype=np.float64)
    recall = np.logical_and(bound_b, near_a).sum(dtype=np.float64) / bound_b.sum(dtype=np.float64)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def pairwise(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "dice": dice(a, b),
        "iou": iou(a, b),
        "hd95": hd95(a, b),
        "assd": assd(a, b),
        "boundary_f1": boundary_f1(a, b),
        "area_difference": area_difference(a, b),
        "centroid_distance": centroid_distance(a, b),
    }


def mean_and_min_dice(masks: list[np.ndarray]) -> tuple[float, float]:
    """Mean and minimum pairwise Dice across two or more masks."""
    if len(masks) < 2:
        return float("nan"), float("nan")
    scores = [
        dice(masks[i], masks[j])
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
    ]
    return float(np.mean(scores)), float(np.min(scores))
