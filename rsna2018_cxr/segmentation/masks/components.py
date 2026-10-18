"""Connected components with SciPy, OpenCV, or a small pure-Python fallback."""

from __future__ import annotations

import numpy as np


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    region = np.asarray(mask) > 0
    try:
        from scipy import ndimage

        labels, count = ndimage.label(region)
        return labels.astype(np.int32), int(count)
    except ImportError:
        pass
    try:
        import cv2

        count, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32), int(count - 1)
    except ImportError:
        return _flood_fill(region)


def _flood_fill(region: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = region.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for y in range(height):
        for x in range(width):
            if not region[y, x] or labels[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            labels[y, x] = count
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if region[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count
