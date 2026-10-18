"""Classical lung-field candidate used on the inspection subset.

OpenCV CLAHE, a bilateral filter, and Otsu. It traces a body silhouette and
is drawn in orange where it disagrees with the U-Net. It is not a lung model
and it does not run on the full cohort.
"""

from __future__ import annotations

import time

import numpy as np

from rsna2018_cxr.segmentation.masks.postprocess import postprocess_merged, split_lungs
from rsna2018_cxr.segmentation.models.types import LungPrediction


def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    import cv2

    _, normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverted = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = gray.shape
    x0, x1 = int(width * 0.2), int(width * 0.8)
    y0, y1 = int(height * 0.15), int(height * 0.9)
    score_normal = np.count_nonzero(normal[y0:y1, x0:x1])
    score_inverted = np.count_nonzero(inverted[y0:y1, x0:x1])
    return normal if score_normal >= score_inverted else inverted


def segment_classical(gray_u8: np.ndarray) -> LungPrediction:
    import cv2

    started = time.perf_counter()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_u8)
    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    mask = _otsu_mask(enhanced)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    merged = postprocess_merged(mask > 0, min_area_fraction=0.01)
    left, right = split_lungs(merged, radiological=True)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    return LungPrediction(
        left_lung=left,
        right_lung=right,
        merged_lungs=merged,
        probability_map=None,
        runtime_ms=runtime_ms,
        model_name="classical_otsu",
        model_version="1",
        metadata={"normalization": "clahe"},
    )
