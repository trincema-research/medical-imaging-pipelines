"""Ordinal severity metrics for 3-class lumbar grades (0 Normal/Mild, 1 Moderate, 2 Severe)."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

DEFAULT_IGNORE_LABEL = -1
ORDINAL_CLASS_LABELS = (0, 1, 2)


def _as_int_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return y_true, y_pred


def ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact match rate on valid samples (same as accuracy for ordered classes)."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute grade distance."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int] = ORDINAL_CLASS_LABELS,
) -> float:
    """Quadratic weighted kappa; distant disagreements penalized more strongly."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(
        cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic",
            labels=list(labels),
        )
    )


def severe_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with grade gap >= 2 (e.g. true 0 vs pred 2)."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred) >= 2))


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Core ordinal metrics for one flat label vector.

    Keys (with optional prefix): oa, omae, qwk, ser.
    """
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}oa": ordinal_accuracy(y_true, y_pred),
        f"{p}omae": ordinal_mae(y_true, y_pred),
        f"{p}qwk": quadratic_weighted_kappa(y_true, y_pred),
        f"{p}ser": severe_error_rate(y_true, y_pred),
    }


def metric_display_names() -> Dict[str, str]:
    return {
        "oa": "Ordinal Accuracy (OA)",
        "omae": "Ordinal MAE (O-MAE)",
        "qwk": "Quadratic Weighted Kappa (QWK)",
        "ser": "Severe Error Rate (SER)",
    }
