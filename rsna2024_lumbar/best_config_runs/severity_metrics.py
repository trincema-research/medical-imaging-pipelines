"""Ordinal severity metrics for 3-class lumbar grades (0 Normal/Mild, 1 Moderate, 2 Severe)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

DEFAULT_IGNORE_LABEL = -1
SEVERITY_CLASS_LABELS = (0, 1, 2)


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
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int] = SEVERITY_CLASS_LABELS,
) -> float:
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
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred) >= 2))


def compute_severity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
) -> Dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}oa": ordinal_accuracy(y_true, y_pred),
        f"{p}omae": ordinal_mae(y_true, y_pred),
        f"{p}qwk": quadratic_weighted_kappa(y_true, y_pred),
        f"{p}ser": severe_error_rate(y_true, y_pred),
    }


# Backward-compatible aliases
compute_ordinal_metrics = compute_severity_metrics
