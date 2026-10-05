from __future__ import annotations

import numpy as np
import pytest

from rsna2024_lumbar.ordinal.metrics import (
    compute_ordinal_metrics,
    ordinal_accuracy,
    ordinal_mae,
    quadratic_weighted_kappa,
    severe_error_rate,
)
from rsna2024_lumbar.ordinal.multi_head import compute_multi_head_ordinal_metrics


def test_ordinal_accuracy_exact_match():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 0])
    assert ordinal_accuracy(y_true, y_pred) == 0.75


def test_ordinal_mae_adjacent_errors():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 1, 1])
    assert ordinal_mae(y_true, y_pred) == pytest.approx(2 / 3)


def test_severe_error_rate_only_gap_two():
    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([2, 1, 0, 1])
    assert severe_error_rate(y_true, y_pred) == 0.5


def test_qwk_perfect_agreement():
    y = np.array([0, 1, 2, 1, 0])
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_compute_ordinal_metrics_keys():
    m = compute_ordinal_metrics(np.array([0, 2]), np.array([0, 0]))
    assert set(m.keys()) == {"oa", "omae", "qwk", "ser"}
    assert m["ser"] == 0.5


def test_multi_head_ignores_invalid_labels():
    # N=1, 5 levels, 3 classes
    logits = np.zeros((1, 5, 3), dtype=np.float32)
    logits[0, 0, 2] = 10.0
    targets = np.full((1, 5), -1, dtype=np.int64)
    targets[0, 0] = 0
    m = compute_multi_head_ordinal_metrics(logits, targets)
    assert m["oa_overall"] == 0.0
    assert m["ser_overall"] == 1.0
