from __future__ import annotations

import numpy as np
import pytest

from rsna2024_lumbar.best_config_runs.multi_head import compute_multi_head_severity_metrics
from rsna2024_lumbar.best_config_runs.severity_metrics import (
    compute_severity_metrics,
    ordinal_accuracy,
    ordinal_mae,
    quadratic_weighted_kappa,
    severe_error_rate,
)


def test_ordinal_accuracy_exact_match():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 0])
    assert ordinal_accuracy(y_true, y_pred) == 0.75


def test_severe_error_rate_only_gap_two():
    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([2, 1, 0, 1])
    assert severe_error_rate(y_true, y_pred) == 0.5


def test_compute_severity_metrics_keys():
    m = compute_severity_metrics(np.array([0, 2]), np.array([0, 0]))
    assert set(m.keys()) == {"oa", "omae", "qwk", "ser"}


def test_multi_head_severe_error():
    logits = np.zeros((1, 5, 3), dtype=np.float32)
    logits[0, 0, 2] = 10.0
    targets = np.full((1, 5), -1, dtype=np.int64)
    targets[0, 0] = 0
    m = compute_multi_head_severity_metrics(logits, targets)
    assert m["oa_overall"] == 0.0
    assert m["ser_overall"] == 1.0
