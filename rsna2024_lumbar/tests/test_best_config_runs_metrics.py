from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score

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


def test_ordinal_accuracy_matches_sklearn():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    assert ordinal_accuracy(y_true, y_pred) == pytest.approx(
        accuracy_score(y_true, y_pred)
    )


def test_qwk_matches_sklearn_quadratic():
    y_true = np.array([0, 0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 1, 2, 1, 2])
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2])
    )


def test_omae_is_mean_absolute_error():
    y_true = np.array([0, 2, 1])
    y_pred = np.array([1, 0, 1])
    assert ordinal_mae(y_true, y_pred) == pytest.approx(1.0)


def _train_vit_style_accuracy_overall(
    logits: np.ndarray, targets: np.ndarray, *, ignore_label: int = -1
) -> float:
    """Mirror ``train_vit_lumbar.compute_multi_head_metrics`` overall accuracy."""
    all_valid = targets.reshape(-1) != ignore_label
    if not all_valid.any():
        return 0.0
    all_true = targets.reshape(-1)[all_valid]
    all_pred = logits.reshape(-1, logits.shape[-1]).argmax(axis=-1)[all_valid]
    return float(accuracy_score(all_true, all_pred))


def test_multi_head_oa_overall_matches_accuracy_overall():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((8, 5, 3)).astype(np.float32)
    targets = rng.integers(0, 3, size=(8, 5), dtype=np.int64)
    targets[0, 2] = -1
    targets[3, :] = -1
    sev = compute_multi_head_severity_metrics(logits, targets)
    assert _train_vit_style_accuracy_overall(logits, targets) == pytest.approx(
        sev["oa_overall"]
    )
    for level_idx, level in enumerate(("L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1")):
        y_true = targets[:, level_idx]
        valid = y_true != -1
        if not valid.any():
            assert sev[f"oa_{level}"] == 0.0
            continue
        y_pred = logits[:, level_idx].argmax(axis=-1)
        assert sev[f"oa_{level}"] == pytest.approx(
            accuracy_score(y_true[valid], y_pred[valid])
        )


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "rsna2024_lumbar/data/best_config_runs/results/pipeline_results_all_models.csv"
    ).is_file(),
    reason="committed pipeline results not present",
)
def test_committed_pipeline_oa_equals_accuracy_overall():
    import pandas as pd

    path = "rsna2024_lumbar/data/best_config_runs/results/pipeline_results_all_models.csv"
    df = pd.read_csv(path)
    for split in ("train", "val", "test"):
        acc = f"{split}_accuracy_overall"
        oa = f"{split}_oa_overall"
        diff = (df[acc].astype(float) - df[oa].astype(float)).abs()
        assert diff.max() == 0.0
