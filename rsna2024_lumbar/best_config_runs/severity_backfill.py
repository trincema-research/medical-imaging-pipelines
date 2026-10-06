"""Optional legacy helpers when old runs omitted severity columns (not a substitute for training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.multi_head import compute_multi_head_severity_metrics
from rsna2024_lumbar.preprocessing.constants import LUMBAR_LEVELS

_LEVEL_CM_GLOB = "confusion_matrix_val_*.csv"


def _is_empty(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _labels_from_confusion_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    class_names = list(df.columns)
    label_map = {name: idx for idx, name in enumerate(class_names)}
    y_true: list[int] = []
    y_pred: list[int] = []
    for true_name, row in df.iterrows():
        true_idx = label_map.get(str(true_name))
        if true_idx is None:
            continue
        for pred_name, count in row.items():
            pred_idx = label_map.get(str(pred_name))
            if pred_idx is None:
                continue
            n = int(count)
            if n <= 0:
                continue
            y_true.extend([true_idx] * n)
            y_pred.extend([pred_idx] * n)
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def severity_from_val_confusion_matrices(repeat_dir: Path) -> dict[str, float]:
    """Val-only severity from saved per-level val confusion CSVs (best val snapshot)."""
    num_classes = 3
    num_levels = len(LUMBAR_LEVELS)
    per_level: list[tuple[np.ndarray, np.ndarray] | None] = [None] * num_levels

    for cm_path in repeat_dir.glob(_LEVEL_CM_GLOB):
        stem = cm_path.stem.replace("confusion_matrix_val_", "")
        try:
            level_idx = [k.lower() for k in LUMBAR_LEVELS].index(stem)
        except ValueError:
            continue
        y_true, y_pred = _labels_from_confusion_csv(cm_path)
        if y_true.size == 0:
            continue
        per_level[level_idx] = (y_true, y_pred)

    if all(v is None for v in per_level):
        return {}

    n = max(v[0].size for v in per_level if v is not None)
    logits = np.zeros((n, num_levels, num_classes), dtype=np.float32)
    targets = np.full((n, num_levels), -1, dtype=np.int64)

    for level_idx, pair in enumerate(per_level):
        if pair is None:
            continue
        y_true, y_pred = pair
        if y_true.size != n:
            raise ValueError(
                f"Level sample count mismatch in {repeat_dir} "
                f"(expected {n}, got {y_true.size} at level {level_idx})"
            )
        targets[:, level_idx] = y_true
        for i, pred in enumerate(y_pred):
            logits[i, level_idx, int(pred)] = 1.0

    metrics = compute_multi_head_severity_metrics(logits, targets)
    return {
        "val_oa_overall": float(metrics["oa_overall"]),
        "val_omae_overall": float(metrics["omae_overall"]),
        "val_qwk_overall": float(metrics["qwk_overall"]),
        "val_ser_overall": float(metrics["ser_overall"]),
    }


def backfill_severity_on_series(
    row: pd.Series,
    *,
    repeat_dir: Path | None,
    legacy: bool = False,
) -> pd.Series:
    """
    Legacy harvest only.

    - OA from ``*_accuracy_overall`` when missing.
    - Val O-MAE/QWK/SER from val confusion matrices when missing.
    Does not invent train/test ordinal metrics.
    """
    if not legacy:
        return row
    out = row.copy()
    for split in ("train", "val", "test"):
        acc_key = f"{split}_accuracy_overall"
        oa_key = f"{split}_oa_overall"
        if acc_key in out.index and _is_empty(out.get(oa_key)) and not _is_empty(out.get(acc_key)):
            out[oa_key] = float(out[acc_key])

    if repeat_dir is not None and repeat_dir.is_dir():
        for key, value in severity_from_val_confusion_matrices(repeat_dir).items():
            if key not in out.index or _is_empty(out.get(key)):
                out[key] = value
    return out
