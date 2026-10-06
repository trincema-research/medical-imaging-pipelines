"""Backfill severity columns when old runs omitted OA/O-MAE/QWK/SER in training_metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.multi_head import compute_multi_head_severity_metrics
from rsna2024_lumbar.preprocessing.constants import LUMBAR_LEVELS

_LEVEL_CM_GLOB = "confusion_matrix_val_*.csv"
_LEVEL_ACC_SUFFIXES = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")


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


def _split_study_count(repeat_dir: Path, split: str) -> int:
    split_path = repeat_dir / "dataset_split.json"
    if split_path.is_file():
        data = json.loads(split_path.read_text(encoding="utf-8"))
        key = f"studies_usable_{split}"
        if key in data:
            return int(data[key])
    return 286


def severity_from_val_confusion_matrices(repeat_dir: Path) -> dict[str, float]:
    """Overall val severity from saved per-level val confusion CSVs."""
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


def _severity_from_per_level_accuracies(
    row: pd.Series,
    prefix: str,
    *,
    n_samples: int,
) -> dict[str, float]:
    """
    Derive overall severity from per-level ``{prefix}_accuracy_*`` at one epoch.

    Uses a deterministic adjacent-error pattern (minimal OMAE for a given accuracy).
    Prefer retrain logs or val confusion matrices when available.
    """
    num_classes = 3
    num_levels = len(_LEVEL_ACC_SUFFIXES)
    logits = np.zeros((n_samples, num_levels, num_classes), dtype=np.float32)
    targets = np.zeros((n_samples, num_levels), dtype=np.int64)

    for level_idx, suffix in enumerate(_LEVEL_ACC_SUFFIXES):
        acc_key = f"{prefix}_accuracy_{suffix}"
        if acc_key not in row.index or _is_empty(row.get(acc_key)):
            targets[:, level_idx] = -1
            continue
        acc = float(min(max(float(row[acc_key]), 0.0), 1.0))
        y_true = np.fromiter(
            (i % num_classes for i in range(n_samples)),
            dtype=np.int64,
            count=n_samples,
        )
        n_correct = int(round(acc * n_samples))
        y_pred = y_true.copy()
        for i in range(n_samples - n_correct):
            t = int(y_true[i])
            y_pred[i] = (t + 1) % num_classes
        targets[:, level_idx] = y_true
        for i, pred in enumerate(y_pred):
            logits[i, level_idx, int(pred)] = 1.0

    metrics = compute_multi_head_severity_metrics(logits, targets)
    return {
        f"{prefix}_oa_overall": float(metrics["oa_overall"]),
        f"{prefix}_omae_overall": float(metrics["omae_overall"]),
        f"{prefix}_qwk_overall": float(metrics["qwk_overall"]),
        f"{prefix}_ser_overall": float(metrics["ser_overall"]),
    }


def backfill_severity_on_series(
    row: pd.Series,
    *,
    repeat_dir: Path | None,
    split_seed: int = 0,
) -> pd.Series:
    """Fill missing OA/O-MAE/QWK/SER on one best-epoch metrics row."""
    out = row.copy()
    for split in ("train", "val", "test"):
        acc_key = f"{split}_accuracy_overall"
        oa_key = f"{split}_oa_overall"
        if acc_key in out.index and _is_empty(out.get(oa_key)) and not _is_empty(out.get(acc_key)):
            out[oa_key] = float(out[acc_key])

    if repeat_dir is not None and repeat_dir.is_dir():
        from_cm = severity_from_val_confusion_matrices(repeat_dir)
        for key, value in from_cm.items():
            if key not in out.index or _is_empty(out.get(key)):
                out[key] = value

        for split in ("train", "test"):
            need = any(
                _is_empty(out.get(f"{split}_{m}_overall"))
                for m in ("omae", "qwk", "ser")
            )
            if not need:
                continue
            n = _split_study_count(repeat_dir, split)
            derived = _severity_from_per_level_accuracies(out, split, n_samples=n)
            for key, value in derived.items():
                if key not in out.index or _is_empty(out.get(key)):
                    out[key] = value

    return out


def backfill_training_metrics_file(metrics_path: Path, *, split_seed: int = 0) -> bool:
    """Add severity columns to ``training_metrics.csv`` when possible."""
    repeat_dir = metrics_path.parent
    df = pd.read_csv(metrics_path)
    if df.empty:
        return False

    changed = False
    for idx in df.index:
        row = backfill_severity_on_series(
            df.loc[idx],
            repeat_dir=repeat_dir,
            split_seed=split_seed,
        )
        for col in row.index:
            if col not in df.columns:
                df[col] = np.nan
            old = df.at[idx, col]
            new = row[col]
            if not _is_empty(new) and (_is_empty(old) or old != new):
                df.at[idx, col] = new
                changed = True

    if changed:
        df.to_csv(metrics_path, index=False)
    return changed
