"""Five-head logits/targets → ordinal metrics (overall + per lumbar level)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rsna2024_lumbar.ordinal.metrics import (
    DEFAULT_IGNORE_LABEL,
    compute_ordinal_metrics,
)

# Match bundled training constants (avoid importing torch bundle at import time).
LUMBAR_LEVELS = ("L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1")


def compute_multi_head_ordinal_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> Dict[str, float]:
    """
    logits: (N, 5, C), targets: (N, 5)
    """
    metrics: Dict[str, float] = {}
    num_levels = logits.shape[1]
    level_oa: list[float] = []
    level_omae: list[float] = []
    level_qwk: list[float] = []
    level_ser: list[float] = []

    for level_idx in range(num_levels):
        level_key = LUMBAR_LEVELS[level_idx] if level_idx < len(LUMBAR_LEVELS) else f"L{level_idx}"
        y_true = targets[:, level_idx]
        y_pred = logits[:, level_idx].argmax(axis=-1)
        valid = y_true != ignore_label
        if not valid.any():
            metrics[f"oa_{level_key}"] = 0.0
            metrics[f"omae_{level_key}"] = 0.0
            metrics[f"qwk_{level_key}"] = 0.0
            metrics[f"ser_{level_key}"] = 0.0
            continue
        level_m = compute_ordinal_metrics(y_true[valid], y_pred[valid])
        metrics[f"oa_{level_key}"] = level_m["oa"]
        metrics[f"omae_{level_key}"] = level_m["omae"]
        metrics[f"qwk_{level_key}"] = level_m["qwk"]
        metrics[f"ser_{level_key}"] = level_m["ser"]
        level_oa.append(level_m["oa"])
        level_omae.append(level_m["omae"])
        level_qwk.append(level_m["qwk"])
        level_ser.append(level_m["ser"])

    metrics["oa_macro_levels"] = float(np.mean(level_oa)) if level_oa else 0.0
    metrics["omae_macro_levels"] = float(np.mean(level_omae)) if level_omae else 0.0
    metrics["qwk_macro_levels"] = float(np.mean(level_qwk)) if level_qwk else 0.0
    metrics["ser_macro_levels"] = float(np.mean(level_ser)) if level_ser else 0.0

    flat_true = targets.reshape(-1)
    flat_pred = logits.reshape(-1, logits.shape[-1]).argmax(axis=-1)
    valid_flat = flat_true != ignore_label
    if valid_flat.any():
        overall = compute_ordinal_metrics(flat_true[valid_flat], flat_pred[valid_flat])
        metrics["oa_overall"] = overall["oa"]
        metrics["omae_overall"] = overall["omae"]
        metrics["qwk_overall"] = overall["qwk"]
        metrics["ser_overall"] = overall["ser"]
    else:
        metrics["oa_overall"] = 0.0
        metrics["omae_overall"] = 0.0
        metrics["qwk_overall"] = 0.0
        metrics["ser_overall"] = 0.0
    return metrics


def merge_ordinal_metrics(
    epoch_history: Dict[str, Any],
    prefix: str,
    metrics: Dict[str, float],
) -> None:
    for key, value in metrics.items():
        epoch_history[f"{prefix}_{key}"] = value


def append_ordinal_to_epoch_history(
    epoch_history: Dict[str, Any],
    prefix: str,
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> None:
    merge_ordinal_metrics(
        epoch_history,
        prefix,
        compute_multi_head_ordinal_metrics(
            logits, targets, ignore_label=ignore_label
        ),
    )
