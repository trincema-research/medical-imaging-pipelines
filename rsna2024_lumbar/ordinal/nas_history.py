"""Ordinal metrics at the NAS best epoch from archived ``training_history.csv``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rsna2024_lumbar.nas.compact import HISTORY_FILENAME
from rsna2024_lumbar.ordinal.config import hyperparameters_dict, load_best_config


def _history_path_for_entry(entry: dict[str, Any], *, compact_root: Path) -> Path | None:
    """Resolve compact ``training_history.csv`` for a best-config JSON entry."""
    family = entry.get("family") or hyperparameters_dict(entry).get("model_type", "")
    layout = entry.get("archive_layout") or hyperparameters_dict(entry).get("input_layout", "2d")
    condition = entry.get("condition")
    trial_id = entry.get("trial_id")
    if not family or not condition or trial_id is None:
        return None
    archive_folder = {
        "vit": "ViT",
        "vit3d": "ViT",
        "maxvit": "MaxViT",
        "maxvit3d": "MaxViT",
        "convnext": "Convnext",
        "convnext3d": "Convnext",
        "efficientnet": "EfficientNet",
        "efficientnet3d": "EfficientNet",
    }.get(str(family), str(family))
    rel = (
        compact_root
        / archive_folder
        / str(layout).lower()
        / str(condition)
        / f"trial_{int(trial_id):04d}"
        / HISTORY_FILENAME
    )
    if rel.is_file():
        return rel
    return None


def _best_epoch_row(df: pd.DataFrame) -> pd.Series:
    if "val_acc" not in df.columns:
        return df.iloc[-1]
    return df.loc[df["val_acc"].idxmax()]


def _oa_from_accuracy_overall(row: pd.Series, prefix: str) -> float | None:
    key = f"{prefix}_accuracy_overall"
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def summarize_entry_from_nas_history(
    entry: dict[str, Any],
    *,
    compact_root: Path,
) -> dict[str, Any]:
    """
    At the best ``val_acc`` epoch in NAS history, map ``*_accuracy_overall`` to OA.

    O-MAE / QWK / SER are not stored in NAS histories (no per-sample preds); use refit
    for full ordinal metrics.
    """
    hist_path = _history_path_for_entry(entry, compact_root=compact_root)
    out: dict[str, Any] = {
        "condition": entry.get("condition"),
        "family": entry.get("family"),
        "archive_layout": entry.get("archive_layout"),
        "trial_id": entry.get("trial_id"),
        "history_path": str(hist_path) if hist_path else "",
        "best_epoch": "",
        "source": "nas_training_history",
    }
    if hist_path is None or not hist_path.is_file():
        out["note"] = "missing compact training_history.csv"
        return out
    df = pd.read_csv(hist_path)
    row = _best_epoch_row(df)
    out["best_epoch"] = int(row.get("epoch", -1))
    for prefix in ("train", "val", "test"):
        oa = _oa_from_accuracy_overall(row, prefix)
        if oa is not None:
            out[f"{prefix}_oa_overall"] = oa
    return out


def write_nas_history_ordinal_summary(
    best_config_path: Path,
    *,
    compact_root: Path,
    output_csv: Path,
) -> list[dict[str, Any]]:
    config = load_best_config(best_config_path)
    rows = [
        summarize_entry_from_nas_history(entry, compact_root=compact_root)
        for entry in config.get("entries", [])
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return rows
