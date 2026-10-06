"""Classification metrics at the NAS best validation epoch (from ``training_history.csv``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rsna2024_lumbar.preprocessing.constants import LUMBAR_LEVELS

# Metrics logged each epoch by ``train_vit_lumbar.compute_multi_head_metrics`` (plus loss/acc).
_SPLIT_PREFIXES = ("train", "val", "test")


def _level_metric_fields() -> tuple[str, ...]:
    fields: list[str] = []
    for prefix in _SPLIT_PREFIXES:
        for level in LUMBAR_LEVELS:
            fields.append(f"{prefix}_accuracy_{level}")
            fields.append(f"{prefix}_f1_macro_{level}")
    return tuple(fields)


NAS_BEST_EPOCH_CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "train_loss",
    "test_loss",
    "train_accuracy_macro_levels",
    "val_accuracy_macro_levels",
    "test_accuracy_macro_levels",
    "train_f1_macro_overall",
    "val_f1_macro_overall",
    "test_f1_macro_overall",
    "train_accuracy_overall",
    "val_accuracy_overall",
    "test_accuracy_overall",
    *_level_metric_fields(),
)


def compact_history_path(
    *,
    family: str,
    archive_layout: str,
    condition: str,
    trial_id: int | str,
    compact_root: Path,
) -> Path | None:
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
        / str(archive_layout).lower()
        / str(condition)
        / f"trial_{int(trial_id):04d}"
        / "training_history.csv"
    )
    return rel if rel.is_file() else None


def metrics_at_best_val_acc_row(history_path: Path) -> dict[str, Any]:
    """Return metric columns from the epoch with maximum ``val_acc``."""
    df = pd.read_csv(history_path)
    if df.empty:
        return {}
    idx = df["val_acc"].idxmax() if "val_acc" in df.columns else df.index[-1]
    row = df.loc[idx]
    out: dict[str, Any] = {}
    for key in (*NAS_BEST_EPOCH_CLASSIFICATION_FIELDS, "val_loss"):
        if key not in row.index or pd.isna(row[key]):
            continue
        out[key] = float(row[key])
    # Align with legacy best-config names (train F1 at best epoch).
    if "train_f1_macro_levels" in row.index and not pd.isna(row["train_f1_macro_levels"]):
        out["train_f1_macro_levels_at_best"] = float(row["train_f1_macro_levels"])
    return out


def enrich_row_from_compact_history(
    row: dict[str, Any],
    *,
    compact_root: Path,
) -> None:
    """Merge NAS classification metrics into a best-config CSV/JSON row in place."""
    family = str(row.get("family") or row.get("model_group") or "")
    layout = str(row.get("archive_layout") or row.get("input_layout") or "")
    condition = row.get("condition")
    trial_id = row.get("trial_id")
    if not family or not layout or not condition or trial_id in ("", None):
        return
    hist = compact_history_path(
        family=family,
        archive_layout=layout,
        condition=str(condition),
        trial_id=trial_id,
        compact_root=compact_root,
    )
    if hist is None:
        return
    for key, value in metrics_at_best_val_acc_row(hist).items():
        row[key] = value
