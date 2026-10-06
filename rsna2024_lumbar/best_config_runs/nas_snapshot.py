"""OA at NAS best val epoch from compact ``training_history.csv`` (no retrain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rsna2024_lumbar.nas.compact import HISTORY_FILENAME
from rsna2024_lumbar.best_config_runs.config import hyperparameters_dict, load_best_config


def _history_path_for_entry(entry: dict[str, Any], *, compact_root: Path) -> Path | None:
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


def summarize_entry_from_nas_history(
    entry: dict[str, Any],
    *,
    compact_root: Path,
) -> dict[str, Any]:
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
    idx = df["val_acc"].idxmax() if "val_acc" in df.columns else df.index[-1]
    row = df.loc[idx]
    out["best_epoch"] = int(row.get("epoch", -1))
    for prefix in ("train", "val", "test"):
        key = f"{prefix}_accuracy_overall"
        if key in row and not pd.isna(row[key]):
            out[f"{prefix}_oa_overall"] = float(row[key])
    return out


def write_nas_snapshot_csv(
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
    for row in rows:
        row["model_config"] = best_config_path.stem
        row["model_label"] = config.get("label") or best_config_path.stem
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return rows


def snapshot_all_best_configs(
    *,
    config_dir: Path,
    compact_root: Path,
    output_base: Path,
) -> list[dict[str, Any]]:
    from rsna2024_lumbar.best_config_runs.config import list_best_config_files

    all_rows: list[dict[str, Any]] = []
    for cfg_path in list_best_config_files(config_dir):
        out_csv = (
            output_base
            / cfg_path.stem
            / f"nas_snapshot_{cfg_path.stem}.csv"
        )
        all_rows.extend(
            write_nas_snapshot_csv(
                cfg_path,
                compact_root=compact_root,
                output_csv=out_csv,
            )
        )
    combined = output_base / "nas_snapshots_all_models.csv"
    combined.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(combined, index=False)
    return all_rows
