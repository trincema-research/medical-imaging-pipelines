"""Export ordinal metrics from training CSVs into benchmark tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ORDINAL_SUFFIXES = ("oa_overall", "omae_overall", "qwk_overall", "ser_overall")


def ordinal_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for prefix in ("train", "val", "test"):
        for suffix in ORDINAL_SUFFIXES:
            key = f"{prefix}_{suffix}"
            if key in df.columns:
                cols.append(key)
    return cols


def epoch_history_to_ordinal_csv(
    training_metrics_path: Path,
    output_path: Path,
) -> None:
    """Write per-epoch ordinal columns only (subset of training_metrics.csv)."""
    df = pd.read_csv(training_metrics_path)
    keep = ["epoch"] + ordinal_columns(df)
    keep = [c for c in keep if c in df.columns]
    if len(keep) <= 1:
        raise ValueError(
            f"No ordinal columns in {training_metrics_path}. "
            "Set RSNA2024_ORDINAL_METRICS=1 when training."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(output_path, index=False)


def best_epoch_ordinal_row(
    training_metrics_path: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(training_metrics_path)
    if df.empty:
        raise ValueError(f"Empty metrics: {training_metrics_path}")
    idx = df["val_acc"].idxmax() if "val_acc" in df.columns else df.index[-1]
    row = df.loc[idx].to_dict()
    out: dict[str, Any] = dict(meta or {})
    out["best_epoch"] = int(row.get("epoch", -1))
    for col in ordinal_columns(df):
        out[col] = row[col]
    if "val_acc" in row:
        out["val_acc_at_best"] = row["val_acc"]
    return out
