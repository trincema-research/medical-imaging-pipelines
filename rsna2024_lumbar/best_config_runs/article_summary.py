"""Mean ± std over repeats at best validation epoch (for article tables)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.config import list_best_config_files
from rsna2024_lumbar.best_config_runs.results import (
    PIPELINE_RESULTS_COLUMNS,
    load_combined_pipeline_results,
    write_pipeline_results_csv,
)

GROUP_KEYS: tuple[str, ...] = (
    "model_config",
    "model_label",
    "family",
    "archive_layout",
    "condition",
)

# Metrics taken from the epoch with highest val_acc in each repeat (see pipeline_results.csv).
ARTICLE_METRIC_COLUMNS: tuple[str, ...] = tuple(
    c
    for c in PIPELINE_RESULTS_COLUMNS
    if c
    not in (
        *GROUP_KEYS,
        "repeat_index",
        "split_seed",
        "nas_trial_id",
        "best_epoch",
        "run_dir",
    )
)


def _format_pm(mean: float, std: float, *, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def aggregate_pipeline_results(
    df: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """One row per model × condition with mean, std, and formatted mean ± std."""
    if df.empty:
        return pd.DataFrame()
    if metric_columns is not None:
        metrics = list(metric_columns)
    else:
        metrics = [c for c in ARTICLE_METRIC_COLUMNS if c in df.columns]
    if not metrics:
        raise ValueError("No metric columns to aggregate")

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(list(GROUP_KEYS), dropna=False, sort=True)
    for key_tuple, group in grouped:
        key_vals = dict(zip(GROUP_KEYS, key_tuple))
        n = len(group)
        row: dict[str, Any] = {**key_vals, "n_repeats": n}
        for col in metrics:
            vals = pd.to_numeric(group[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{col}_mean"] = ""
                row[f"{col}_std"] = ""
                row[f"{col}_pm"] = ""
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_pm"] = _format_pm(mean, std, decimals=decimals)
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_pipeline_results(output_base: Path) -> pd.DataFrame:
    combined = load_combined_pipeline_results(output_base)
    if combined:
        return pd.DataFrame(combined)
    rows: list[dict[str, Any]] = []
    for path in sorted(output_base.glob("nas_best_*/pipeline_results.csv")):
        part = pd.read_csv(path)
        if part.empty:
            continue
        rows.extend(part.to_dict(orient="records"))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def write_article_summaries(
    output_base: Path,
    *,
    decimals: int = 4,
) -> tuple[Path, Path]:
    """
    Write article tables under ``output_base``.

    Returns paths to (by_condition, metrics_long).
    """
    df = load_all_pipeline_results(output_base)
    if df.empty:
        raise ValueError(f"No pipeline_results.csv under {output_base}")

    by_condition = aggregate_pipeline_results(df, decimals=decimals)
    out_wide = output_base / "article_summary_by_condition.csv"
    by_condition.to_csv(out_wide, index=False)

    long_rows: list[dict[str, Any]] = []
    for _, row in by_condition.iterrows():
        base = {k: row[k] for k in (*GROUP_KEYS, "n_repeats")}
        for col in ARTICLE_METRIC_COLUMNS:
            pm = row.get(f"{col}_pm", "")
            if pm == "" or pd.isna(pm):
                continue
            long_rows.append(
                {
                    **base,
                    "metric": col,
                    "mean": row.get(f"{col}_mean", ""),
                    "std": row.get(f"{col}_std", ""),
                    "mean_pm_std": pm,
                }
            )
    long_df = pd.DataFrame(long_rows)
    out_long = output_base / "article_summary_metrics_long.csv"
    long_df.to_csv(out_long, index=False)

    # Refresh combined pipeline CSV when per-model files exist but combined is stale.
    combined_path = output_base / "pipeline_results_all_models.csv"
    if not combined_path.is_file() or combined_path.stat().st_mtime < out_wide.stat().st_mtime:
        write_pipeline_results_csv(df.to_dict(orient="records"), combined_path)

    return out_wide, out_long


def refresh_from_config_dir(output_base: Path, config_dir: Path) -> pd.DataFrame:
    """Load every ``nas_best_*/pipeline_results.csv`` present (for tests)."""
    del config_dir
    return load_all_pipeline_results(output_base)
