"""Standard CSV results for train / val / test at best validation epoch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rsna2024_lumbar.best_config_runs.severity_backfill import backfill_severity_on_series
from rsna2024_lumbar.preprocessing.constants import LUMBAR_LEVELS


def spinal_level_metric_columns() -> tuple[str, ...]:
    """Per-level columns logged in training_metrics.csv (classical + ordinal)."""
    cols: list[str] = []
    classical = ("accuracy", "f1_macro", "precision_macro", "recall_macro")
    ordinal = ("oa", "omae", "qwk", "ser")
    for split in ("train", "val", "test"):
        for level in LUMBAR_LEVELS:
            for metric in classical:
                cols.append(f"{split}_{metric}_{level}")
            sev = level.upper()
            for metric in ordinal:
                cols.append(f"{split}_{metric}_{sev}")
    return tuple(cols)


PIPELINE_RESULTS_BASE_COLUMNS: tuple[str, ...] = (
    "model_config",
    "model_label",
    "family",
    "archive_layout",
    "condition",
    "repeat_index",
    "split_seed",
    "nas_trial_id",
    "best_epoch",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_loss",
    "val_loss",
    "test_loss",
    "train_accuracy_macro_levels",
    "val_accuracy_macro_levels",
    "test_accuracy_macro_levels",
    "train_f1_macro_levels",
    "val_f1_macro_levels",
    "test_f1_macro_levels",
    "train_precision_macro_levels",
    "val_precision_macro_levels",
    "test_precision_macro_levels",
    "train_recall_macro_levels",
    "val_recall_macro_levels",
    "test_recall_macro_levels",
    "train_accuracy_overall",
    "val_accuracy_overall",
    "test_accuracy_overall",
    "train_f1_macro_overall",
    "val_f1_macro_overall",
    "test_f1_macro_overall",
    "train_precision_macro_overall",
    "val_precision_macro_overall",
    "test_precision_macro_overall",
    "train_recall_macro_overall",
    "val_recall_macro_overall",
    "test_recall_macro_overall",
    "train_oa_overall",
    "train_omae_overall",
    "train_qwk_overall",
    "train_ser_overall",
    "val_oa_overall",
    "val_omae_overall",
    "val_qwk_overall",
    "val_ser_overall",
    "test_oa_overall",
    "test_omae_overall",
    "test_qwk_overall",
    "test_ser_overall",
    "run_dir",
)
PIPELINE_RESULTS_SPINAL_COLUMNS: tuple[str, ...] = spinal_level_metric_columns()
PIPELINE_RESULTS_COLUMNS: tuple[str, ...] = (
    PIPELINE_RESULTS_BASE_COLUMNS + PIPELINE_RESULTS_SPINAL_COLUMNS
)


def _get(row: pd.Series, key: str) -> Any:
    if key not in row.index or pd.isna(row[key]):
        return ""
    return row[key]


def row_from_training_metrics(
    training_metrics_path: Path,
    *,
    meta: dict[str, Any],
    repeat_dir: Path | None = None,
    legacy_severity_backfill: bool = False,
) -> dict[str, Any]:
    df = pd.read_csv(training_metrics_path)
    if df.empty:
        raise ValueError(f"Empty training metrics: {training_metrics_path}")
    idx = df["val_acc"].idxmax() if "val_acc" in df.columns else df.index[-1]
    row = backfill_severity_on_series(
        df.loc[idx],
        repeat_dir=repeat_dir or training_metrics_path.parent,
        legacy=legacy_severity_backfill,
    )

    out: dict[str, Any] = {k: meta.get(k, "") for k in PIPELINE_RESULTS_COLUMNS}
    out.update(meta)
    out["best_epoch"] = int(_get(row, "epoch"))
    out["train_acc"] = _get(row, "train_acc")
    out["val_acc"] = _get(row, "val_acc")
    out["test_acc"] = _get(row, "test_acc")
    for split in ("train", "val", "test"):
        out[f"{split}_loss"] = _get(row, f"{split}_loss")
        out[f"{split}_accuracy_macro_levels"] = _get(row, f"{split}_accuracy_macro_levels")
        out[f"{split}_f1_macro_levels"] = _get(row, f"{split}_f1_macro_levels")
        out[f"{split}_precision_macro_levels"] = _get(row, f"{split}_precision_macro_levels")
        out[f"{split}_recall_macro_levels"] = _get(row, f"{split}_recall_macro_levels")
        out[f"{split}_accuracy_overall"] = _get(row, f"{split}_accuracy_overall")
        out[f"{split}_f1_macro_overall"] = _get(row, f"{split}_f1_macro_overall")
        out[f"{split}_precision_macro_overall"] = _get(row, f"{split}_precision_macro_overall")
        out[f"{split}_recall_macro_overall"] = _get(row, f"{split}_recall_macro_overall")
        out[f"{split}_oa_overall"] = _get(row, f"{split}_oa_overall")
        out[f"{split}_omae_overall"] = _get(row, f"{split}_omae_overall")
        out[f"{split}_qwk_overall"] = _get(row, f"{split}_qwk_overall")
        out[f"{split}_ser_overall"] = _get(row, f"{split}_ser_overall")
    for col in PIPELINE_RESULTS_SPINAL_COLUMNS:
        out[col] = _get(row, col)
    out["run_dir"] = str(training_metrics_path.parent)
    return {k: out.get(k, "") for k in PIPELINE_RESULTS_COLUMNS}


def write_pipeline_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(PIPELINE_RESULTS_COLUMNS)).to_csv(path, index=False)


def load_combined_pipeline_results(output_base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_base.glob("nas_best_*/pipeline_results.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.extend(df.to_dict(orient="records"))
    return rows


def write_per_run_best_epoch_csv(
    training_metrics_path: Path,
    output_path: Path,
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    row = row_from_training_metrics(training_metrics_path, meta=meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=list(PIPELINE_RESULTS_COLUMNS)).to_csv(
        output_path, index=False
    )
    return row
