"""Standard CSV results for train / val / test at best validation epoch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_RESULTS_COLUMNS: tuple[str, ...] = (
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
    "train_accuracy_overall",
    "val_accuracy_overall",
    "test_accuracy_overall",
    "train_f1_macro_overall",
    "val_f1_macro_overall",
    "test_f1_macro_overall",
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


def _get(row: pd.Series, key: str) -> Any:
    if key not in row.index or pd.isna(row[key]):
        return ""
    return row[key]


def row_from_training_metrics(
    training_metrics_path: Path,
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    df = pd.read_csv(training_metrics_path)
    if df.empty:
        raise ValueError(f"Empty training metrics: {training_metrics_path}")
    idx = df["val_acc"].idxmax() if "val_acc" in df.columns else df.index[-1]
    row = df.loc[idx]

    out: dict[str, Any] = {k: meta.get(k, "") for k in PIPELINE_RESULTS_COLUMNS}
    out.update(meta)
    out["best_epoch"] = int(_get(row, "epoch"))
    out["train_acc"] = _get(row, "train_acc")
    out["val_acc"] = _get(row, "val_acc")
    out["test_acc"] = _get(row, "test_acc")
    for split in ("train", "val", "test"):
        out[f"{split}_accuracy_overall"] = _get(row, f"{split}_accuracy_overall")
        out[f"{split}_f1_macro_overall"] = _get(row, f"{split}_f1_macro_overall")
        out[f"{split}_oa_overall"] = _get(row, f"{split}_oa_overall")
        out[f"{split}_omae_overall"] = _get(row, f"{split}_omae_overall")
        out[f"{split}_qwk_overall"] = _get(row, f"{split}_qwk_overall")
        out[f"{split}_ser_overall"] = _get(row, f"{split}_ser_overall")
    out["run_dir"] = str(training_metrics_path.parent)
    return {k: out.get(k, "") for k in PIPELINE_RESULTS_COLUMNS}


def write_pipeline_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(PIPELINE_RESULTS_COLUMNS)).to_csv(path, index=False)


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
