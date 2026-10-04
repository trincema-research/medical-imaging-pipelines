"""Export NAS best-trial summaries to CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rsna2024_lumbar.nas.rank import SharedConfigPick, metric_value
from rsna2024_lumbar.nas.results import TrialRecord
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

CSV_FIELDNAMES: tuple[str, ...] = (
    "selection",
    "family",
    "results_root",
    "rank_metric",
    "rank_metric_value",
    "condition",
    "trial_id",
    "grid_index",
    "model_type",
    "input_layout",
    "variant",
    "image_width",
    "image_height",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "freeze_backbone",
    "use_pretrained",
    "amp",
    "head_depth",
    "head_hidden_dim",
    "head_dropout",
    "head_activation",
    "optimizer_type",
    "scheduler_type",
    "best_epoch",
    "epochs_ran",
    "duration_sec",
    "val_acc",
    "max_val_acc",
    "final_val_acc",
    "val_f1_macro_levels",
    "test_f1_macro_levels",
    "val_loss",
    "trainable_params",
    "total_params",
    "result_path",
    "trial_dir",
    "shared_mean_rank_metric",
)


def default_csv_path(family: str, layout: str | None = None) -> Path:
    parts = ["nas_best", family]
    if layout:
        parts.append(layout)
    return Path("runs") / f"{'_'.join(parts)}.csv"


def _safe_metric(record: TrialRecord, metric: str) -> float | None:
    try:
        return float(metric_value(record, metric))
    except ValueError:
        return None


def trial_record_to_row(
    record: TrialRecord,
    *,
    family: str,
    results_root: str | Path,
    rank_metric: str,
    selection: str = "best_per_condition",
    shared_mean_rank_metric: float | None = None,
) -> dict[str, Any]:
    rank_val = _safe_metric(record, rank_metric)
    return {
        "selection": selection,
        "family": family,
        "results_root": str(results_root),
        "rank_metric": rank_metric,
        "rank_metric_value": rank_val if rank_val is not None else "",
        "condition": record.condition,
        "trial_id": record.trial_id,
        "grid_index": record.grid_index if record.grid_index is not None else "",
        "model_type": record.model_type,
        "input_layout": record.input_layout,
        "variant": record.variant,
        "image_width": record.image_width,
        "image_height": record.image_height,
        "batch_size": record.batch_size,
        "learning_rate": record.learning_rate,
        "weight_decay": record.weight_decay,
        "freeze_backbone": record.freeze_backbone,
        "use_pretrained": record.use_pretrained,
        "amp": record.amp,
        "head_depth": record.head_depth,
        "head_hidden_dim": record.head_hidden_dim if record.head_hidden_dim is not None else "",
        "head_dropout": record.head_dropout,
        "head_activation": record.head_activation,
        "optimizer_type": record.optimizer_type,
        "scheduler_type": record.scheduler_type,
        "best_epoch": record.best_epoch,
        "epochs_ran": record.epochs_ran,
        "duration_sec": record.duration_sec,
        "val_acc": record.val_acc if record.val_acc is not None else "",
        "max_val_acc": record.max_val_acc,
        "final_val_acc": record.final_val_acc,
        "val_f1_macro_levels": record.val_f1_macro_levels,
        "test_f1_macro_levels": record.test_f1_macro_levels,
        "val_loss": record.val_loss if record.val_loss is not None else "",
        "trainable_params": "",
        "total_params": "",
        "result_path": str(record.result_path),
        "trial_dir": record.result_path.parent.name,
        "shared_mean_rank_metric": shared_mean_rank_metric if shared_mean_rank_metric is not None else "",
    }


def _enrich_params_from_result_json(row: dict[str, Any]) -> None:
    """Optional trainable/total params if present in result.json (not on TrialRecord)."""
    import json

    path = Path(str(row["result_path"]))
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "trainable_params" in payload:
        row["trainable_params"] = payload["trainable_params"]
    if "total_params" in payload:
        row["total_params"] = payload["total_params"]


def rows_best_per_condition(
    best: dict[str, TrialRecord],
    *,
    family: str,
    results_root: str | Path,
    rank_metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        record = best.get(condition)
        if record is None:
            continue
        row = trial_record_to_row(
            record,
            family=family,
            results_root=results_root,
            rank_metric=rank_metric,
            selection="best_per_condition",
        )
        _enrich_params_from_result_json(row)
        rows.append(row)
    return rows


def rows_shared_config(
    shared: SharedConfigPick,
    *,
    family: str,
    results_root: str | Path,
    rank_metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        record = shared.per_condition.get(condition)
        if record is None:
            continue
        row = trial_record_to_row(
            record,
            family=family,
            results_root=results_root,
            rank_metric=rank_metric,
            selection="shared_hyperparams",
            shared_mean_rank_metric=shared.mean_score,
        )
        _enrich_params_from_result_json(row)
        rows.append(row)
    return rows


def write_best_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
