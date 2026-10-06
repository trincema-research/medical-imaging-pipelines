"""Export NAS best-trial summaries to CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rsna2024_lumbar.nas.history_metrics import NAS_BEST_EPOCH_CLASSIFICATION_FIELDS
from rsna2024_lumbar.nas.rank import SharedConfigPick, metric_value
from rsna2024_lumbar.nas.results import TrialRecord
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

HYPERPARAMETER_CSV_FIELDS: tuple[str, ...] = (
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
    "crop_policy",
    "image_source",
)

METRIC_CSV_FIELDS: tuple[str, ...] = (
    "best_epoch",
    "epochs_ran",
    "duration_sec",
    "train_acc_at_best_epoch",
    "val_acc",
    "test_acc_at_best_epoch",
    "final_train_acc",
    "max_train_acc",
    "final_val_acc",
    "max_val_acc",
    "final_test_acc",
    "max_test_acc",
    "train_f1_macro_levels_at_best",
    "val_f1_macro_levels",
    "test_f1_macro_levels",
    "val_loss",
    *NAS_BEST_EPOCH_CLASSIFICATION_FIELDS,
)

COMBINED_BEST_CONFIG_STEM = "nas_best_all"
LEGACY_COMBINED_BEST_CONFIG_STEM = "nas_best_vit_maxvit_convnext_all"


def combined_best_config_csv_path(csv_dir: Path) -> Path:
    return csv_dir / f"{COMBINED_BEST_CONFIG_STEM}.csv"


def merge_combined_best_rows(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace rows that share (family, archive_layout, condition); keep other models."""
    new_keys = {
        (str(r.get("family", "")), str(r.get("archive_layout", "")), str(r.get("condition", "")))
        for r in new_rows
    }
    kept = [
        r
        for r in existing
        if (str(r.get("family", "")), str(r.get("archive_layout", "")), str(r.get("condition", "")))
        not in new_keys
    ]
    return kept + new_rows


CSV_FIELDNAMES: tuple[str, ...] = (
    "selection",
    "model_group",
    "archive_layout",
    "archive_label",
    "family",
    "results_root",
    "rank_metric",
    "rank_metric_value",
    "condition",
    "trial_id",
    "grid_index",
    *HYPERPARAMETER_CSV_FIELDS,
    *METRIC_CSV_FIELDS,
    "trainable_params",
    "total_params",
    "checkpoint_path",
    "result_path",
    "trial_dir",
    "shared_mean_rank_metric",
)


def default_csv_path(family: str, layout: str | None = None) -> Path:
    parts = ["nas_best", family]
    if layout:
        parts.append(layout)
    return Path("runs") / f"{'_'.join(parts)}.csv"


def default_json_path(family: str, layout: str | None = None) -> Path:
    parts = ["nas_best", family]
    if layout:
        parts.append(layout)
    return Path("runs") / f"{'_'.join(parts)}.json"


def _safe_metric(record: TrialRecord, metric: str) -> float | None:
    try:
        return float(metric_value(record, metric))
    except ValueError:
        return None


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    return value


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
    row: dict[str, Any] = {
        "selection": selection,
        "model_group": "",
        "archive_layout": "",
        "archive_label": "",
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
        "crop_policy": "",
        "image_source": "",
        "best_epoch": record.best_epoch,
        "epochs_ran": record.epochs_ran,
        "duration_sec": record.duration_sec,
        "train_acc_at_best_epoch": _fmt(record.train_acc_at_best_epoch),
        "val_acc": _fmt(record.val_acc),
        "test_acc_at_best_epoch": _fmt(record.test_acc_at_best_epoch),
        "final_train_acc": record.final_train_acc,
        "max_train_acc": record.max_train_acc,
        "final_val_acc": record.final_val_acc,
        "max_val_acc": record.max_val_acc,
        "final_test_acc": record.final_test_acc,
        "max_test_acc": record.max_test_acc,
        "train_f1_macro_levels_at_best": _fmt(record.train_f1_macro_levels_at_best),
        "val_f1_macro_levels": record.val_f1_macro_levels,
        "test_f1_macro_levels": record.test_f1_macro_levels,
        "val_loss": _fmt(record.val_loss),
        "trainable_params": "",
        "total_params": "",
        "checkpoint_path": "",
        "result_path": str(record.result_path),
        "trial_dir": record.result_path.parent.name,
        "shared_mean_rank_metric": shared_mean_rank_metric if shared_mean_rank_metric is not None else "",
    }
    _enrich_from_result_json(row)
    return row


def _enrich_from_result_json(row: dict[str, Any]) -> None:
    path = Path(str(row["result_path"]))
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "trainable_params" in payload:
        row["trainable_params"] = payload["trainable_params"]
    if "total_params" in payload:
        row["total_params"] = payload["total_params"]
    if "checkpoint_path" in payload:
        row["checkpoint_path"] = payload["checkpoint_path"]
    if payload.get("crop_policy"):
        row["crop_policy"] = payload["crop_policy"]
    if payload.get("image_source"):
        row["image_source"] = payload["image_source"]


def row_to_json_entry(row: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = {key: row.get(key, "") for key in HYPERPARAMETER_CSV_FIELDS}
    metrics = {key: row.get(key, "") for key in METRIC_CSV_FIELDS}
    return {
        "selection": row.get("selection"),
        "model_group": row.get("model_group"),
        "archive_layout": row.get("archive_layout"),
        "archive_label": row.get("archive_label"),
        "family": row.get("family"),
        "condition": row.get("condition"),
        "trial_id": row.get("trial_id"),
        "grid_index": row.get("grid_index"),
        "rank_metric": row.get("rank_metric"),
        "rank_metric_value": row.get("rank_metric_value"),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "trainable_params": row.get("trainable_params"),
        "total_params": row.get("total_params"),
        "checkpoint_path": row.get("checkpoint_path"),
        "result_path": row.get("result_path"),
        "trial_dir": row.get("trial_dir"),
        "shared_mean_rank_metric": row.get("shared_mean_rank_metric"),
    }


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
        rows.append(
            trial_record_to_row(
                record,
                family=family,
                results_root=results_root,
                rank_metric=rank_metric,
                selection="best_per_condition",
            )
        )
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
        rows.append(
            trial_record_to_row(
                record,
                family=family,
                results_root=results_root,
                rank_metric=rank_metric,
                selection="shared_hyperparams",
                shared_mean_rank_metric=shared.mean_score,
            )
        )
    return rows


def write_best_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_best_json(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    rank_metric: str,
    archive_root: str | Path | None = None,
    label: str | None = None,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "rank_metric": rank_metric,
        "archive_root": str(archive_root) if archive_root is not None else "",
        "label": label or "",
        "row_count": len(rows),
        "entries": [row_to_json_entry(row) for row in rows],
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
