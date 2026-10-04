"""Copy NAS trial artifacts into a small, git-friendly tree (histories + config JSON)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rsna2024_lumbar.nas.catalog import ArchiveResultTarget
from rsna2024_lumbar.nas.results import load_trial_record, resolve_results_root

HISTORY_FILENAME = "training_history.csv"
CONFIG_FILENAME = "trial_config.json"
SUMMARY_FILENAME = "trial_summary.json"

# Hyperparameters + identifiers from legacy ``result.json`` (metrics live in the CSV).
CONFIG_KEYS_FROM_RESULT = (
    "trial_id",
    "condition",
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
    "final_train_acc",
    "max_train_acc",
    "final_val_acc",
    "max_val_acc",
    "final_test_acc",
    "max_test_acc",
    "val_f1_macro_levels",
    "test_f1_macro_levels",
    "crop_policy",
    "image_source",
    "trainable_params",
    "total_params",
)


@dataclass(frozen=True)
class TrialExportPaths:
    source_dir: Path
    relative_dir: Path
    dest_dir: Path


def compact_relative_dir(trial_dir: Path, results_root: Path) -> Path:
    """
    Short destination path for git/Windows: ``<condition>/trial_XXXX/``.

    Keeps the original trial folder name in ``trial_summary.json`` only.
    """
    trial_dir = trial_dir.resolve()
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition", "unknown"))
        trial_id = int(payload.get("trial_id", 0))
        return Path(condition) / f"trial_{trial_id:04d}"
    rel = trial_dir.relative_to(results_root.resolve())
    return Path(rel.parts[0]) / trial_dir.name[:64]


def iter_trial_dirs_with_history(results_root: Path) -> Iterator[TrialExportPaths]:
    """Yield trial folders under ``results_root`` that contain ``training_history.csv``."""
    results_root = results_root.resolve()
    for history in sorted(results_root.rglob(HISTORY_FILENAME)):
        if not history.is_file():
            continue
        trial_dir = history.parent
        rel = compact_relative_dir(trial_dir, results_root)
        yield TrialExportPaths(
            source_dir=trial_dir,
            relative_dir=rel,
            dest_dir=Path(),  # filled by caller
        )


def trial_config_from_result(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = {key: payload[key] for key in CONFIG_KEYS_FROM_RESULT if key in payload}
    best_metrics = result_path.parent / "best_epoch_metrics.json"
    if best_metrics.is_file():
        bm = json.loads(best_metrics.read_text(encoding="utf-8"))
        metrics = bm.get("metrics") or {}
        if "val_acc" in metrics:
            config["val_acc_at_best_epoch"] = metrics["val_acc"]
        if "val_loss" in metrics:
            config["val_loss_at_best_epoch"] = metrics["val_loss"]
        if "train_acc" in metrics:
            config["train_acc_at_best_epoch"] = metrics["train_acc"]
        if "test_acc" in metrics:
            config["test_acc_at_best_epoch"] = metrics["test_acc"]
        if "train_f1_macro_levels" in metrics:
            config["train_f1_macro_levels_at_best"] = metrics["train_f1_macro_levels"]
    return config


def export_trial_dir(source_trial_dir: Path, dest_trial_dir: Path) -> list[Path]:
    """Copy history CSV and write slim JSON sidecars; return written paths."""
    source_trial_dir = source_trial_dir.resolve()
    dest_trial_dir = dest_trial_dir.resolve()
    dest_trial_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    history_src = source_trial_dir / HISTORY_FILENAME
    if not history_src.is_file():
        raise FileNotFoundError(f"Missing {HISTORY_FILENAME} under {source_trial_dir}")
    history_dest = dest_trial_dir / HISTORY_FILENAME
    shutil.copy2(history_src, history_dest)
    written.append(history_dest)

    result_src = source_trial_dir / "result.json"
    if result_src.is_file():
        config = trial_config_from_result(result_src)
        config_dest = dest_trial_dir / CONFIG_FILENAME
        config_dest.write_text(json.dumps(config, indent=2), encoding="utf-8")
        written.append(config_dest)
        summary_dest = dest_trial_dir / SUMMARY_FILENAME
        summary_dest.write_text(
            json.dumps(
                {
                    "source_trial_dir": source_trial_dir.name,
                    "source_result": str(result_src),
                    "note": "Per-epoch metrics are in training_history.csv; hyperparameters in trial_config.json.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(summary_dest)
    return written


def dest_layout_root(output_root: Path, target: ArchiveResultTarget) -> Path:
    return output_root / target.archive_folder / target.layout


def export_trials(
    *,
    target: ArchiveResultTarget,
    archive_root: Path,
    output_root: Path,
    only_trial_dirs: set[Path] | None = None,
) -> dict[str, object]:
    """
    Copy trials from legacy ``NAS results/<Model>/<layout>/`` into
    ``output_root/<Model>/<layout>/...`` preserving condition/trial folder names.
    """
    source_root = resolve_results_root(
        target.family,
        archive_root=archive_root,
        layout=target.layout,
    )
    layout_dest = dest_layout_root(output_root, target)
    exported = 0
    skipped = 0
    bytes_written = 0

    for entry in iter_trial_dirs_with_history(source_root):
        if only_trial_dirs is not None and entry.source_dir.resolve() not in only_trial_dirs:
            skipped += 1
            continue
        dest = layout_dest / entry.relative_dir
        files = export_trial_dir(entry.source_dir, dest)
        exported += 1
        for path in files:
            bytes_written += path.stat().st_size

    return {
        "label": target.label,
        "family": target.family,
        "layout": target.layout,
        "source_root": str(source_root),
        "dest_root": str(layout_dest),
        "exported_trials": exported,
        "skipped_trials": skipped,
        "bytes_written": bytes_written,
    }


def estimate_history_bytes(archive_root: Path, target: ArchiveResultTarget) -> tuple[int, int]:
    """Return (trial_count, total_bytes) for ``training_history.csv`` under one target."""
    source_root = resolve_results_root(
        target.family,
        archive_root=archive_root,
        layout=target.layout,
    )
    total = 0
    count = 0
    for entry in iter_trial_dirs_with_history(source_root):
        path = entry.source_dir / HISTORY_FILENAME
        total += path.stat().st_size
        count += 1
    return count, total


def load_best_trial_dirs(
    target: ArchiveResultTarget,
    archive_root: Path,
    metric: str,
) -> set[Path]:
    from rsna2024_lumbar.nas.extract import extract_best_for_target

    result = extract_best_for_target(
        target,
        archive_root=archive_root,
        metric=metric,
        include_shared=False,
    )
    dirs: set[Path] = set()
    for record in result.best.values():
        dirs.add(record.result_path.parent.resolve())
    return dirs
