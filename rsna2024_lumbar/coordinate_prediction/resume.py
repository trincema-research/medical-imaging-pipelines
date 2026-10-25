"""Crash-resume helpers for landmark train / NAS.

Severity NAS writes ``training_metrics.csv`` after every epoch
(``train_vit_lumbar.py``) and ``best_config_runs`` treats a run as complete
when that CSV reaches the planned epoch count. Landmark NAS uses the same
files and rule.

A trial is **complete** when ``training_metrics.csv`` has ``epoch >= planned``
or a legacy ``*_history.json`` exists. An in-flight epoch cannot be recovered.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rsna2024_lumbar.coordinate_prediction.constants import LUMBAR_LEVELS
from rsna2024_lumbar.coordinate_prediction.models import LEVEL_METRIC_KEYS

STATE_COMPLETE = "complete"
STATE_INCOMPLETE = "incomplete"
STATE_PENDING = "pending"

TRAINING_METRICS_CSV = "training_metrics.csv"
TRAINING_HISTORY_CSV = "training_history.csv"
EPOCH_FLAT_KEYS = ("n", "loss", "bce", *LEVEL_METRIC_KEYS)

RESUME_HPARAM_KEYS = (
    "condition",
    "backbone",
    "image_size",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "pretrained",
    "all_slices",
)


def history_path(output_dir: Path, condition: str, backbone: str) -> Path:
    return Path(output_dir) / f"{condition}_{backbone}_history.json"


def last_checkpoint_path(output_dir: Path, condition: str, backbone: str) -> Path:
    return Path(output_dir) / f"{condition}_{backbone}_last.pt"


def training_metrics_path(output_dir: Path) -> Path:
    return Path(output_dir) / TRAINING_METRICS_CSV


def training_history_csv_path(output_dir: Path) -> Path:
    return Path(output_dir) / TRAINING_HISTORY_CSV


def _csv_cell(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return ""
    return value


def flatten_epoch_metrics(epoch: int, splits: dict[str, dict]) -> dict[str, Any]:
    """One wide row, same idea as ``train_vit_lumbar`` epoch_history."""
    row: dict[str, Any] = {"epoch": int(epoch)}
    for split, metrics in splits.items():
        metrics = metrics or {}
        for key in EPOCH_FLAT_KEYS:
            row[f"{split}_{key}"] = metrics.get(key)
        nested = metrics.get("levels") or {}
        for level in LUMBAR_LEVELS:
            part = nested.get(level) or {}
            for key in LEVEL_METRIC_KEYS:
                row[f"{split}_{key}_{level}"] = part.get(key, metrics.get(f"{key}_{level}"))
    return row


def append_training_metrics_row(path: Path, row: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    fieldnames = list(row.keys())
    if not write_header:
        with path.open(newline="", encoding="utf-8") as fh:
            existing = csv.DictReader(fh).fieldnames or []
            fieldnames = list(existing)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    return path


def write_epoch_history_csvs(output_dir: Path, row: dict[str, Any]) -> None:
    append_training_metrics_row(training_metrics_path(output_dir), row)
    append_training_metrics_row(training_history_csv_path(output_dir), row)


def reset_epoch_csvs(output_dir: Path) -> None:
    for path in (training_metrics_path(output_dir), training_history_csv_path(output_dir)):
        if path.is_file():
            path.unlink()


def last_logged_epoch(path: Path) -> int:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except Exception:
        return 0
    if frame.empty or "epoch" not in frame.columns:
        return 0
    return int(frame["epoch"].max())


def trial_is_complete(
    output_dir: Path,
    condition: str,
    backbone: str,
    epochs: int | None = None,
) -> bool:
    if history_path(output_dir, condition, backbone).is_file():
        return True
    if epochs is None:
        return False
    logged = last_logged_epoch(training_metrics_path(output_dir))
    if logged == 0:
        logged = last_logged_epoch(training_history_csv_path(output_dir))
    return logged >= int(epochs)


def trial_output_dir(output_base: Path, family: str, condition: str, slug: str) -> Path:
    return Path(output_base) / family / condition / slug


def trial_has_started(output_dir: Path, condition: str, backbone: str) -> bool:
    out = Path(output_dir)
    if not out.exists():
        return False
    markers = (
        last_checkpoint_path(out, condition, backbone),
        training_metrics_path(out),
        training_history_csv_path(out),
        out / f"{condition}_{backbone}.pt",
        out / f"{condition}_{backbone}_metrics.json",
        out / "trial.json",
    )
    if any(path.is_file() for path in markers):
        return True
    return out.is_dir() and any(out.iterdir())


def classify_trial(
    output_dir: Path,
    condition: str,
    backbone: str,
    epochs: int | None = None,
) -> str:
    if trial_is_complete(output_dir, condition, backbone, epochs=epochs):
        return STATE_COMPLETE
    if trial_has_started(output_dir, condition, backbone):
        return STATE_INCOMPLETE
    return STATE_PENDING


@dataclass(frozen=True)
class TrialRunStatus:
    trial_id: int
    slug: str
    condition: str
    state: str
    output_dir: Path


def collect_trial_statuses(
    trials: Iterable[Any],
    *,
    output_base: Path,
    family: str,
) -> list[TrialRunStatus]:
    rows: list[TrialRunStatus] = []
    for trial in trials:
        out = trial_output_dir(output_base, family, trial.condition, trial.slug)
        rows.append(
            TrialRunStatus(
                trial_id=int(trial.trial_id),
                slug=str(trial.slug),
                condition=str(trial.condition),
                state=classify_trial(
                    out, trial.condition, trial.backbone, epochs=getattr(trial, "epochs", None)
                ),
                output_dir=out,
            )
        )
    return rows


def format_id_ranges(ids: Iterable[int]) -> str:
    ordered = sorted({int(value) for value in ids})
    if not ordered:
        return "-"
    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def format_status_report(
    rows: list[TrialRunStatus],
    *,
    family: str,
    output_base: Path,
) -> str:
    lines = [f"family={family}  output={Path(output_base)}"]
    conditions = list(dict.fromkeys(row.condition for row in rows))
    for condition in conditions:
        part = [row for row in rows if row.condition == condition]
        complete = [row for row in part if row.state == STATE_COMPLETE]
        incomplete = [row for row in part if row.state == STATE_INCOMPLETE]
        pending = [row for row in part if row.state == STATE_PENDING]
        lines.append(
            f"{condition}  {len(complete)}/{len(part)} complete  "
            f"{len(incomplete)} incomplete  {len(pending)} pending"
        )
        lines.append(f"  complete    {format_id_ranges(row.trial_id for row in complete)}")
        if incomplete:
            for row in incomplete:
                lines.append(f"  incomplete  {row.trial_id}  {row.slug}")
        else:
            lines.append("  incomplete  -")
        lines.append(f"  pending     {format_id_ranges(row.trial_id for row in pending)}")
    return "\n".join(lines)


def resume_hparams(
    *,
    condition: str,
    backbone: str,
    image_size: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    pretrained: bool,
    all_slices: bool,
) -> dict[str, Any]:
    return {
        "condition": str(condition),
        "backbone": str(backbone),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "pretrained": bool(pretrained),
        "all_slices": bool(all_slices),
    }


def resume_hparams_match(saved: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if not saved:
        return False
    return all(saved.get(key) == current.get(key) for key in RESUME_HPARAM_KEYS)


def next_epoch(last_completed: int) -> int:
    start = int(last_completed) + 1
    return 1 if start < 1 else start


def wants_resume(args: Any) -> bool:
    return not bool(getattr(args, "no_resume", False))


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
