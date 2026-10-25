"""Crash-resume helpers for landmark train / NAS.

A trial is **complete** when ``{condition}_{backbone}_history.json`` exists
(written only after the full epoch loop finishes). Anything else on disk
without that file is **incomplete**. No folder means **pending**.

We cannot recover an epoch that died mid-step. Re-running the same command
skips complete trials and retries incomplete / pending ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STATE_COMPLETE = "complete"
STATE_INCOMPLETE = "incomplete"
STATE_PENDING = "pending"

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


def trial_is_complete(output_dir: Path, condition: str, backbone: str) -> bool:
    return history_path(output_dir, condition, backbone).is_file()


def trial_output_dir(output_base: Path, family: str, condition: str, slug: str) -> Path:
    return Path(output_base) / family / condition / slug


def trial_has_started(output_dir: Path, condition: str, backbone: str) -> bool:
    out = Path(output_dir)
    if not out.exists():
        return False
    markers = (
        last_checkpoint_path(out, condition, backbone),
        out / f"{condition}_{backbone}.pt",
        out / f"{condition}_{backbone}_metrics.json",
        out / "trial.json",
    )
    if any(path.is_file() for path in markers):
        return True
    return out.is_dir() and any(out.iterdir())


def classify_trial(output_dir: Path, condition: str, backbone: str) -> str:
    if trial_is_complete(output_dir, condition, backbone):
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
                state=classify_trial(out, trial.condition, trial.backbone),
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
