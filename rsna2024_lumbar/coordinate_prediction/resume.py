"""Crash-resume helpers for landmark train / NAS.

A trial is complete when ``{condition}_{backbone}_history.json`` exists.
An interrupted trial keeps ``{condition}_{backbone}_last.pt`` after each
finished epoch so the same command can continue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
