"""Load ``result.json`` trials from NAS run trees (legacy or ``runs/lumbar_nas*``)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from rsna2024_lumbar.nas.families import FamilySpec, require_family
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

# Top-level folders under ``runs/NAS results/`` (legacy archive layout).
ARCHIVE_DIR_BY_FAMILY: dict[str, str] = {
    "vit": "ViT",
    "vit3d": "ViT",
    "maxvit": "MaxViT",
    "maxvit3d": "MaxViT",
    "convnext": "Convnext",
    "convnext3d": "Convnext",
    "efficientnet": "EfficientNet",
    "efficientnet3d": "EfficientNet",
}

LAYOUT_SUBDIR_ALIASES: dict[str, tuple[str, ...]] = {
    "2d": ("2d", "2D"),
    "3d": ("3d", "3D"),
}


@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    condition: str
    model_type: str
    input_layout: str
    variant: str
    image_width: int
    image_height: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    freeze_backbone: bool
    use_pretrained: bool
    amp: bool
    head_depth: int
    head_hidden_dim: int | None
    head_dropout: float
    head_activation: str
    optimizer_type: str
    scheduler_type: str
    best_epoch: int
    epochs_ran: int
    duration_sec: float
    final_val_acc: float
    max_val_acc: float
    val_acc: float | None
    val_f1_macro_levels: float
    test_f1_macro_levels: float
    val_loss: float | None
    final_train_acc: float
    max_train_acc: float
    final_test_acc: float
    max_test_acc: float
    train_acc_at_best_epoch: float | None
    test_acc_at_best_epoch: float | None
    train_f1_macro_levels_at_best: float | None
    result_path: Path
    output_dir: str
    grid_index: int | None = None

    def hyperparam_key(self) -> tuple[Any, ...]:
        hidden = self.head_hidden_dim
        return (
            self.variant,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.freeze_backbone,
            self.use_pretrained,
            self.amp,
            self.head_depth,
            hidden,
            self.head_dropout,
            self.head_activation,
            self.optimizer_type,
            self.scheduler_type,
            self.image_width,
            self.image_height,
            self.input_layout,
        )


def default_archive_root() -> Path | None:
    env = os.environ.get("LUMBAR_NAS_RESULTS_ROOT")
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path.resolve()
    candidates = [
        Path(r"F:\datasets_lumbar\rsna-2024-lumbar-spine-degenerative-classification\runs\NAS results"),
        Path("runs/NAS results"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_results_root(
    family: str,
    *,
    results_root: Path | None = None,
    archive_root: Path | None = None,
    layout: str | None = None,
) -> Path:
    """Resolve a directory to scan for ``result.json`` files."""
    if results_root is not None:
        path = Path(results_root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"--results-root is not a directory: {path}")
        return path

    spec = require_family(family)
    base = archive_root or default_archive_root()
    if base is None:
        raise FileNotFoundError(
            "No results directory. Pass --results-root, --archive-root, or set LUMBAR_NAS_RESULTS_ROOT."
        )
    base = Path(base).resolve()
    folder = ARCHIVE_DIR_BY_FAMILY.get(spec.name, spec.name)
    root = base / folder
    if layout:
        layout = layout.strip().lower()
        if layout not in LAYOUT_SUBDIR_ALIASES:
            raise ValueError(f"Unknown layout {layout!r}. Choose 2d or 3d.")
        for name in LAYOUT_SUBDIR_ALIASES[layout]:
            candidate = root / name
            if candidate.is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"No {layout} folder under {root} (tried {LAYOUT_SUBDIR_ALIASES[layout]}).")
    if root.is_dir():
        return root.resolve()
    raise FileNotFoundError(f"Archive folder not found: {root}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_trial_record(result_path: Path) -> TrialRecord:
    result_path = result_path.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    val_loss: float | None = None
    val_acc: float | None = None
    train_acc_at_best: float | None = None
    test_acc_at_best: float | None = None
    train_f1_at_best: float | None = None
    best_path = result_path.parent / "best_epoch_metrics.json"
    if best_path.is_file():
        best_payload = json.loads(best_path.read_text(encoding="utf-8"))
        metrics = best_payload.get("metrics") or {}
        val_loss = _optional_float(metrics.get("val_loss"))
        val_acc = _optional_float(metrics.get("val_acc"))
        train_acc_at_best = _optional_float(metrics.get("train_acc"))
        test_acc_at_best = _optional_float(metrics.get("test_acc"))
        train_f1_at_best = _optional_float(metrics.get("train_f1_macro_levels"))

    hidden = payload.get("head_hidden_dim")
    return TrialRecord(
        trial_id=int(payload["trial_id"]),
        condition=str(payload["condition"]),
        model_type=str(payload.get("model_type", "")),
        input_layout=str(payload.get("input_layout", "2d")),
        variant=str(payload["variant"]),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        freeze_backbone=bool(payload["freeze_backbone"]),
        use_pretrained=bool(payload["use_pretrained"]),
        amp=bool(payload["amp"]),
        head_depth=int(payload["head_depth"]),
        head_hidden_dim=int(hidden) if hidden is not None else None,
        head_dropout=float(payload["head_dropout"]),
        head_activation=str(payload["head_activation"]),
        optimizer_type=str(payload["optimizer_type"]),
        scheduler_type=str(payload["scheduler_type"]),
        best_epoch=int(payload.get("best_epoch", -1)),
        epochs_ran=int(payload.get("epochs_ran", 0)),
        duration_sec=float(payload.get("duration_sec", 0.0)),
        final_val_acc=float(payload.get("final_val_acc", 0.0)),
        max_val_acc=float(payload.get("max_val_acc", 0.0)),
        val_acc=val_acc,
        val_f1_macro_levels=float(payload.get("val_f1_macro_levels", 0.0)),
        test_f1_macro_levels=float(payload.get("test_f1_macro_levels", 0.0)),
        val_loss=val_loss,
        final_train_acc=float(payload.get("final_train_acc", 0.0)),
        max_train_acc=float(payload.get("max_train_acc", 0.0)),
        final_test_acc=float(payload.get("final_test_acc", 0.0)),
        max_test_acc=float(payload.get("max_test_acc", 0.0)),
        train_acc_at_best_epoch=train_acc_at_best,
        test_acc_at_best_epoch=test_acc_at_best,
        train_f1_macro_levels_at_best=train_f1_at_best,
        result_path=result_path,
        output_dir=str(payload.get("output_dir", "")),
    )


def iter_trial_records(root: Path) -> Iterator[TrialRecord]:
    for path in sorted(root.rglob("result.json")):
        if path.is_file():
            yield load_trial_record(path)


def filter_records(
    records: list[TrialRecord],
    *,
    family: str | None = None,
    condition: str | None = None,
    model_type: str | None = None,
) -> list[TrialRecord]:
    spec: FamilySpec | None = require_family(family) if family else None
    expected_type = model_type or (spec.model_type if spec else None)
    out: list[TrialRecord] = []
    for record in records:
        if expected_type and record.model_type != expected_type:
            continue
        if condition and record.condition != condition:
            continue
        out.append(record)
    return out


def attach_grid_indices(
    records: list[TrialRecord],
    *,
    family: str,
    crop_policy: str = "centered",
) -> list[TrialRecord]:
    """Set ``grid_index`` by matching hyperparameters to ``expand_family`` ordering."""
    from rsna2024_lumbar.nas.search_space import expand_family

    by_condition: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)

    indexed: list[TrialRecord] = []
    for condition, group in by_condition.items():
        _spec, _space, trials = expand_family(
            family, condition=condition, crop_policy=crop_policy
        )

        def trial_key(t) -> tuple[Any, ...]:
            hidden = t.head_hidden_dim
            return (
                t.variant,
                t.batch_size,
                t.learning_rate,
                t.weight_decay,
                t.freeze_backbone,
                t.use_pretrained,
                t.amp,
                t.head_depth,
                hidden,
                t.head_dropout,
                t.head_activation,
                t.optimizer_type,
                t.scheduler_type,
                t.image_width,
                t.image_height,
                t.input_layout,
            )

        index_by_key = {trial_key(t): i + 1 for i, t in enumerate(trials)}
        for record in group:
            grid_index = index_by_key.get(record.hyperparam_key())
            indexed.append(replace(record, grid_index=grid_index))
    return indexed


def summarize_scan(root: Path, records: list[TrialRecord]) -> dict[str, Any]:
    conditions = sorted({r.condition for r in records})
    model_types = sorted({r.model_type for r in records})
    return {
        "root": str(root),
        "trial_count": len(records),
        "conditions": conditions,
        "model_types": model_types,
        "expected_conditions": list(CONDITIONS),
    }
