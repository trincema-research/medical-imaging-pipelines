"""Synthetic NAS trial trees for unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path


def write_nas_result(
    dest: Path,
    *,
    trial_id: int,
    condition: str,
    val_f1: float = 0.5,
    variant: str = "vit_b_16",
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_backbone: bool = False,
    use_pretrained: bool = True,
    amp: bool = True,
    head_depth: int = 1,
    head_hidden_dim: int | None = None,
    head_dropout: float = 0.0,
    head_activation: str = "gelu",
    optimizer_type: str = "AdamW",
    scheduler_type: str = "CosineAnnealingLR",
    image_width: int = 64,
    image_height: int = 64,
    model_type: str = "vit",
    input_layout: str = "2d",
    max_val_acc: float | None = None,
    val_acc_at_best: float | None = None,
    val_loss_at_best: float | None = None,
    include_best_metrics: bool = True,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if max_val_acc is None:
        max_val_acc = 0.5 + val_f1 * 0.1
    if val_acc_at_best is None:
        val_acc_at_best = max_val_acc
    if val_loss_at_best is None:
        val_loss_at_best = 1.0 - val_f1

    payload = {
        "trial_id": trial_id,
        "condition": condition,
        "variant": variant,
        "image_width": image_width,
        "image_height": image_height,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "freeze_backbone": freeze_backbone,
        "use_pretrained": use_pretrained,
        "amp": amp,
        "head_depth": head_depth,
        "head_hidden_dim": head_hidden_dim,
        "head_dropout": head_dropout,
        "head_activation": head_activation,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "final_train_acc": 0.55,
        "final_val_acc": 0.5,
        "final_test_acc": 0.52,
        "max_train_acc": 0.6,
        "max_val_acc": max_val_acc,
        "max_test_acc": 0.53,
        "crop_policy": "centered",
        "image_source": "png",
        "best_epoch": 1,
        "duration_sec": 1.0,
        "output_dir": str(dest),
        "trainable_params": 100,
        "total_params": 200,
        "epochs_ran": 1,
        "val_f1_macro_levels": val_f1,
        "test_f1_macro_levels": val_f1,
        "checkpoint_path": "",
        "model_type": model_type,
        "input_layout": input_layout,
    }
    (dest / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    if include_best_metrics:
        (dest / "best_epoch_metrics.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "val_loss": val_loss_at_best,
                        "val_acc": val_acc_at_best,
                        "train_acc": 0.58,
                        "test_acc": 0.51,
                        "train_f1_macro_levels": 0.49,
                    }
                }
            ),
            encoding="utf-8",
        )


def write_nas_result_at_grid_index(
    dest: Path,
    *,
    family: str,
    condition: str,
    trial_id: int,
    grid_index: int = 0,
    **kwargs: object,
) -> None:
    """Write ``result.json`` hyperparameters that match ``expand_family`` ordering."""
    from rsna2024_lumbar.nas.search_space import expand_family

    _spec, _space, trials = expand_family(family, condition=condition)
    t = trials[grid_index]
    write_nas_result(
        dest,
        trial_id=trial_id,
        condition=condition,
        variant=t.variant,
        batch_size=t.batch_size,
        lr=t.learning_rate,
        weight_decay=t.weight_decay,
        freeze_backbone=t.freeze_backbone,
        use_pretrained=t.use_pretrained,
        amp=t.amp,
        head_depth=t.head_depth,
        head_hidden_dim=t.head_hidden_dim,
        head_dropout=t.head_dropout,
        head_activation=t.head_activation,
        optimizer_type=t.optimizer_type,
        scheduler_type=t.scheduler_type,
        image_width=t.image_width,
        image_height=t.image_height,
        model_type=t.model_type,
        input_layout=t.input_layout,
        **kwargs,  # type: ignore[arg-type]
    )


def write_legacy_archive_trial(
    archive_root: Path,
    *,
    model_folder: str,
    layout: str,
    condition: str,
    trial_dir_name: str,
    trial_id: int,
    **kwargs: object,
) -> Path:
    """Layout: ``NAS results/<Model>/<layout>/<condition>/<trial>/``."""
    root = archive_root / model_folder / layout / condition / trial_dir_name
    write_nas_result(root, trial_id=trial_id, condition=condition, **kwargs)  # type: ignore[arg-type]
    return root
