#!/usr/bin/env python
"""
Train a five-head lumbar classifier on RSNA 2024 MR coordinate crops.

Model families (``--model-type``):

**2D** — five ROI crops ``(B, 5, 3, H, W)``: ``vit``, ``convnext``, ``efficientnet``,
``avengers``, ``clam``, ``maxvit``, ``medsam``

**3D** — volume ``(B, 3, D, H, W)``: ``vit3d``, ``convnext3d``, ``efficientnet3d``,
``avengers3d``, ``clam3d``, ``maxvit3d``

Usage::

    python train_vit_lumbar.py --condition spinal_canal_stenosis --model-type vit --epochs 50

    python train_vit_lumbar.py --model-type convnext --condition spinal_canal_stenosis --save-checkpoints

    python train_vit_lumbar.py --model-type convnext3d --input-layout 3d_level_stack --save-checkpoints
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lumbar_constants import (
    CONDITION_SPECS,
    CROP_POLICIES,
    CROP_POLICY_CENTERED,
    DEFAULT_VOLUME_DEPTH_SERIES,
    IGNORE_LABEL,
    IMAGE_SOURCE_DICOM,
    IMAGE_SOURCES,
    INPUT_LAYOUT_2D,
    INPUT_LAYOUT_3D_LEVEL_STACK,
    INPUT_LAYOUTS,
    LUMBAR_LEVELS,
    LUMBAR_LEVEL_DISPLAY,
    LUMBAR_MODEL_TYPES,
    MODEL_TYPE_AVENGERS,
    MODEL_TYPE_AVENGERS3D,
    MODEL_TYPE_CLAM,
    MODEL_TYPE_CLAM3D,
    MODEL_TYPE_CONVNEXT,
    MODEL_TYPE_CONVNEXT3D,
    MODEL_TYPE_EFFICIENTNET,
    MODEL_TYPE_EFFICIENTNET3D,
    MODEL_TYPE_MAXVIT,
    MODEL_TYPE_MAXVIT3D,
    MODEL_TYPE_MEDSAM,
    MODEL_TYPE_VIT,
    MODEL_TYPE_VIT3D,
    SEVERITY_CLASSES,
    VIT3D_VARIANTS,
    VIT_VARIANTS,
    CNN3D_VARIANTS,
    CropSettings,
    ImageSize,
    describe_crop_policy,
    describe_image_size,
    describe_training_mode,
    resolve_crop_settings,
    resolve_image_size,
    resolve_training_crops_root,
)
from lumbar_dataset import (
    build_lumbar_dataloaders,
    log_lumbar_dataset_split,
    validate_split_fractions,
)
from lumbar_model_factory import (
    build_lumbar_model,
    model_type_run_dir,
    pretrained_weights_for_model,
    resolve_default_backbone_variant,
    validate_backbone_variant,
)
from lumbar_pipeline import describe_pipeline, resolve_pipeline_from_training_args
from train_vit import create_grad_scaler, iter_with_progress, save_confusion_matrix_artifacts, set_seed


def multi_head_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: (B, 5, C)
    targets: (B, 5) with IGNORE_LABEL entries masked out.
    """
    batch_size, num_levels, num_classes = logits.shape
    logits_flat = logits.reshape(batch_size * num_levels, num_classes)
    targets_flat = targets.reshape(batch_size * num_levels)
    valid = targets_flat != IGNORE_LABEL
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(
        logits_flat[valid],
        targets_flat[valid],
        weight=class_weights,
    )


@torch.no_grad()
def multi_head_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    valid = targets != IGNORE_LABEL
    if not valid.any():
        return 0.0
    return float((preds[valid] == targets[valid]).float().mean().item())


def compute_multi_head_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    logits: (N, 5, C), targets: (N, 5)
    """
    metrics: Dict[str, float] = {}
    num_levels = logits.shape[1]
    level_accs = []
    level_f1s = []

    for level_idx in range(num_levels):
        y_true = targets[:, level_idx]
        y_pred = logits[:, level_idx].argmax(axis=-1)
        valid = y_true != IGNORE_LABEL
        if not valid.any():
            metrics[f"accuracy_{LUMBAR_LEVELS[level_idx]}"] = 0.0
            metrics[f"f1_macro_{LUMBAR_LEVELS[level_idx]}"] = 0.0
            continue
        acc = accuracy_score(y_true[valid], y_pred[valid])
        f1 = f1_score(
            y_true[valid],
            y_pred[valid],
            average="macro",
            labels=list(range(len(SEVERITY_CLASSES))),
            zero_division=0,
        )
        metrics[f"accuracy_{LUMBAR_LEVELS[level_idx]}"] = float(acc)
        metrics[f"f1_macro_{LUMBAR_LEVELS[level_idx]}"] = float(f1)
        level_accs.append(acc)
        level_f1s.append(f1)

    metrics["accuracy_macro_levels"] = float(np.mean(level_accs)) if level_accs else 0.0
    metrics["f1_macro_levels"] = float(np.mean(level_f1s)) if level_f1s else 0.0

    all_true = targets[targets != IGNORE_LABEL]
    all_pred = logits.reshape(-1, logits.shape[-1]).argmax(axis=-1)
    all_valid = targets.reshape(-1) != IGNORE_LABEL
    if all_valid.any():
        metrics["accuracy_overall"] = float(
            accuracy_score(all_true, all_pred[all_valid])
        )
        metrics["f1_macro_overall"] = float(
            f1_score(
                all_true,
                all_pred[all_valid],
                average="macro",
                labels=list(range(len(SEVERITY_CLASSES))),
                zero_division=0,
            )
        )
    else:
        metrics["accuracy_overall"] = 0.0
        metrics["f1_macro_overall"] = 0.0
    return metrics


def merge_multi_head_metrics(
    epoch_history: Dict[str, Any],
    prefix: str,
    metrics: Dict[str, float],
) -> None:
    """Add prefixed keys from compute_multi_head_metrics into epoch_history."""
    for key, value in metrics.items():
        epoch_history[f"{prefix}_{key}"] = value


def save_per_level_confusion_matrices(
    logits: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    *,
    split: str,
    epoch: int,
    val_acc: float,
    condition: str | None = None,
) -> None:
    """
    Save one CSV + PNG confusion matrix per lumbar level.

    Uses fixed filenames (overwrites prior best) to limit disk use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_levels = logits.shape[1]
    level_summaries: List[Dict[str, Any]] = []

    for level_idx in range(num_levels):
        level_key = LUMBAR_LEVELS[level_idx]
        level_label = LUMBAR_LEVEL_DISPLAY.get(level_key, level_key)
        y_true = targets[:, level_idx]
        y_pred = logits[:, level_idx].argmax(axis=-1)
        valid = y_true != IGNORE_LABEL
        if not valid.any():
            continue

        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]
        stem = f"confusion_matrix_{split}_{level_key}"
        title_parts = [split.upper(), level_label]
        if condition:
            title_parts.insert(0, condition)
        title = " | ".join(title_parts) + f" (epoch {epoch}, acc={val_acc:.4f})"

        save_confusion_matrix_artifacts(
            y_true_v,
            y_pred_v,
            SEVERITY_CLASSES,
            output_dir / f"{stem}.csv",
            output_dir / f"{stem}.png",
            title,
        )
        level_summaries.append(
            {
                "level": level_key,
                "level_display": level_label,
                "csv": f"{stem}.csv",
                "png": f"{stem}.png",
                "n_samples": int(valid.sum()),
            }
        )

    meta = {
        "split": split,
        "epoch": epoch,
        "val_acc": float(val_acc),
        "condition": condition,
        "severity_classes": SEVERITY_CLASSES,
        "levels": level_summaries,
    }
    with (output_dir / f"confusion_matrices_{split}.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def format_lumbar_config_slug(
    *,
    condition: str,
    model_type: str,
    variant: str,
    image_size: ImageSize,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    freeze_backbone: bool,
    use_pretrained: bool,
    amp: bool,
    head_depth: int = 1,
    head_hidden_dim: int | None = None,
    crop_policy: str = CROP_POLICY_CENTERED,
    image_source: str = IMAGE_SOURCE_DICOM,
    input_layout: str = INPUT_LAYOUT_2D,
) -> str:
    """Filesystem-safe slug describing one lumbar training configuration."""
    layout_part = f"_layout{input_layout}" if input_layout != INPUT_LAYOUT_2D else ""
    hidden_part = f"_hd{head_hidden_dim}" if head_hidden_dim is not None else ""
    return (
        f"{condition}_{model_type}_{variant}_img{image_size.slug()}_bs{batch_size}_"
        f"lr{learning_rate:g}_wd{weight_decay:g}_head{head_depth}{hidden_part}{layout_part}_"
        f"crop{crop_policy}_src{image_source}_"
        f"{'freeze' if freeze_backbone else 'finetune'}_"
        f"{'pre' if use_pretrained else 'scratch'}_"
        f"{'amp' if amp else 'fp32'}"
    )


def resolve_training_output_dir(args: argparse.Namespace) -> Path:
    """
    Auto layout: runs/lumbar_<model_type>/<condition>/<config_slug>/
    Override with --output-dir when set.
    """
    if args.output_dir is not None:
        return Path(args.output_dir)
    slug = format_lumbar_config_slug(
        condition=args.condition,
        model_type=args.model_type,
        variant=args.backbone_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        use_pretrained=not args.no_pretrained,
        amp=args.amp,
        head_depth=args.head_depth,
        head_hidden_dim=args.head_hidden_dim,
        crop_policy=getattr(args, "crop_policy", CROP_POLICY_CENTERED),
        image_source=getattr(args, "image_source", IMAGE_SOURCE_DICOM),
        input_layout=args.input_layout,
    )
    return Path("runs") / model_type_run_dir(args.model_type) / args.condition / slug


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
    *,
    show_batch_progress: bool = False,
    batch_progress_desc: str = "Train batches",
    log_interval: int = 0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    seen = 0
    loop_start = time.time()
    num_batches = len(loader)

    for batch_idx, (images, targets, _study_ids) in enumerate(
        iter_with_progress(loader, enabled=show_batch_progress, desc=batch_progress_desc),
        start=1,
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda")
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += multi_head_accuracy(outputs, targets) * batch_size
        seen += batch_size

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.time() - loop_start
            eta = (elapsed / batch_idx) * (num_batches - batch_idx)
            print(
                f"  {batch_progress_desc} [{batch_idx}/{num_batches}] "
                f"avg_acc={running_acc / max(seen, 1):.4f} avg_loss={running_loss / max(seen, 1):.4f} "
                f"elapsed={_format_mmss(elapsed)} eta={_format_mmss(eta)}",
                flush=True,
            )

    return running_loss / max(seen, 1), running_acc / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    *,
    show_batch_progress: bool = False,
    batch_progress_desc: str = "Eval batches",
    log_interval: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    seen = 0
    all_logits = []
    all_targets = []
    loop_start = time.time()
    num_batches = len(loader)

    for batch_idx, (images, targets, _study_ids) in enumerate(
        iter_with_progress(loader, enabled=show_batch_progress, desc=batch_progress_desc),
        start=1,
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += multi_head_accuracy(outputs, targets) * batch_size
        seen += batch_size
        all_logits.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.time() - loop_start
            eta = (elapsed / batch_idx) * (num_batches - batch_idx)
            print(
                f"  {batch_progress_desc} [{batch_idx}/{num_batches}] "
                f"avg_acc={running_acc / max(seen, 1):.4f} avg_loss={running_loss / max(seen, 1):.4f} "
                f"elapsed={_format_mmss(elapsed)} eta={_format_mmss(eta)}",
                flush=True,
            )

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 5, 3))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 5))
    return running_loss / max(seen, 1), running_acc / max(seen, 1), logits, targets_np


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    condition_key: str,
    output_dir: Path,
    filename: str,
    *,
    config: Dict[str, Any] | None = None,
    val_acc: float | None = None,
    best_metric: str | None = None,
    best_metric_value: float | None = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "condition_key": condition_key,
        "num_level_heads": len(LUMBAR_LEVELS),
        "severity_classes": SEVERITY_CLASSES,
    }
    if config is not None:
        state["config"] = config
    if val_acc is not None:
        state["val_acc"] = float(val_acc)
    if best_metric is not None:
        state["best_metric"] = best_metric
    if best_metric_value is not None:
        state["best_metric_value"] = float(best_metric_value)
    torch.save(state, output_dir / filename)


def log_dataset_split(
    split_summary: Dict[str, Any],
    output_path: Path | None = None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
) -> None:
    output_dir = output_path.parent if output_path is not None else None
    log_lumbar_dataset_split(
        split_summary,
        output_dir=output_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument(
        "--condition",
        type=str,
        default="spinal_canal_stenosis",
        choices=list(CONDITION_SPECS.keys()),
        help="Which degenerative condition to train (one model per condition).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Legacy square ViT input override (sets width and height).",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="ViT input width in pixels (default: condition-specific).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="ViT input height in pixels (default: condition-specific).",
    )
    parser.add_argument(
        "--crop-policy",
        type=str,
        default=CROP_POLICY_CENTERED,
        choices=CROP_POLICIES,
        help=(
            "Named crop policy (pairs crop geometry with ViT size): "
            "'centered' = 12.5%%×12.5%% → ViT 64×64; "
            "'extend50' = +50%% width → ViT 96×64."
        ),
    )
    parser.add_argument(
        "--image-source",
        type=str,
        default=IMAGE_SOURCE_DICOM,
        choices=IMAGE_SOURCES,
        help=(
            "Where to load training images from: 'dicom' = crop from train_images/ "
            "at runtime; 'png' = load pre-exported crops (see --crops-root)."
        ),
    )
    parser.add_argument(
        "--crops-root",
        type=str,
        default=None,
        help=(
            "Root of exported crop PNGs when --image-source png. "
            "Default: <data-root>/training_crops_png or training_crops_png_extend50 "
            "matching --crop-policy."
        ),
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Fixed square DICOM crop side in pixels (overrides condition default).",
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=None,
        help="Square crop side as fraction of min(slice width, height).",
    )
    parser.add_argument(
        "--crop-width-fraction",
        type=float,
        default=None,
        help="Rectangular crop width as fraction of slice width (use with --crop-height-fraction).",
    )
    parser.add_argument(
        "--crop-height-fraction",
        type=float,
        default=None,
        help="Rectangular crop height as fraction of slice height (use with --crop-width-fraction).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--model-type",
        type=str,
        default=MODEL_TYPE_VIT,
        choices=LUMBAR_MODEL_TYPES,
        help=(
            "Classifier family: vit | convnext | efficientnet | avengers | clam | maxvit | medsam (2D) "
            "or vit3d | convnext3d | efficientnet3d | avengers3d | clam3d | maxvit3d (3D volume)."
        ),
    )
    parser.add_argument(
        "--input-layout",
        type=str,
        default=None,
        choices=INPUT_LAYOUTS,
        help=(
            "Dataset tensor layout. Default: '2d' for vit/convnext/efficientnet; "
            "'3d_level_stack' for 3D models (five ROI crops as D=5 volume) or "
            "'3d_series' (full padded series Z=32)."
        ),
    )
    parser.add_argument(
        "--volume-depth",
        type=int,
        default=None,
        help="Fixed depth Z for 3D inputs (default: 5 level_stack, 32 series).",
    )
    parser.add_argument(
        "--backbone-variant",
        type=str,
        default=None,
        help=(
            "Backbone id. Defaults: vit→vit_b_16, convnext→convnext_small, "
            "efficientnet→efficientnet_v2_s, *3d→cnn3d_s or vit3d_s."
        ),
    )
    parser.add_argument(
        "--vit-variant",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--head-depth", type=int, default=1)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument(
        "--head-activation",
        type=str,
        default="gelu",
        choices=["relu", "gelu", "tanh"],
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="MIL bag size per level (avengers/clam). Requires stacked channels when >1.",
    )
    parser.add_argument(
        "--mil-hidden-dim",
        type=int,
        default=128,
        help="Hidden dim for gated MIL attention (avengers/clam).",
    )
    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=256,
        help="Bi-LSTM hidden dim for avengers / avengers3d.",
    )
    parser.add_argument(
        "--no-mil-attention",
        action="store_true",
        help="Disable gated MIL pooling in avengers (slice mean when num_instances>1).",
    )
    parser.add_argument(
        "--medsam-checkpoint",
        type=str,
        default=None,
        help="Path to MedSAM ViT-B checkpoint (required for --model-type medsam).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional override. Default: runs/lumbar_<model_type>/<condition>/<config_slug>/ "
            "(derived from model type, backbone, LR, WD, etc.)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after N epochs without improvement on --best-metric. "
            "Omit to train for all --epochs."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Train fraction; train + val + test must equal 1.0 (default 0.70).",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Shared train/val/test JSON (same patients for all conditions).",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc", "val_f1_macro_levels"],
        help="Metric for early stopping / best_model.pt (val_acc = max validation accuracy).",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_model.pt and last_model.pt. Default: metrics only.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.early_stop_patience is not None:
        if args.early_stop_patience < 1:
            raise SystemExit("--early-stop-patience must be >= 1.")
        args.patience = args.early_stop_patience
    elif args.patience is not None:
        args.early_stop_patience = args.patience
    if args.vit_variant is not None and args.backbone_variant is None:
        args.backbone_variant = args.vit_variant
    if args.backbone_variant is None:
        args.backbone_variant = resolve_default_backbone_variant(args.model_type)
    try:
        pipeline = resolve_pipeline_from_training_args(
            condition=args.condition,
            model_type=args.model_type,
            backbone_variant=args.backbone_variant,
            input_layout=args.input_layout,
            crop_policy=args.crop_policy,
            image_source=args.image_source,
            image_width=args.image_width,
            image_height=args.image_height,
            volume_depth=args.volume_depth,
            crop_size=args.crop_size,
            crop_fraction=args.crop_fraction,
            crop_width_fraction=args.crop_width_fraction,
            crop_height_fraction=args.crop_height_fraction,
        )
    except (ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    args.model_type = pipeline.model_type
    args.backbone_variant = pipeline.backbone_variant
    args.input_layout = pipeline.input_layout
    args.image_size = pipeline.image_size
    args.crop_settings = pipeline.crop_settings
    args.volume_depth = pipeline.volume_depth
    try:
        validate_backbone_variant(args.model_type, args.backbone_variant)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.model_type == MODEL_TYPE_MEDSAM and not args.medsam_checkpoint:
        raise SystemExit(
            "--model-type medsam requires --medsam-checkpoint pointing to MedSAM ViT-B weights."
        )
    args.resolved_crops_root = None
    if args.image_source == "png":
        args.resolved_crops_root = resolve_training_crops_root(
            args.data_root,
            crop_policy=args.crop_policy,
            crops_root=args.crops_root,
        )
    output_dir = resolve_training_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.pop("crop_settings", None)
    run_config.pop("image_size", None)
    run_config.pop("resolved_crops_root", None)
    run_config["crop_policy"] = args.crop_policy
    run_config["image_source"] = args.image_source
    run_config["crops_root"] = (
        str(args.resolved_crops_root) if args.resolved_crops_root is not None else None
    )
    run_config["crop_settings"] = {
        "crop_size": args.crop_settings.crop_size,
        "crop_fraction": args.crop_settings.crop_fraction,
        "crop_width_fraction": args.crop_settings.crop_width_fraction,
        "crop_height_fraction": args.crop_settings.crop_height_fraction,
        "crop_base_width_fraction": args.crop_settings.crop_base_width_fraction,
        "crop_width_extra_right_fraction": args.crop_settings.crop_width_extra_right_fraction,
        "crop_width_extra_left_fraction": args.crop_settings.crop_width_extra_left_fraction,
    }
    run_config["image_size"] = args.image_size.to_dict()
    run_config["model_type"] = args.model_type
    run_config["backbone_variant"] = args.backbone_variant
    run_config["variant"] = args.backbone_variant
    run_config["input_layout"] = args.input_layout
    run_config["volume_depth"] = args.volume_depth
    run_config["early_stop_patience"] = args.early_stop_patience
    run_config["patience"] = args.patience
    run_config["best_metric"] = args.best_metric
    run_config["output_dir_resolved"] = str(output_dir)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    try:
        validate_split_fractions(
            args.train_fraction, args.val_fraction, args.test_fraction
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Saving results to: {output_dir}")
    print(f"Condition: {args.condition} ({CONDITION_SPECS[args.condition]['coord_name']})")
    print(
        f"Crop: {describe_crop_policy(args.crop_settings)} "
        f"(ViT input resize: {describe_image_size(args.image_size)})"
    )
    print(f"Mode: {describe_training_mode(args.crop_policy)}")
    print(f"Image source: {args.image_source}")
    if args.resolved_crops_root is not None:
        print(f"Crops root: {args.resolved_crops_root}")
    print(f"Architecture: model_type={args.model_type} backbone={args.backbone_variant}")
    if args.model_type == MODEL_TYPE_VIT:
        print(
            f"  shared ViT encoder + {len(LUMBAR_LEVELS)} level heads x "
            f"{len(SEVERITY_CLASSES)} classes"
        )
    elif args.model_type in (MODEL_TYPE_CONVNEXT, MODEL_TYPE_EFFICIENTNET):
        print(
            f"  shared 2D CNN ({args.backbone_variant}) + {len(LUMBAR_LEVELS)} level heads x "
            f"{len(SEVERITY_CLASSES)} classes (ViT-style, no LSTM)"
        )
    elif args.model_type == MODEL_TYPE_AVENGERS:
        print(
            f"  Avengers MIL: CNN ({args.backbone_variant}) + gated MIL + bi-LSTM + "
            f"{len(LUMBAR_LEVELS)} level heads (num_instances={args.num_instances})"
        )
    elif args.model_type == MODEL_TYPE_CLAM:
        print(
            f"  CLAM-style MIL: CNN ({args.backbone_variant}) + gated attention + "
            f"{len(LUMBAR_LEVELS)} level heads (num_instances={args.num_instances})"
        )
    elif args.model_type == MODEL_TYPE_MAXVIT:
        print(
            f"  MaxViT ({args.backbone_variant}) + {len(LUMBAR_LEVELS)} level heads x "
            f"{len(SEVERITY_CLASSES)} classes"
        )
    elif args.model_type == MODEL_TYPE_MEDSAM:
        print(
            f"  MedSAM encoder (checkpoint={args.medsam_checkpoint}) + "
            f"{len(LUMBAR_LEVELS)} level heads (segmentation encoder, classification heads)"
        )
    elif args.model_type in (
        MODEL_TYPE_VIT3D,
        MODEL_TYPE_CONVNEXT3D,
        MODEL_TYPE_EFFICIENTNET3D,
        MODEL_TYPE_MAXVIT3D,
    ):
        depth_note = args.volume_depth or (
            5 if args.input_layout == INPUT_LAYOUT_3D_LEVEL_STACK else DEFAULT_VOLUME_DEPTH_SERIES
        )
        print(
            f"  single 3D encoder ({args.model_type}) over input_layout={args.input_layout} "
            f"(depth Z={depth_note}) + {len(LUMBAR_LEVELS)} level heads"
        )
    elif args.model_type == MODEL_TYPE_AVENGERS3D:
        depth_note = args.volume_depth or (
            5 if args.input_layout == INPUT_LAYOUT_3D_LEVEL_STACK else DEFAULT_VOLUME_DEPTH_SERIES
        )
        print(
            f"  Avengers3D: 2D MIL path for level_stack or volume+LSTM for series "
            f"(layout={args.input_layout}, Z={depth_note}, backbone={args.backbone_variant})"
        )
    elif args.model_type == MODEL_TYPE_CLAM3D:
        depth_note = args.volume_depth or (
            5 if args.input_layout == INPUT_LAYOUT_3D_LEVEL_STACK else DEFAULT_VOLUME_DEPTH_SERIES
        )
        print(
            f"  CLAM3D: cross-level attention MIL (layout={args.input_layout}, Z={depth_note}, "
            f"backbone={args.backbone_variant})"
        )
    print(f"Pipeline: {describe_pipeline(pipeline)}")
    if args.save_checkpoints:
        print("Checkpoint saving: ON")
    else:
        print("Checkpoint saving: OFF (metrics only). Pass --save-checkpoints to save .pt files.")
    if args.early_stop_patience is not None:
        print(
            f"Early stopping: patience={args.early_stop_patience} on {args.best_metric} "
            f"(max epochs={args.epochs})."
        )

    weight_enum = pretrained_weights_for_model(
        args.model_type,
        args.backbone_variant,
        use_pretrained=not args.no_pretrained,
    )
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    random.seed(split_seed)

    fixed_split = None
    split_manifest_path = None
    if args.split_manifest:
        from lumbar_shared_split import load_shared_split

        shared = load_shared_split(Path(args.split_manifest))
        fixed_split = shared.as_fixed_tuple()
        split_manifest_path = str(Path(args.split_manifest).resolve())
        print(f"Using shared split manifest: {split_manifest_path}")
        print(
            f"  train={shared.n_train} val={shared.n_val} test={shared.n_test} "
            f"(seed={shared.split_seed})"
        )

    train_loader, val_loader, test_loader, split_summary, train_dataset, val_dataset, test_dataset = (
        build_lumbar_dataloaders(
            data_root=Path(args.data_root),
            condition_key=args.condition,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weights=weight_enum,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            crop_settings=args.crop_settings,
            max_studies=args.max_studies,
            image_source=args.image_source,
            crops_root=args.resolved_crops_root,
            fixed_split=fixed_split,
            split_manifest_path=split_manifest_path,
            input_layout=args.input_layout,
            volume_depth=args.volume_depth,
            pipeline=pipeline,
        )
    )
    split_summary["split_seed_used"] = split_seed
    split_summary["training_seed_after_split"] = args.seed
    set_seed(args.seed)
    log_dataset_split(
        split_summary,
        output_dir / "dataset_split.json",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    model, _weights = build_lumbar_model(
        args.model_type,
        args.backbone_variant,
        image_size=args.image_size,
        input_layout=args.input_layout,
        freeze_backbone=args.freeze_backbone,
        use_pretrained=not args.no_pretrained,
        head_depth=args.head_depth,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_activation=args.head_activation,
        num_instances=args.num_instances,
        mil_hidden_dim=args.mil_hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        use_mil_attention=not args.no_mil_attention,
        medsam_checkpoint=args.medsam_checkpoint,
    )
    model.to(device)

    checkpoint_config = {
        **pipeline.to_checkpoint_dict(),
        "head_depth": args.head_depth,
        "head_hidden_dim": args.head_hidden_dim,
        "head_dropout": args.head_dropout,
        "head_activation": args.head_activation,
        "num_instances": args.num_instances,
        "mil_hidden_dim": args.mil_hidden_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "use_mil_attention": not args.no_mil_attention,
        "medsam_checkpoint": args.medsam_checkpoint,
    }

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = multi_head_cross_entropy
    scaler = create_grad_scaler(args.amp, device)

    best_val_loss = math.inf
    best_metric_value = math.inf if args.best_metric == "val_loss" else -math.inf
    best_epoch = -1
    best_val_acc_for_cm = -1.0
    history: List[Dict[str, Any]] = []
    epochs_no_improve = 0
    csv_path = output_dir / "training_metrics.csv"
    csv_initialized = False

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            show_batch_progress=args.progress,
            batch_progress_desc=f"Epoch {epoch}/{args.epochs} train",
            log_interval=args.log_interval,
        )
        val_loss, val_acc, val_logits, val_targets = evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_batch_progress=args.progress,
            batch_progress_desc=f"Epoch {epoch}/{args.epochs} val",
            log_interval=args.log_interval,
        )
        test_loss, test_acc, test_logits, test_targets = evaluate(
            model,
            test_loader,
            criterion,
            device,
            show_batch_progress=args.progress,
            batch_progress_desc=f"Epoch {epoch}/{args.epochs} test",
            log_interval=args.log_interval,
        )

        model.eval()
        train_logits_list = []
        train_targets_list = []
        with torch.no_grad():
            for images, targets, _ in train_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                train_logits_list.append(outputs.cpu().numpy())
                train_targets_list.append(targets.numpy())
        train_logits = np.concatenate(train_logits_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)
        train_metrics = compute_multi_head_metrics(train_logits, train_targets)
        val_metrics = compute_multi_head_metrics(val_logits, val_targets)
        test_metrics = compute_multi_head_metrics(test_logits, test_targets)
        model.train()

        scheduler.step()
        epoch_time_sec = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} | "
            f"test_acc={test_acc:.4f} test_loss={test_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | epoch_time_sec={epoch_time_sec:.1f}",
            flush=True,
        )

        epoch_history: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_time_sec,
        }
        for prefix, metric_dict in (
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ):
            merge_multi_head_metrics(epoch_history, prefix, metric_dict)
        history.append(epoch_history)

        write_mode = "a" if csv_initialized else "w"
        with csv_path.open(write_mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(epoch_history.keys()))
            if not csv_initialized:
                writer.writeheader()
                csv_initialized = True
            writer.writerow(epoch_history)

        best_val_loss = min(best_val_loss, val_loss)
        selection_value = (
            val_loss
            if args.best_metric == "val_loss"
            else float(val_acc)
            if args.best_metric == "val_acc"
            else float(val_metrics["f1_macro_levels"])
        )
        is_better = (
            selection_value < best_metric_value
            if args.best_metric == "val_loss"
            else selection_value > best_metric_value
        )
        if is_better:
            best_metric_value = selection_value
            best_epoch = epoch
            epochs_no_improve = 0
            if args.save_checkpoints:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    args.condition,
                    output_dir,
                    "best_model.pt",
                    config=checkpoint_config,
                    val_acc=float(val_acc),
                    best_metric=args.best_metric,
                    best_metric_value=float(best_metric_value),
                )
        if val_acc > best_val_acc_for_cm:
            best_val_acc_for_cm = val_acc
            save_per_level_confusion_matrices(
                val_logits,
                val_targets,
                output_dir,
                split="val",
                epoch=epoch,
                val_acc=val_acc,
                condition=args.condition,
            )
            print(
                f"  -> Updated val confusion matrices (best val_acc={val_acc:.4f}, epoch {epoch})",
                flush=True,
            )
        if not is_better:
            epochs_no_improve += 1
            if args.patience is not None and epochs_no_improve >= args.patience:
                print(
                    f"Early stopping after {args.patience} epoch(s) without "
                    f"{args.best_metric} improvement (stopped at epoch {epoch})."
                )
                break

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if args.save_checkpoints:
        save_checkpoint(
            model,
            optimizer,
            history[-1]["epoch"] if history else 0,
            best_val_loss,
            args.condition,
            output_dir,
            "last_model.pt",
            config=checkpoint_config,
        )
    print(f"Best epoch: {best_epoch}")
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
