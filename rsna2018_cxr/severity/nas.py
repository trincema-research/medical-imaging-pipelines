#!/usr/bin/env python
"""
Automated Vision Transformer NAS for chest X-ray folders.

Default ``--data-root .`` targets the RSNA ``severity_split_jpg_dataset`` layout (four classes:
normal + three pneumonia severities). Any torchvision ImageFolder layout (e.g. NORMAL/PNEUMONIA)
still works when severity folders are absent.

Edit the `DEFAULT_SEARCH_SPACE` object below to change which architectures and hyperparameters
are explored. After that, you can run the script with no extra CLI arguments:

    python vit_nas.py

All overrides are optional; you can still tweak behaviour via flags such as `--mode`,
`--num-trials`, or by pointing to a JSON config file with `--config-file`.
By default the script now runs the full cartesian product defined in `DEFAULT_SEARCH_SPACE`
and reports a tabular summary plus a CSV file with every trial.

PATCH NOTE (custom high resolutions like 1024):
- Torchvision VisionTransformer asserts input H/W == model.image_size (often 224).
- Pretrained ViT also has learned positional embeddings tied to the patch grid.
This script adapts torchvision ViT models to any square resolution divisible by patch size
by updating model.image_size and bicubically interpolating positional embeddings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from rsna2018_cxr.severity.train_vit import (
    build_dataloaders,
    build_model,
    build_transforms,
    compute_classification_metrics,
    count_files_per_class,
    create_grad_scaler,
    evaluate,
    iter_with_progress,
    log_dataset_split,
    save_confusion_matrix_artifacts,
    set_seed,
    train_one_epoch,
)


# =========================
# ViT 1024-resolution patch
# =========================

def _resize_pos_embed_vit(
    pos_embed: torch.Tensor,  # (1, old_tokens, dim)
    old_grid: int,
    new_grid: int,
) -> torch.Tensor:
    """
    Interpolate ViT positional embeddings from old_grid x old_grid to new_grid x new_grid.
    Assumes first token is CLS.
    """
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Unexpected pos_embed shape: {tuple(pos_embed.shape)}")

    cls_tok = pos_embed[:, :1, :]          # (1, 1, dim)
    grid_tok = pos_embed[:, 1:, :]         # (1, old_grid*old_grid, dim)

    dim = grid_tok.shape[-1]
    grid_tok = grid_tok.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)  # (1, dim, H, W)

    # Bicubic interpolation in embedding space
    grid_tok = F.interpolate(
        grid_tok,
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    )

    grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)  # (1, new_tokens, dim)
    return torch.cat([cls_tok, grid_tok], dim=1)  # (1, 1+new_tokens, dim)


def adapt_torchvision_vit_to_image_size(model: nn.Module, image_size: int) -> nn.Module:
    """
    Makes torchvision VisionTransformer accept a new square image_size by:
      - updating model.image_size (fixes torchvision forward assertion)
      - resizing encoder.pos_embedding to match the new token grid (pretrained models)

    If model isn't a torchvision ViT (no expected attrs), this is a no-op.
    """
    # Heuristic: torchvision ViT has: image_size, conv_proj, encoder.pos_embedding
    if not (hasattr(model, "image_size") and hasattr(model, "conv_proj") and hasattr(model, "encoder")):
        return model
    if not hasattr(model.encoder, "pos_embedding"):
        return model

    conv = model.conv_proj
    if not hasattr(conv, "kernel_size") or not hasattr(conv, "stride"):
        return model

    patch = conv.kernel_size[0]
    if image_size % patch != 0:
        raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch}")

    old_img = int(getattr(model, "image_size"))
    old_grid = old_img // patch
    new_grid = image_size // patch

    # Update image_size so torchvision forward() assertion passes
    model.image_size = image_size

    # Resize positional embedding if token counts differ
    with torch.no_grad():
        pos = model.encoder.pos_embedding  # nn.Parameter
        old_tokens = pos.shape[1]          # 1 + old_grid*old_grid
        new_tokens = 1 + new_grid * new_grid
        if old_tokens != new_tokens:
            new_pos = _resize_pos_embed_vit(pos.data, old_grid=old_grid, new_grid=new_grid)
            model.encoder.pos_embedding = nn.Parameter(new_pos)

    return model


# =========================
# Search space definitions
# =========================

@dataclass
class SearchSpace:
    image_sizes: List[int] = field(default_factory=lambda: [512])
    batch_sizes: List[int] = field(default_factory=lambda: [8])
    learning_rates: List[float] = field(default_factory=lambda: [3e-4, 1e-4, 5e-5])
    weight_decays: List[float] = field(default_factory=lambda: [5e-2, 1e-2])
    vit_variants: List[str] = field(
        default_factory=lambda: ["vit_b_16", "vit_l_16", "vit_b_32", "vit_l_32"]
    )
    freeze_options: List[bool] = field(default_factory=lambda: [False])
    use_pretrained: List[bool] = field(default_factory=lambda: [True])
    amp_options: List[bool] = field(default_factory=lambda: [True])
    # Classification head architecture
    head_depths: List[int] = field(default_factory=lambda: [1])
    head_hidden_dims: List[int | None] = field(default_factory=lambda: [None])
    head_dropout_rates: List[float] = field(default_factory=lambda: [0.0])
    head_activations: List[str] = field(default_factory=lambda: ["gelu"])
    # Optimizer and scheduler
    optimizer_types: List[str] = field(default_factory=lambda: ["AdamW"])
    scheduler_types: List[str] = field(default_factory=lambda: ["CosineAnnealingLR"])
    # Training parameters
    epochs: int = 250
    patience: int | None = None
    num_workers: int = 4
    default_num_trials: int | None = None  # None => run all combinations
    search_mode: str = "grid"  # "random" or "grid"


DEFAULT_SEARCH_SPACE = SearchSpace()


@dataclass
class TrialConfig:
    variant: str
    image_size: int
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
    epochs: int
    patience: int
    num_workers: int
    seed: int


@dataclass
class TrialResult:
    trial_id: int
    variant: str
    image_size: int
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
    final_train_acc: float
    final_val_acc: float
    final_test_acc: float
    max_train_acc: float
    max_val_acc: float
    max_test_acc: float
    best_epoch: int
    duration_sec: float
    output_dir: str
    trainable_params: int
    total_params: int
    epochs_ran: int
    # Classification metrics (from best epoch)
    train_precision_macro: float
    train_recall_macro: float
    train_f1_macro: float
    val_precision_macro: float
    val_recall_macro: float
    val_f1_macro: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAS over Vision Transformer configs.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Dataset root: RSNA severity_split (normal/ + pneumonia/severity_*) or ImageFolder (NORMAL/PNEUMONIA). Use '.' when running inside severity_split_jpg_dataset.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="runs/nas",
        help="Base directory for storing individual trial outputs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "grid"],
        default=None,
        help="Override search mode (defaults to the value in the params object).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials to execute. In grid mode this truncates the cartesian product.",
    )
    parser.add_argument(
        "--start-trial",
        type=int,
        default=1,
        metavar="N",
        help="1-based trial index to start from (skip trials with ID < N). Default: 1.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional JSON file to override `DEFAULT_SEARCH_SPACE` fields.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs for every NAS trial.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1234,
        help="RNG seed for every trial (split + train init/dropout). Same across trials unless --unique-seed-per-trial.",
    )
    parser.add_argument(
        "--unique-seed-per-trial",
        action="store_true",
        help="Use base_seed + trial_index per trial (different split/init each trial). Default: same seed for all trials.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials whose output directory already has result.json.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Per-class val fraction (same pool as test). Passed to train_vit.build_dataloaders.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Per-batch tqdm bars in train/val (same as train_vit --progress).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print intra-epoch batch progress every N batches (0 disables).",
    )
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse"],
        help="Class weights for CrossEntropy on train split.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for train loader.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_f1_macro", "val_acc_then_train_acc"],
        help="Metric used to select and save best checkpoint.",
    )
    return parser.parse_args(argv)


def load_search_space(args: argparse.Namespace) -> SearchSpace:
    space = copy.deepcopy(DEFAULT_SEARCH_SPACE)
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        for key, value in overrides.items():
            if not hasattr(space, key):
                raise KeyError(f"Unknown search space key in config file: {key}")
            setattr(space, key, value)
    if args.mode:
        space.search_mode = args.mode
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1.")
        space.epochs = args.epochs
    return space


def build_trial_configs(
    space: SearchSpace,
    base_seed: int,
    *,
    unique_seed_per_trial: bool = False,
) -> List[TrialConfig]:
    """
    By default every trial reuses ``base_seed`` so data split and (pre-)training RNG match
    across NAS runs for fair comparison. Set ``unique_seed_per_trial`` to use
    ``base_seed + trial_index`` (legacy behavior).
    """
    patience = space.patience if space.patience is not None else space.epochs
    configs: List[TrialConfig] = []
    for counter, (
        variant, image_size, batch_size, lr, wd, freeze, pretrained, amp,
        head_depth, head_hidden_dim, head_dropout, head_activation,
        optimizer_type, scheduler_type
    ) in enumerate(
        itertools.product(
            space.vit_variants,
            space.image_sizes,
            space.batch_sizes,
            space.learning_rates,
            space.weight_decays,
            space.freeze_options,
            space.use_pretrained,
            space.amp_options,
            space.head_depths,
            space.head_hidden_dims,
            space.head_dropout_rates,
            space.head_activations,
            space.optimizer_types,
            space.scheduler_types,
        )
    ):
        configs.append(
            TrialConfig(
                variant=variant,
                image_size=image_size,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,
                freeze_backbone=freeze,
                use_pretrained=pretrained,
                amp=amp,
                head_depth=head_depth,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
                head_activation=head_activation,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                epochs=space.epochs,
                patience=patience,
                num_workers=space.num_workers,
                seed=(base_seed + counter) if unique_seed_per_trial else base_seed,
            )
        )
    return configs


def select_trials(
    configs: List[TrialConfig],
    mode: str,
    default_num_trials: int | None,
    explicit_num_trials: int | None,
) -> List[TrialConfig]:
    if mode == "grid":
        if explicit_num_trials is None:
            return configs
        return configs[:explicit_num_trials]

    num_trials = explicit_num_trials if explicit_num_trials is not None else default_num_trials
    if num_trials is None or num_trials >= len(configs):
        return configs
    return random.sample(configs, k=num_trials)


def create_optimizer(params, optimizer_type: str, lr: float, weight_decay: float):
    """Create optimizer based on type."""
    if optimizer_type == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer, scheduler_type: str, epochs: int):
    """Create learning rate scheduler based on type."""
    if scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "StepLR":
        # Avoid step_size=0 if epochs < 3
        step = max(1, epochs // 3)
        return StepLR(optimizer, step_size=step, gamma=0.1)
    elif scheduler_type == "ExponentialLR":
        return ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def run_trial(
    trial_id: int,
    config: TrialConfig,
    data_root: Path,
    output_dir: Path,
    val_fraction: float = 0.10,
    show_batch_progress: bool = False,
    log_interval: int = 0,
    class_weight_mode: str = "none",
    balanced_sampler: bool = False,
    best_metric: str = "val_loss",
) -> TrialResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration as JSON
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    # Track the current best model file for this trial (so we can delete it)
    best_model_file: Path | None = None

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Trial {trial_id} | Device: {device}")
    print(f"Dataset: {data_root}")
    print(f"{'='*80}")

    # Load data
    train_loader, val_loader, test_loader, class_to_idx, split_summary = build_dataloaders(
        data_root=data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_fraction=val_fraction,
        balanced_sampler=balanced_sampler,
    )

    log_dataset_split(split_summary, output_dir / "dataset_split.json")

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # --- NEW: Save CSV listing val/test images and their classes ---
    val_test_csv_path = output_dir / "val_test_images.csv"
    with val_test_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "image_path", "class_name", "class_index"])
        for split_name, dataset in (("val", val_loader.dataset), ("test", test_loader.dataset)):
            for path, class_idx in dataset.samples:
                class_name = idx_to_class[class_idx]
                writer.writerow([split_name, str(path), class_name, class_idx])
    print(f"Saved validation/test image list to {val_test_csv_path}")

    # Print dataset statistics
    print("\nDataset Statistics:")
    train_counts = count_files_per_class(train_loader.dataset, idx_to_class)
    val_counts = count_files_per_class(val_loader.dataset, idx_to_class)
    test_counts = count_files_per_class(test_loader.dataset, idx_to_class)

    print(f"  Train: {len(train_loader.dataset):,} files")
    for class_name in sorted(idx_to_class.values()):
        print(f"    - {class_name}: {train_counts[class_name]:,} files")

    print(f"  Validation: {len(val_loader.dataset):,} files")
    for class_name in sorted(idx_to_class.values()):
        print(f"    - {class_name}: {val_counts[class_name]:,} files")

    print(f"  Test: {len(test_loader.dataset):,} files")
    for class_name in sorted(idx_to_class.values()):
        print(f"    - {class_name}: {test_counts[class_name]:,} files")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  ViT Variant: {config.variant}")
    print(f"  Image Size: {config.image_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Optimizer: {config.optimizer_type}")
    print(f"  Scheduler: {config.scheduler_type}")
    print(f"  Head Depth: {config.head_depth}")
    print(f"  Head Hidden Dim: {config.head_hidden_dim}")
    print(f"  Head Dropout: {config.head_dropout}")
    print(f"  Head Activation: {config.head_activation}")
    print(f"  Freeze Backbone: {config.freeze_backbone}")
    print(f"  Pretrained: {config.use_pretrained}")
    print(f"  AMP: {config.amp}")
    print(f"  Epochs: {config.epochs}")
    print(f"{'='*80}\n")

    model, weights = build_model(
        config.variant,
        num_classes=num_classes,
        freeze_backbone=config.freeze_backbone,
        use_pretrained=config.use_pretrained,
        head_depth=config.head_depth,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        head_activation=config.head_activation,
    )

    # --- Critical fix for high custom resolutions (e.g. 1024) ---
    model = adapt_torchvision_vit_to_image_size(model, config.image_size)

    train_tf, eval_tf = build_transforms(config.image_size, weights)
    train_loader.dataset.transform = train_tf
    val_loader.dataset.transform = eval_tf
    test_loader.dataset.transform = eval_tf

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(params, config.optimizer_type, config.learning_rate, config.weight_decay)
    scheduler = create_scheduler(optimizer, config.scheduler_type, config.epochs)
    class_weight_tensor = None
    if class_weight_mode != "none":
        train_counts = split_summary["files_per_class_train"]
        counts = np.array(
            [float(train_counts.get(idx_to_class[i], 0)) for i in range(num_classes)],
            dtype=np.float64,
        )
        if np.any(counts <= 0):
            raise RuntimeError(f"Invalid train class counts for class weighting: {counts.tolist()}")
        if class_weight_mode == "inverse":
            w = 1.0 / counts
        else:  # sqrt_inverse
            w = 1.0 / np.sqrt(counts)
        w = w * (len(w) / float(w.sum()))
        class_weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Using class weights ({class_weight_mode}): {w.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    scaler = create_grad_scaler(config.amp, device)

    best_val_loss = float("inf")
    if best_metric == "val_loss":
        best_metric_value = float("inf")
    elif best_metric == "val_f1_macro":
        best_metric_value = float("-inf")
    elif best_metric == "val_acc_then_train_acc":
        best_metric_value = (float("-inf"), float("-inf"))
    else:
        raise ValueError(f"Unknown best_metric: {best_metric}")
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    history_csv_path = output_dir / "training_history.csv"
    history_csv_initialized = False
    last_epoch = 0
    last_train_preds = np.array([], dtype=np.int64)
    last_train_targets = np.array([], dtype=np.int64)
    last_val_preds = np.array([], dtype=np.int64)
    last_val_targets = np.array([], dtype=np.int64)
    last_test_preds = np.array([], dtype=np.int64)
    last_test_targets = np.array([], dtype=np.int64)

    start_trial = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_runtime_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            show_batch_progress=show_batch_progress,
            batch_progress_desc=f"T{trial_id} ep{epoch}/{config.epochs} train",
            log_interval=log_interval,
        )
        val_loss, val_acc, val_preds, val_targets, val_runtime_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_batch_progress=show_batch_progress,
            batch_progress_desc=f"T{trial_id} ep{epoch}/{config.epochs} val",
            log_interval=log_interval,
        )
        test_loss, test_acc, test_preds, test_targets, test_runtime_stats = evaluate(
            model,
            test_loader,
            criterion,
            device,
            show_batch_progress=show_batch_progress,
            batch_progress_desc=f"T{trial_id} ep{epoch}/{config.epochs} test",
            log_interval=log_interval,
        )

        # Get train predictions
        model.eval()
        train_preds = []
        train_targets = []
        with torch.no_grad():
            _tr_it = iter_with_progress(
                train_loader,
                enabled=show_batch_progress,
                desc=f"T{trial_id} ep{epoch}/{config.epochs} train metrics",
            )
            for images, targets in _tr_it:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(targets.cpu().numpy())
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        last_epoch = epoch
        last_train_preds = train_preds
        last_train_targets = train_targets
        last_val_preds = val_preds
        last_val_targets = val_targets
        last_test_preds = test_preds
        last_test_targets = test_targets
        model.train()

        # Compute classification metrics
        train_metrics = compute_classification_metrics(train_targets, train_preds, num_classes)
        val_metrics = compute_classification_metrics(val_targets, val_preds, num_classes)
        test_metrics = compute_classification_metrics(test_targets, test_preds, num_classes)

        # Step scheduler (handle ReduceLROnPlateau separately)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = (
            scheduler.get_last_lr()[0]
            if not isinstance(scheduler, ReduceLROnPlateau)
            else optimizer.param_groups[0]["lr"]
        )

        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }

        # Add all classification metrics
        for key, value in train_metrics.items():
            epoch_history[f"train_{key}"] = value
        for key, value in val_metrics.items():
            epoch_history[f"val_{key}"] = value
        for key, value in test_metrics.items():
            epoch_history[f"test_{key}"] = value
        for key, value in train_runtime_stats.items():
            epoch_history[f"train_{key}"] = value
        for key, value in val_runtime_stats.items():
            epoch_history[f"val_{key}"] = value
        for key, value in test_runtime_stats.items():
            epoch_history[f"test_{key}"] = value

        history.append(epoch_history)

        # Persist per-epoch metrics immediately so partial progress survives interruptions.
        write_mode = "a" if history_csv_initialized else "w"
        with history_csv_path.open(write_mode, newline="", encoding="utf-8") as fh:
            fieldnames = list(epoch_history.keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not history_csv_initialized:
                writer.writeheader()
                history_csv_initialized = True
            writer.writerow(epoch_history)

        train_class_acc_str = "  ".join(
            f"{idx_to_class[i]}={train_metrics.get(f'accuracy_class_{i}', 0.0):.4f}"
            for i in range(num_classes)
        )
        val_class_acc_str = "  ".join(
            f"{idx_to_class[i]}={val_metrics.get(f'accuracy_class_{i}', 0.0):.4f}"
            for i in range(num_classes)
        )
        print(
            f"Trial {trial_id} | Epoch {epoch:03d}/{config.epochs} | "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  | "
            f"lr={current_lr:.2e}  | {epoch_time:.1f}s",
            flush=True,
        )
        print(f"  train_class_acc: {train_class_acc_str}", flush=True)
        print(f"  val_class_acc:   {val_class_acc_str}", flush=True)

        best_val_loss = min(best_val_loss, val_loss)
        if best_metric == "val_loss":
            selection_value = val_loss
            is_better = selection_value < best_metric_value
        elif best_metric == "val_f1_macro":
            selection_value = float(val_metrics["f1_macro"])
            is_better = selection_value > best_metric_value
        elif best_metric == "val_acc_then_train_acc":
            selection_value = (float(val_acc), float(train_acc))
            is_better = selection_value > best_metric_value
        else:
            raise ValueError(f"Unknown best_metric: {best_metric}")

        if is_better:
            best_metric_value = selection_value
            best_epoch = epoch
            epochs_no_improve = 0

            # Prepare checkpoint state
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": asdict(config),
                "class_to_idx": class_to_idx,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }

            # Delete previous best checkpoint (if any) so we keep only one file
            if best_model_file is not None and best_model_file.exists():
                best_model_file.unlink()

            # Filename with metrics
            acc_tag = f"val{val_acc:.4f}_train{train_acc:.4f}"
            best_model_file = output_dir / f"best_model_{acc_tag}.pt"
            torch.save(state, best_model_file)

            print(f"  -> Saved new best model to {best_model_file}")
        else:
            epochs_no_improve += 1
            if config.patience is not None and epochs_no_improve >= config.patience:
                print(f"Early stopping triggered after {config.patience} epochs without improvement.")
                break

    if not history:
        raise RuntimeError("No training epochs completed; check configuration.")
    if best_epoch == -1:
        best_epoch = len(history)

    if history:
        print(f"Saved detailed training history to {history_csv_path}")

    # Save explicit last-epoch checkpoint.
    last_state = {
        "epoch": last_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "class_to_idx": class_to_idx,
        "train_acc": float(history[-1]["train_acc"]),
        "val_acc": float(history[-1]["val_acc"]),
    }
    last_model_file = output_dir / "last_model.pt"
    torch.save(last_state, last_model_file)
    print(f"Saved last-epoch checkpoint to {last_model_file}")

    class_names = [idx_to_class[i] for i in range(num_classes)]
    save_confusion_matrix_artifacts(
        last_train_targets,
        last_train_preds,
        class_names,
        output_dir / "confusion_matrix_last_train.csv",
        output_dir / "confusion_matrix_last_train.png",
        f"Train Confusion Matrix (Last Epoch {last_epoch})",
    )
    save_confusion_matrix_artifacts(
        last_val_targets,
        last_val_preds,
        class_names,
        output_dir / "confusion_matrix_last_val.csv",
        output_dir / "confusion_matrix_last_val.png",
        f"Validation Confusion Matrix (Last Epoch {last_epoch})",
    )
    save_confusion_matrix_artifacts(
        last_test_targets,
        last_test_preds,
        class_names,
        output_dir / "confusion_matrix_last_test.csv",
        output_dir / "confusion_matrix_last_test.png",
        f"Test Confusion Matrix (Last Epoch {last_epoch})",
    )
    print("Saved last-epoch confusion matrices (CSV + PNG) for train/val/test.")

    # Get metrics from best epoch
    best_history = next((h for h in history if h["epoch"] == best_epoch), history[-1])

    final_train_acc = history[-1]["train_acc"]
    final_val_acc = history[-1]["val_acc"]
    final_test_acc = history[-1]["test_acc"]
    max_train_acc = max(entry["train_acc"] for entry in history)
    max_val_acc = max(entry["val_acc"] for entry in history)
    max_test_acc = max(entry["test_acc"] for entry in history)

    duration = time.time() - start_trial
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    epochs_ran = len(history)

    result = TrialResult(
        trial_id=trial_id,
        variant=config.variant,
        image_size=config.image_size,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_backbone=config.freeze_backbone,
        use_pretrained=config.use_pretrained,
        amp=config.amp,
        head_depth=config.head_depth,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        head_activation=config.head_activation,
        optimizer_type=config.optimizer_type,
        scheduler_type=config.scheduler_type,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        final_test_acc=final_test_acc,
        max_train_acc=max_train_acc,
        max_val_acc=max_val_acc,
        max_test_acc=max_test_acc,
        best_epoch=best_epoch,
        duration_sec=duration,
        output_dir=str(output_dir),
        trainable_params=trainable_params,
        total_params=total_params,
        epochs_ran=epochs_ran,
        train_precision_macro=best_history.get("train_precision_macro", 0.0),
        train_recall_macro=best_history.get("train_recall_macro", 0.0),
        train_f1_macro=best_history.get("train_f1_macro", 0.0),
        val_precision_macro=best_history.get("val_precision_macro", 0.0),
        val_recall_macro=best_history.get("val_recall_macro", 0.0),
        val_f1_macro=best_history.get("val_f1_macro", 0.0),
        test_precision_macro=best_history.get("test_precision_macro", 0.0),
        test_recall_macro=best_history.get("test_recall_macro", 0.0),
        test_f1_macro=best_history.get("test_f1_macro", 0.0),
    )

    return result


def format_trial_dir(trial_id: int, config: TrialConfig) -> str:
    head_dim_str = f"hdim{config.head_hidden_dim}" if config.head_hidden_dim else "hdimNone"
    return (
        f"trial_{trial_id:03d}_"
        f"{config.variant}_img{config.image_size}_bs{config.batch_size}_"
        f"lr{config.learning_rate:g}_wd{config.weight_decay:g}_"
        f"head{config.head_depth}_{head_dim_str}_do{config.head_dropout}_{config.head_activation}_"
        f"{config.optimizer_type}_{config.scheduler_type}_"
        f"{'freeze' if config.freeze_backbone else 'finetune'}_"
        f"{'pre' if config.use_pretrained else 'scratch'}_"
        f"{'amp' if config.amp else 'fp32'}"
    )


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in [0, 1).")
    if args.start_trial < 1:
        raise SystemExit("--start-trial must be >= 1.")
    random.seed(args.base_seed)

    data_root = Path(args.data_root)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    space = load_search_space(args)
    configs = build_trial_configs(
        space,
        args.base_seed,
        unique_seed_per_trial=args.unique_seed_per_trial,
    )

    if args.unique_seed_per_trial:
        print(
            "NAS mode: each trial uses seed = base_seed + trial_index "
            "(different data split / init per trial).\n"
        )
    else:
        print(
            f"NAS mode: all trials use the same seed ({args.base_seed}) "
            "for identical stratified split and reproducible training RNG — recommended for comparisons.\n"
        )

    if not configs:
        print("Search space is empty; adjust DEFAULT_SEARCH_SPACE.")
        return

    selected = select_trials(
        configs=configs,
        mode=space.search_mode,
        default_num_trials=space.default_num_trials,
        explicit_num_trials=args.num_trials,
    )

    results: List[TrialResult] = []
    for idx, config in enumerate(selected, start=1):
        if idx < args.start_trial:
            print(f"Skipping trial {idx} (--start-trial={args.start_trial})")
            continue
        
        trial_dir = output_base / format_trial_dir(idx, config)
        print(
            f"\n=== Trial {idx}/{len(selected)} ===\n"
            f"Variant: {config.variant}, image={config.image_size}, batch={config.batch_size}, "
            f"lr={config.learning_rate}, wd={config.weight_decay}, "
            f"head(depth={config.head_depth}, hidden={config.head_hidden_dim}, dropout={config.head_dropout}, act={config.head_activation}), "
            f"optimizer={config.optimizer_type}, scheduler={config.scheduler_type}, "
            f"freeze={config.freeze_backbone}, pretrained={config.use_pretrained}, amp={config.amp}"
        )

        try:
            result = run_trial(
                idx,
                config,
                data_root,
                trial_dir,
                val_fraction=args.val_fraction,
                show_batch_progress=args.progress,
                log_interval=args.log_interval,
                class_weight_mode=args.class_weight_mode,
                balanced_sampler=args.balanced_sampler,
                best_metric=args.best_metric,
            )
            results.append(result)
            print(
                f"Trial {idx} finished: "
                f"final_val_acc={result.final_val_acc:.4f}, max_val_acc={result.max_val_acc:.4f}, "
                f"final_test_acc={result.final_test_acc:.4f}, max_test_acc={result.max_test_acc:.4f}, "
                f"epochs_ran={result.epochs_ran}, duration={result.duration_sec/60:.1f} min"
            )
        except RuntimeError as exc:
            print(f"Trial {idx} failed with runtime error: {exc}")
            with (trial_dir / "error.txt").open("w", encoding="utf-8") as fh:
                fh.write(str(exc))
        except Exception as exc:
            # Catch-all to also log assertion errors or unexpected model/dataloader issues
            print(f"Trial {idx} failed with error: {exc}")
            with (trial_dir / "error.txt").open("w", encoding="utf-8") as fh:
                fh.write(repr(exc))

    if not results:
        print("No trials completed.")
        return

    summary_path = output_base / "summary.csv"
    file_exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        if not file_exists:
            writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))
    print(f"\nAppended summary results to {summary_path}")

    print("\nTrial Summary:")
    header = (
        f"{'ID':>3}  {'Variant':<10}  {'Img':>4}  {'Batch':>5}  {'Head':>6}  "
        f"{'LR':>9}  {'FinalVal':>9}  {'MaxVal':>8}  {'FinalTest':>9}  {'MaxTest':>8}  "
        f"{'Epochs':>6}  {'Time(min)':>9}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        head_str = f"{res.head_depth}L"
        print(
            f"{res.trial_id:>3}  {res.variant:<10}  {res.image_size:>4}  {res.batch_size:>5}  {head_str:>6}  "
            f"{res.learning_rate:>9.2e}  {res.final_val_acc:>9.4f}  {res.max_val_acc:>8.4f}  "
            f"{res.final_test_acc:>9.4f}  {res.max_test_acc:>8.4f}  "
            f"{res.epochs_ran:>6}  {res.duration_sec / 60:>9.1f}"
        )

    best_val = max(results, key=lambda r: r.max_val_acc)
    best_test = max(results, key=lambda r: r.max_test_acc)
    print(
        f"\nTop trial by max_val_acc: {best_val.variant} | img={best_val.image_size} | "
        f"head={best_val.head_depth}L | max_val_acc={best_val.max_val_acc:.4f} | max_test_acc={best_val.max_test_acc:.4f}"
    )
    print(
        f"Top trial by max_test_acc: {best_test.variant} | img={best_test.image_size} | "
        f"head={best_test.head_depth}L | max_val_acc={best_test.max_val_acc:.4f} | max_test_acc={best_test.max_test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
