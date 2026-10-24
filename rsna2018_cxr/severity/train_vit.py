#!/usr/bin/env python
"""
Train a Vision Transformer on chest X-ray JPEG folders with PyTorch.

Supported layouts:

1) RSNA severity split (four-class), same as ``severity_split_jpg_dataset`` export::

    data_root/
        normal/
        pneumonia/
            severity_1_low_mild/
            severity_2_moderate/
            severity_3_severe/

2) Binary Mooney-style::

    data_root/
        NORMAL/
        PNEUMONIA/

Stratified split **per class** is controlled by ``--val-fraction`` (default ``0.10``):
each class contributes about that fraction to validation; **test uses the same
indices as validation**. Training gets the remainder (≈ ``1 - val_fraction``).
Example: ``--val-fraction 0.15`` → ~85% train / ~15% val+test per class.

``--image-size`` larger than the ViT defaults (e.g. 512): ``train_vit`` resizes positional
embeddings like ``vit_nas`` so torchvision ViT can run at that square resolution (must be
divisible by the model patch size).

Usage examples::

    # RSNA severity_split_jpg_dataset (run from this folder):
    python train_vit.py --data-root . --epochs 200 --image-size 224 --batch-size 16 \\
        --vit-variant vit_b_16 --output-dir runs/vit_b_16

    # Binary Mooney-style folders elsewhere:
    python train_vit.py --data-root "../[1.15 GB] Chest Xray Mooney" --epochs 200
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)
from torchvision.models.vision_transformer import VisionTransformer
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def create_grad_scaler(use_amp: bool, device: torch.device):
    if not use_amp or device.type != "cuda":
        return None
    # Prefer torch.amp (PyTorch 2.4+); string device avoids FutureWarning on torch.cuda.amp.GradScaler.
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        try:
            return torch.amp.GradScaler(device_type="cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(
    image_size: int,
    weights=None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if weights is not None:
        preprocess = weights.transforms()
        mean = preprocess.mean
        std = preprocess.std

    train_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pil_loader_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# RSNA ``severity_split_jpg_dataset`` layout (must match create_severity_split_jpg_dataset.py)
SEVERITY_SPLIT_CLASS_DIRS: List[Tuple[str, Path]] = [
    ("normal", Path("normal")),
    ("severity_1_low_mild", Path("pneumonia") / "severity_1_low_mild"),
    ("severity_2_moderate", Path("pneumonia") / "severity_2_moderate"),
    ("severity_3_severe", Path("pneumonia") / "severity_3_severe"),
]


def is_severity_split_layout(data_root: Path) -> bool:
    if not data_root.is_dir():
        return False
    for _name, rel in SEVERITY_SPLIT_CLASS_DIRS:
        if not (data_root / rel).is_dir():
            return False
    return True


def infer_class_to_idx(data_root: Path) -> Dict[str, int]:
    """Resolve class names → index without building loaders (single split later)."""
    if is_severity_split_layout(data_root):
        _samples, cti = collect_severity_split_samples(data_root)
        return cti
    return ImageFolder(data_root).class_to_idx


def collect_severity_split_samples(data_root: Path) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    class_names = [name for name, _ in SEVERITY_SPLIT_CLASS_DIRS]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    for class_name, rel in SEVERITY_SPLIT_CLASS_DIRS:
        d = data_root / rel
        label = class_to_idx[class_name]
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                samples.append((str(p.resolve()), label))
    return samples, class_to_idx


class SamplesListDataset(Dataset):
    """Flat list of (path, class_index) like ImageFolder; RGB loader for grayscale JPEG."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_to_idx: Dict[str, int],
        loader=pil_loader_rgb,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)


class SplitImageFolder(Dataset):
    """
    A view over an ImageFolder or SamplesListDataset restricted to a subset of indices,
    with its own transform and a `samples` attribute for statistics.
    """

    def __init__(
        self,
        base_dataset: Union[ImageFolder, "SamplesListDataset"],
        indices,
        transform,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

        self.samples = [self.base_dataset.samples[i] for i in self.indices]
        self.class_to_idx = self.base_dataset.class_to_idx
        self.loader = getattr(base_dataset, "loader", pil_loader_rgb)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, target = self.base_dataset.samples[real_idx]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def count_files_per_class(dataset: ImageFolder, idx_to_class: Dict[int, str]) -> Dict[str, int]:
    """Count files per class in a dataset."""
    class_counts = {cls: 0 for cls in idx_to_class.values()}
    for _, class_idx in dataset.samples:
        class_name = idx_to_class[class_idx]
        class_counts[class_name] += 1
    return class_counts


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    weights=None,
    val_fraction: float = 0.10,
    balanced_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, Any]]:
    """
    Build train/val/test dataloaders.

    If ``data_root`` matches the RSNA severity layout (``normal/`` + ``pneumonia/severity_*``),
    loads four classes. Otherwise uses torchvision ``ImageFolder`` (e.g. NORMAL/PNEUMONIA).

    Stratified split per class: shuffle, assign ``val_fraction`` of each class to val/test
    (same indices for both), remainder to train. Tiny classes clamp so at least one sample
    stays in train when ``n >= 2``.
    """
    train_tf, eval_tf = build_transforms(image_size, weights)

    if not data_root.exists():
        raise FileNotFoundError(f"Could not locate {data_root}.")

    use_severity_layout = is_severity_split_layout(data_root)
    if use_severity_layout:
        samples, class_to_idx = collect_severity_split_samples(data_root)
        if not samples:
            raise FileNotFoundError(
                f"No images found under severity split folders in {data_root}. "
                "Run create_severity_split_jpg_dataset.py first."
            )
        base_dataset = SamplesListDataset(samples, class_to_idx)
    else:
        base_dataset = ImageFolder(data_root)
        class_to_idx = base_dataset.class_to_idx

    idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}

    # Build per-class index lists
    indices_by_class: Dict[int, list] = {class_idx: [] for class_idx in class_to_idx.values()}
    for idx, (_, label) in enumerate(base_dataset.samples):
        indices_by_class[label].append(idx)

    per_class_before = {
        idx_to_class[class_idx]: len(idxs) for class_idx, idxs in indices_by_class.items()
    }

    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    train_indices = []
    val_indices = []

    # Stratified split per class; val and test share the same indices
    for class_idx, idx_list in indices_by_class.items():
        random.shuffle(idx_list)
        n = len(idx_list)
        if n == 0:
            continue

        n_eval = int(round(val_fraction * n))
        n_eval = max(0, min(n_eval, n))
        if n >= 2 and n_eval >= n:
            n_eval = n - 1  # keep at least one sample for training

        train_part = idx_list[n_eval:]
        eval_part = idx_list[:n_eval]

        train_indices.extend(train_part)
        val_indices.extend(eval_part)

    def _count_by_class(indices: List[int]) -> Dict[str, int]:
        out: Dict[str, int] = {idx_to_class[i]: 0 for i in idx_to_class}
        for j in indices:
            _, lab = base_dataset.samples[j]
            out[idx_to_class[lab]] += 1
        return out

    split_summary: Dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "dataset_layout": "rsna_severity_4class" if use_severity_layout else "imagefolder",
        "num_classes": len(class_to_idx),
        "class_to_idx": dict(class_to_idx),
        "total_files_loaded": len(base_dataset.samples),
        "files_per_class_before_split": dict(sorted(per_class_before.items(), key=lambda x: x[0])),
        "files_per_class_train": dict(sorted(_count_by_class(train_indices).items(), key=lambda x: x[0])),
        "files_per_class_val": dict(sorted(_count_by_class(val_indices).items(), key=lambda x: x[0])),
        "files_per_class_test": dict(sorted(_count_by_class(val_indices).items(), key=lambda x: x[0])),
        "val_fraction": val_fraction,
        "train_fraction_target": round(1.0 - val_fraction, 6),
        "split_rule": (
            f"Stratified per class: shuffle with script RNG; "
            f"~{100 * val_fraction:.2f}% of each class -> val (test identical to val); "
            f"remainder -> train. Tiny classes may clamp so at least one train sample remains."
        ),
        "train_split_size": len(train_indices),
        "val_split_size": len(val_indices),
        "test_split_size": len(val_indices),
        "val_and_test_identical": True,
    }

    # Build split datasets with appropriate transforms
    train_dataset = SplitImageFolder(base_dataset, train_indices, train_tf)
    val_dataset = SplitImageFolder(base_dataset, val_indices, eval_tf)
    # Test dataset uses the exact same indices as validation
    test_dataset = SplitImageFolder(base_dataset, val_indices, eval_tf)

    # Optional balanced sampling for train split (class-frequency inverse weights).
    train_sampler = None
    if balanced_sampler and len(train_dataset.samples) > 0:
        class_counts: Dict[int, int] = {class_idx: 0 for class_idx in class_to_idx.values()}
        for _, cls_idx in train_dataset.samples:
            class_counts[cls_idx] += 1
        sample_weights = [
            (1.0 / float(class_counts[cls_idx])) if class_counts[cls_idx] > 0 else 0.0
            for _, cls_idx in train_dataset.samples
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    loaders = []
    for dataset, shuffle, sampler in (
        (train_dataset, True, train_sampler),
        (val_dataset, False, None),
        (test_dataset, False, None),
    ):
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=bool(shuffle and sampler is None),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        )

    return (*loaders, class_to_idx, split_summary)


def log_dataset_split(split_summary: Dict[str, Any], output_path: Path | None = None) -> None:
    """Print and optionally save dataset load + split details."""
    print("\n" + "=" * 72)
    print("DATASET LOAD & SPLIT")
    print("=" * 72)
    print(f"  data_root:              {split_summary['data_root']}")
    print(f"  layout:                 {split_summary['dataset_layout']}")
    print(f"  val_fraction (per cls): {split_summary['val_fraction']}")
    print(f"  train_fraction (~):     {split_summary['train_fraction_target']}")
    if "split_seed_used" in split_summary:
        print(f"  split_seed (shuffle):    {split_summary['split_seed_used']}")
    if "training_seed_after_split" in split_summary:
        print(f"  training_seed:           {split_summary['training_seed_after_split']}")
    print(f"  num_classes:            {split_summary['num_classes']}")
    print(f"  total_files_loaded:     {split_summary['total_files_loaded']:,}")
    print("  files_per_class (before split):")
    for name, n in split_summary["files_per_class_before_split"].items():
        print(f"    - {name}: {n:,}")
    print(f"  train / val / test:     {split_summary['train_split_size']:,} / "
          f"{split_summary['val_split_size']:,} / {split_summary['test_split_size']:,}")
    print("  files_per_class (train):")
    for name, n in split_summary["files_per_class_train"].items():
        print(f"    - {name}: {n:,}")
    print("  files_per_class (val == test):")
    for name, n in split_summary["files_per_class_val"].items():
        print(f"    - {name}: {n:,}")
    print(f"  split_rule: {split_summary['split_rule']}")
    print("=" * 72 + "\n")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(split_summary, fh, indent=2)
        print(f"Saved dataset split log: {output_path}\n")


def build_model(
    variant: str,
    num_classes: int,
    freeze_backbone: bool,
    use_pretrained: bool,
    head_depth: int = 1,
    head_hidden_dim: int | None = None,
    head_dropout: float = 0.0,
    head_activation: str = "relu",
) -> Tuple[VisionTransformer, object]:
    """
    Build a Vision Transformer model with configurable classification head.
    
    Args:
        variant: ViT variant name
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the backbone
        use_pretrained: Whether to use pretrained weights
        head_depth: Number of hidden layers in classification head (1 = single linear layer)
        head_hidden_dim: Hidden dimension size (None = use in_features)
        head_dropout: Dropout rate for hidden layers
        head_activation: Activation function ('relu', 'gelu', 'tanh')
    """
    registry = {
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
        "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
        "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
        "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1),
    }
    constructor, default_weights = registry[variant]
    weights = default_weights if use_pretrained else None
    model = constructor(weights=weights)

    # Build classification head
    in_features = model.heads.head.in_features
    hidden_dim = head_hidden_dim if head_hidden_dim is not None else in_features
    
    # Activation function mapping
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    activation = activation_map.get(head_activation.lower(), nn.ReLU())
    
    if head_depth == 1:
        # Single linear layer (original behavior)
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        # Multi-layer head
        layers = []
        # First layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if head_depth > 2:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)
        if head_dropout > 0:
            layers.append(nn.Dropout(head_dropout))
        
        # Hidden layers
        for _ in range(head_depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
        
        # Final output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        model.heads.head = nn.Sequential(*layers)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    return model, weights


def pretrained_weights_for_variant(variant: str, use_pretrained: bool):
    """
    Weights enum used by torchvision ViT constructors (for normalization in ``build_transforms``).
    Lets us build dataloaders before final ``build_model`` once ``num_classes`` is known.
    """
    registry = {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1,
        "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1,
        "vit_l_32": ViT_L_32_Weights.IMAGENET1K_V1,
    }
    w = registry[variant]
    return w if use_pretrained else None


def _resize_pos_embed_vit(
    pos_embed: torch.Tensor,  # (1, old_tokens, dim)
    old_grid: int,
    new_grid: int,
) -> torch.Tensor:
    """Interpolate ViT CLS+grid positional embeddings to a new spatial grid."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Unexpected pos_embed shape: {tuple(pos_embed.shape)}")
    cls_tok = pos_embed[:, :1, :]
    grid_tok = pos_embed[:, 1:, :]
    dim = grid_tok.shape[-1]
    grid_tok = grid_tok.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    grid_tok = F.interpolate(
        grid_tok,
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    )
    grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat([cls_tok, grid_tok], dim=1)


def adapt_torchvision_vit_to_image_size(model: nn.Module, image_size: int) -> nn.Module:
    """
    Allow torchvision VisionTransformer at a new square ``image_size`` (e.g. 512):
    updates ``model.image_size`` and resizes ``encoder.pos_embedding`` when the token grid changes.
    No-op if the architecture does not look like torchvision ViT.
    """
    if not (hasattr(model, "image_size") and hasattr(model, "conv_proj") and hasattr(model, "encoder")):
        return model
    if not hasattr(model.encoder, "pos_embedding"):
        return model
    conv = model.conv_proj
    if not hasattr(conv, "kernel_size") or not hasattr(conv, "stride"):
        return model
    patch = conv.kernel_size[0]
    if image_size % patch != 0:
        raise ValueError(f"--image-size {image_size} must be divisible by patch_size={patch} for this ViT")

    old_img = int(getattr(model, "image_size"))
    old_grid = old_img // patch
    new_grid = image_size // patch
    model.image_size = image_size

    with torch.no_grad():
        pos = model.encoder.pos_embedding
        old_tokens = pos.shape[1]
        new_tokens = 1 + new_grid * new_grid
        if old_tokens != new_tokens:
            new_pos = _resize_pos_embed_vit(pos.data, old_grid=old_grid, new_grid=new_grid)
            model.encoder.pos_embedding = nn.Parameter(new_pos)
    return model


_TQDM_MISSING_WARNED = False


def iter_with_progress(iterable, *, enabled: bool, desc: str | None = None):
    """Wrap iterable with tqdm when enabled (requires tqdm installed)."""
    global _TQDM_MISSING_WARNED
    if not enabled:
        return iterable
    try:
        import sys

        from tqdm import tqdm

        # stderr so progress still shows when stdout is piped; tqdm works without a TTY (e.g. Docker logs).
        return tqdm(iterable, desc=desc or "", leave=True, file=sys.stderr)
    except ImportError:
        if not _TQDM_MISSING_WARNED:
            print(
                "train_vit: tqdm is not installed — batch progress bars disabled. "
                "`pip install tqdm` to enable (--progress).",
                flush=True,
            )
            _TQDM_MISSING_WARNED = True
        return iterable


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum()
    return correct.float() / target.size(0)


def _summarize_gpu_epoch_stats(
    batch_times_ms: List[float], batch_mem_mib: List[float], device: torch.device
) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}

    stats: Dict[str, float] = {}
    if batch_times_ms:
        bt = np.asarray(batch_times_ms, dtype=np.float64)
        stats["gpu_batch_time_median_ms"] = float(np.median(bt))
        stats["gpu_batch_time_max_ms"] = float(np.max(bt))
    if batch_mem_mib:
        bm = np.asarray(batch_mem_mib, dtype=np.float64)
        stats["gpu_mem_median_mib"] = float(np.median(bm))
        stats["gpu_mem_max_mib"] = float(np.max(bm))

    stats["gpu_peak_allocated_mib"] = float(
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )
    stats["gpu_peak_reserved_mib"] = float(
        torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    )
    return stats


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def train_one_epoch(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    *,
    show_batch_progress: bool = False,
    batch_progress_desc: str = "Train batches",
    log_interval: int = 0,
) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batch_times_ms: List[float] = []
    batch_mem_mib: List[float] = []
    use_gpu_stats = device.type == "cuda" and torch.cuda.is_available()
    if use_gpu_stats:
        torch.cuda.reset_peak_memory_stats(device)
    loop_start = time.time()
    it = iter_with_progress(
        loader, enabled=show_batch_progress, desc=batch_progress_desc
    )
    num_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(it, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_gpu_stats:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

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

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets).item() * images.size(0)
        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            seen = min(batch_idx * loader.batch_size, len(loader.dataset))
            avg_loss = running_loss / max(seen, 1)
            avg_acc = running_acc / max(seen, 1)
            elapsed = time.time() - loop_start
            sec_per_batch = elapsed / batch_idx
            eta = sec_per_batch * (num_batches - batch_idx)
            batches_per_sec = batch_idx / max(elapsed, 1e-9)
            print(
                f"  {batch_progress_desc} [{batch_idx}/{num_batches}] "
                f"({seen}/{len(loader.dataset)}) avg_acc={avg_acc:.4f} avg_loss={avg_loss:.4f} "
                f"elapsed={_format_mmss(elapsed)} eta={_format_mmss(eta)} "
                f"speed={batches_per_sec:.2f} batches/s",
                flush=True,
            )
        if use_gpu_stats:
            end_event.record()
            end_event.synchronize()
            batch_times_ms.append(start_event.elapsed_time(end_event))
            batch_mem_mib.append(
                float(torch.cuda.memory_allocated(device) / (1024 ** 2))
            )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    gpu_stats = _summarize_gpu_epoch_stats(batch_times_ms, batch_mem_mib, device)
    return epoch_loss, epoch_acc, gpu_stats


@torch.no_grad()
def evaluate(
    model: VisionTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    show_batch_progress: bool = False,
    batch_progress_desc: str = "Eval batches",
    log_interval: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_targets = []
    batch_times_ms: List[float] = []
    batch_mem_mib: List[float] = []
    use_gpu_stats = device.type == "cuda" and torch.cuda.is_available()
    if use_gpu_stats:
        torch.cuda.reset_peak_memory_stats(device)
    loop_start = time.time()
    it = iter_with_progress(
        loader, enabled=show_batch_progress, desc=batch_progress_desc
    )
    num_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(it, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_gpu_stats:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets).item() * images.size(0)
        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            seen = min(batch_idx * loader.batch_size, len(loader.dataset))
            avg_loss = running_loss / max(seen, 1)
            avg_acc = running_acc / max(seen, 1)
            elapsed = time.time() - loop_start
            sec_per_batch = elapsed / batch_idx
            eta = sec_per_batch * (num_batches - batch_idx)
            batches_per_sec = batch_idx / max(elapsed, 1e-9)
            print(
                f"  {batch_progress_desc} [{batch_idx}/{num_batches}] "
                f"({seen}/{len(loader.dataset)}) avg_acc={avg_acc:.4f} avg_loss={avg_loss:.4f} "
                f"elapsed={_format_mmss(elapsed)} eta={_format_mmss(eta)} "
                f"speed={batches_per_sec:.2f} batches/s",
                flush=True,
            )
        if use_gpu_stats:
            end_event.record()
            end_event.synchronize()
            batch_times_ms.append(start_event.elapsed_time(end_event))
            batch_mem_mib.append(
                float(torch.cuda.memory_allocated(device) / (1024 ** 2))
            )
        
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    gpu_stats = _summarize_gpu_epoch_stats(batch_times_ms, batch_mem_mib, device)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets), gpu_stats


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Per-class ``accuracy_class_k`` is the fraction of samples with true label *k*
    that are predicted as *k* (confusion-matrix diagonal divided by row sum);
    for multi-class this equals per-class recall / sensitivity.
    """
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class and macro-averaged metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Macro-averaged metrics
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Weighted-averaged metrics
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = precision[i] if i < len(precision) else 0.0
        metrics[f"recall_class_{i}"] = recall[i] if i < len(recall) else 0.0
        metrics[f"f1_class_{i}"] = f1[i] if i < len(f1) else 0.0
    
    # Confusion matrix (fixed label order so shape is always num_classes²)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if num_classes == 2:
        # Binary classification: TP, TN, FP, FN
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
    else:
        # Multi-class: store full confusion matrix values
        for i in range(num_classes):
            for j in range(num_classes):
                metrics[f"cm_{i}_{j}"] = int(cm[i, j])

    for i in range(num_classes):
        row_sum = float(cm[i].sum())
        metrics[f"accuracy_class_{i}"] = (
            float(cm[i, i] / row_sum) if row_sum > 0.0 else 0.0
        )

    return metrics


def save_confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_csv: Path,
    output_png: Path,
    title: str,
) -> None:
    """Save confusion matrix as CSV and heatmap PNG."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual\\pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())

    fig_w = max(6, len(class_names) * 1.2)
    fig_h = max(5, len(class_names) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_checkpoint(
    model: VisionTransformer,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    class_to_idx: Dict[str, int],
    output_dir: Path,
    filename: str,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "class_to_idx": class_to_idx,
    }
    torch.save(state, output_dir / filename)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Dataset root: RSNA severity split (normal/ + pneumonia/severity_*) or ImageFolder (e.g. NORMAL/PNEUMONIA).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input resolution. Larger images may require more GPU memory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size per iteration.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of background DataLoader workers.",
    )
    parser.add_argument(
        "--vit-variant",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
        help="Vision Transformer backbone.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretraining weights.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (automatic).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ViT backbone and train only classification head.",
    )
    parser.add_argument(
        "--head-depth",
        type=int,
        default=1,
        help="Number of hidden layers in classification head (1 = single linear layer).",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension size for classification head (None = use in_features).",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.0,
        help="Dropout rate for hidden layers in classification head.",
    )
    parser.add_argument(
        "--head-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh"],
        help="Activation function for classification head.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/vit",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (Python/NumPy/torch): controls train/val split shuffling and training stochasticity. Same seed => same split.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience based on validation loss. Set to None to disable early stopping.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help=(
            "Per-class fraction for validation (test pool matches val). "
            "Example: 0.15 → ~85%% train / 15%% val per class. Must be in [0, 1)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Seed for Python's random.shuffle when building stratified train/val (val and test "
            "use the same held-out indices). Default: reuse --seed. Override to fix the split while "
            "changing training randomness via --seed."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show per-batch tqdm bars during train/val/test passes (requires tqdm).",
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
        help=(
            "Class weights for CrossEntropy on train split: "
            "'none' (default), 'inverse' (1/freq), 'sqrt_inverse' (1/sqrt(freq))."
        ),
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for train loader (class-frequency inverse weights).",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_f1_macro"],
        help="Metric used to select and save best model checkpoint.",
    )
    return parser.parse_args(argv)


def _write_val_test_images_csv(
    val_loader: DataLoader,
    test_loader: DataLoader,
    idx_to_class: Dict[int, str],
    csv_path: Path,
) -> None:
    """Mirror ``vit_nas`` export: list val/test images after stratified split."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "image_path", "class_name", "class_index"])
        for split_name, dataset in (("val", val_loader.dataset), ("test", test_loader.dataset)):
            for path, class_idx in dataset.samples:
                writer.writerow([split_name, str(path), idx_to_class[class_idx], class_idx])
    print(f"Saved validation/test image list to {csv_path}")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in [0, 1).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_enum = pretrained_weights_for_variant(args.vit_variant, use_pretrained=not args.no_pretrained)

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    random.seed(split_seed)
    train_loader, val_loader, test_loader, class_to_idx, split_summary = build_dataloaders(
        data_root=Path(args.data_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weights=weight_enum,
        val_fraction=args.val_fraction,
        balanced_sampler=args.balanced_sampler,
    )
    split_summary["split_seed_used"] = split_seed
    split_summary["training_seed_after_split"] = args.seed
    set_seed(args.seed)
    print(f"Stratified split: split_seed={split_seed}; training RNG seed={args.seed}")

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    print(f"Classes: {idx_to_class}")

    model, weights = build_model(
        args.vit_variant,
        num_classes=len(class_to_idx),
        freeze_backbone=args.freeze_backbone,
        use_pretrained=not args.no_pretrained,
        head_depth=args.head_depth,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_activation=args.head_activation,
    )
    model = adapt_torchvision_vit_to_image_size(model, args.image_size)

    log_dataset_split(split_summary, output_dir / "dataset_split.json")
    _write_val_test_images_csv(val_loader, test_loader, idx_to_class, output_dir / "val_test_images.csv")

    # Print dataset statistics (matches split_summary; keeps epoch-table orientation)
    print("\nDataset Statistics (same split as dataset_split.json):")
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
    print()
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weight_tensor = None
    if args.class_weight_mode != "none":
        train_counts = split_summary["files_per_class_train"]
        counts = np.array(
            [float(train_counts.get(idx_to_class[i], 0)) for i in range(len(idx_to_class))],
            dtype=np.float64,
        )
        if np.any(counts <= 0):
            raise RuntimeError(f"Invalid train class counts for class weighting: {counts.tolist()}")
        if args.class_weight_mode == "inverse":
            w = 1.0 / counts
        else:  # sqrt_inverse
            w = 1.0 / np.sqrt(counts)
        # Normalize to keep average weight around 1.0
        w = w * (len(w) / float(w.sum()))
        class_weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Using class weights ({args.class_weight_mode}): {w.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    scaler = create_grad_scaler(args.amp, device)

    best_val_loss = math.inf
    best_metric_value = math.inf if args.best_metric == "val_loss" else -math.inf
    best_epoch = -1
    history = []
    epochs_no_improve = 0
    num_classes = len(class_to_idx)

    print(
        "Per-epoch logging (stdout + training_metrics.csv + training_history.json): "
        "train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, lr, epoch_time; "
        "plus sklearn metrics prefixed train_/val_/test_ (accuracy, macro/weighted P/R/F1, "
        "per-class precision/recall/F1, per-class accuracy_class_* , confusion matrix entries).\n"
    )

    csv_path = output_dir / "training_metrics.csv"
    csv_initialized = False
    last_epoch = 0
    last_train_preds = np.array([], dtype=np.int64)
    last_train_targets = np.array([], dtype=np.int64)
    last_val_preds = np.array([], dtype=np.int64)
    last_val_targets = np.array([], dtype=np.int64)
    last_test_preds = np.array([], dtype=np.int64)
    last_test_targets = np.array([], dtype=np.int64)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc, train_runtime_stats = train_one_epoch(
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

        val_loss, val_acc, val_preds, val_targets, val_runtime_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_batch_progress=args.progress,
            batch_progress_desc=f"Epoch {epoch}/{args.epochs} val",
            log_interval=args.log_interval,
        )
        test_loss, test_acc, test_preds, test_targets, test_runtime_stats = evaluate(
            model,
            test_loader,
            criterion,
            device,
            show_batch_progress=args.progress,
            batch_progress_desc=f"Epoch {epoch}/{args.epochs} test",
            log_interval=args.log_interval,
        )

        # Full train pass in eval mode for sklearn metrics (can differ slightly from batch train_acc).
        model.eval()
        train_preds = []
        train_targets = []
        with torch.no_grad():
            _tr_it = iter_with_progress(
                train_loader,
                enabled=args.progress,
                desc=f"Epoch {epoch}/{args.epochs} train (metrics)",
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
        
        scheduler.step()

        epoch_time = time.time() - start_time
        train_class_acc_str = "  ".join(
            f"{idx_to_class[i]}={train_metrics.get(f'accuracy_class_{i}', 0.0):.4f}"
            for i in range(num_classes)
        )
        val_class_acc_str = "  ".join(
            f"{idx_to_class[i]}={val_metrics.get(f'accuracy_class_{i}', 0.0):.4f}"
            for i in range(num_classes)
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  | "
            f"lr={scheduler.get_last_lr()[0]:.2e}  | "
            f"{epoch_time:.1f}s",
            flush=True,
        )
        print(f"  train_class_acc: {train_class_acc_str}", flush=True)
        print(f"  val_class_acc:   {val_class_acc_str}", flush=True)

        # Combine all metrics into history
        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_time,
        }
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            epoch_history[f"train_{key}"] = value
        
        # Add val metrics with prefix
        for key, value in val_metrics.items():
            epoch_history[f"val_{key}"] = value
        
        # Add test metrics with prefix
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
        write_mode = "a" if csv_initialized else "w"
        with csv_path.open(write_mode, newline="", encoding="utf-8") as fh:
            fieldnames = list(epoch_history.keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not csv_initialized:
                writer.writeheader()
                csv_initialized = True
            writer.writerow(epoch_history)

        best_val_loss = min(best_val_loss, val_loss)
        selection_value = val_loss if args.best_metric == "val_loss" else float(val_metrics["f1_macro"])
        is_better = (
            (selection_value < best_metric_value)
            if args.best_metric == "val_loss"
            else (selection_value > best_metric_value)
        )

        if is_better:
            best_metric_value = selection_value
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                class_to_idx,
                output_dir,
                "best_model.pt",
            )
        else:
            epochs_no_improve += 1
            if args.patience is not None and epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved training history to {history_path}")
    
    if history:
        print(f"Saved training metrics to {csv_path}")

    print(f"Best model at epoch {best_epoch}")

    # Save the true last-epoch model state (before loading best checkpoint).
    last_state_path = output_dir / "last_model.pt"
    save_checkpoint(
        model,
        optimizer,
        last_epoch,
        best_val_loss,
        class_to_idx,
        output_dir,
        last_state_path.name,
    )
    print(f"Saved last-epoch checkpoint to {last_state_path}")

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

    print("\nEvaluating best model on the test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get final train, val, and test accuracies from best epoch
    best_history = next((h for h in history if h["epoch"] == best_epoch), None)
    if best_history:
        print(f"\nFinal Results (Best Model at Epoch {best_epoch}):")
        print(f"  Train Accuracy: {best_history['train_acc']:.4f}")
        print(f"  Val Accuracy: {best_history['val_acc']:.4f}")
        print(f"  Test Accuracy: {best_history.get('test_acc', test_acc):.4f}")

    final_state_path = output_dir / "final_model.pt"
    save_checkpoint(
        model,
        optimizer,
        epoch,
        best_val_loss,
        class_to_idx,
        output_dir,
        final_state_path.name,
    )
    print(f"Saved final checkpoint to {final_state_path}")


if __name__ == "__main__":
    main()
