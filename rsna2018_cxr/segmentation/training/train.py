"""Supervised compact U-Net on images that already have lung masks.

The severity cohort does not ship lung masks, and this trainer did not write
the iteration-1 cohort masks. Those came from the external VGG11 checkpoint.
Labeled films that are not already 1024x1024 are resized to that grid here
only. The cohort segmenter never resizes. Silver or pseudo labels are not
human labels unless the caller opts in with an explicit sample weight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rsna2018_cxr.segmentation.models.compact_unet import CompactUNet


def pair_image_masks(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    masks: dict[str, Path] = {}
    for path in Path(masks_dir).iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = path.stem[:-5] if path.stem.endswith("_mask") else path.stem
        masks[stem] = path
    pairs = []
    for path in sorted(Path(images_dir).iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        mask = masks.get(path.stem)
        if mask is not None:
            pairs.append((path, mask))
    return pairs


class MaskDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int):
        self.pairs = pairs
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image_tensor = _load_gray(image_path, self.image_size)
        mask_tensor = _load_mask(mask_path, self.image_size)
        return image_tensor, mask_tensor


def _load_gray(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != (image_size, image_size):
            gray = gray.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def _load_mask(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as mask:
        binary = mask.convert("L")
        if binary.size != (image_size, image_size):
            binary = binary.resize((image_size, image_size), Image.NEAREST)
        array = (np.asarray(binary) > 127).astype(np.float32)
    return torch.from_numpy(array)[None]


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(2, 3))
    denominator = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce_weight * bce + (1.0 - bce_weight) * dice


def batch_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3))
    denominator = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    scores = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return float(scores.mean().item())


def split_pairs(pairs: list[tuple[Path, Path]], seed: int = 2026, train_fraction: float = 0.8):
    from rsna2018_cxr.segmentation.data.splits import assign_split

    train, val = [], []
    for image_path, mask_path in pairs:
        bucket = assign_split(image_path.stem, seed=seed, train=train_fraction, val=1.0 - train_fraction)
        if bucket == "train":
            train.append((image_path, mask_path))
        else:
            val.append((image_path, mask_path))
    if not train or not val:
        raise RuntimeError(f"Split produced train={len(train)} val={len(val)}. Need both.")
    return train, val


def train_compact_unet(
    images_dir: Path,
    masks_dir: Path,
    output_checkpoint: Path,
    *,
    image_size: int = 1024,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    seed: int = 2026,
    device: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    pairs = pair_image_masks(images_dir, masks_dir)
    if len(pairs) < 4:
        raise RuntimeError(f"Need at least 4 image/mask pairs, found {len(pairs)}.")
    train_pairs, val_pairs = split_pairs(pairs, seed=seed)
    use_cuda = torch.cuda.is_available() if device is None else device.startswith("cuda")
    chosen = torch.device(device or ("cuda" if use_cuda else "cpu"))
    train_loader = DataLoader(
        MaskDataset(train_pairs, image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=chosen.type == "cuda",
    )
    val_loader = DataLoader(
        MaskDataset(val_pairs, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=chosen.type == "cuda",
    )
    model = CompactUNet().to(chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_dice = -1.0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(chosen, non_blocking=True)
            masks = masks.to(chosen, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        val_scores = []
        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(chosen, non_blocking=True)
                masks = masks.to(chosen, non_blocking=True)
                val_scores.append(batch_dice(model(images), masks))
        val_dice = float(np.mean(val_scores))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice}
        history.append(row)
        print(f"epoch {epoch}/{epochs}  train_loss {row['train_loss']:.4f}  val_dice {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "compact_unet",
                    "image_size": image_size,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "n_train": len(train_pairs),
                    "n_val": len(val_pairs),
                },
                output_checkpoint,
            )
    return {"best_val_dice": best_dice, "history": history, "checkpoint": str(output_checkpoint), "n_pairs": len(pairs)}
