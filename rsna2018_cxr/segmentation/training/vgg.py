"""Train the VGG11-BN U-Net that segmented the severity cohort.

This is the local training setup from the earlier chest X-ray project:
BCE-with-logits, Adam at 5e-4, batch 4, 1024 input, mixed precision, and the
best epoch chosen by validation loss. The saved file keeps the weights and
the scalar metrics. It does not keep the Adam state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rsna2018_cxr.segmentation.constants import NATIVE_HW
from rsna2018_cxr.segmentation.models.vgg_unet import UNetVGG11


def load_split_ids(path: Path) -> dict[str, list[str]]:
    groups = {"train": [], "val": [], "test": []}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["split"]].append(row["image_id"])
    if not groups["train"] or not groups["val"]:
        raise RuntimeError(f"Split file {path} needs train and val ids.")
    return groups


class _PairDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, image_ids: list[str], augment: bool):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_ids = list(image_ids)
        self.augment = augment
        self.size = NATIVE_HW[0]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        with Image.open(self.images_dir / f"{image_id}.png") as image:
            gray = image.convert("L").resize((self.size, self.size), Image.BILINEAR)
        mask_path = self.masks_dir / f"{image_id}.png"
        if mask_path.is_file():
            with Image.open(mask_path) as mask:
                binary = mask.convert("L").resize((self.size, self.size), Image.NEAREST)
        else:
            binary = Image.new("L", (self.size, self.size), 0)
        image_tensor = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)[None]
        mask_tensor = torch.from_numpy((np.asarray(binary) > 127).astype(np.float32))[None]
        if self.augment:
            image_tensor, mask_tensor = _augment(image_tensor, mask_tensor)
        return image_tensor, mask_tensor


def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same draws as the original loader: flips, a small rotation, sometimes zoom."""
    import torchvision.transforms.functional as TF

    if np.random.random() > 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if np.random.random() > 0.3:
        image, mask = TF.vflip(image), TF.vflip(mask)
    angle = float(np.random.uniform(-10, 10))
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    if np.random.random() > 0.5:
        translate = (float(np.random.uniform(-0.1, 0.1)), float(np.random.uniform(-0.1, 0.1)))
        scale = float(np.random.uniform(0.9, 1.1))
        image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
        mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)
    return image, (mask > 0.5).float()


def _dice_jaccard(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    intersection = (pred * truth).sum(dim=1)
    denom = pred.sum(dim=1) + truth.sum(dim=1)
    dice = ((2.0 * intersection) / (denom + 1e-7)).mean()
    union = denom - intersection
    jaccard = (intersection / (union + 1e-7)).mean()
    return float(dice.item()), float(jaccard.item())


def train_vgg11(
    images_dir: Path,
    masks_dir: Path,
    splits_csv: Path,
    output_checkpoint: Path,
    *,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    torch.manual_seed(42)
    groups = load_split_ids(splits_csv)
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(
        _PairDataset(images_dir, masks_dir, groups["train"], augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        _PairDataset(images_dir, masks_dir, groups["val"], augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = UNetVGG11(n_channels=1, n_classes=1, use_batch_norm=True, bilinear=True).to(chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    scaler = torch.amp.GradScaler("cuda", enabled=chosen.type == "cuda")
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(chosen)
            masks = masks.to(chosen)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=chosen.type, enabled=chosen.type == "cuda"):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        val_dice = []
        val_jaccard = []
        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(chosen)
                masks = masks.to(chosen)
                logits = model(images)
                val_losses.append(float(F.binary_cross_entropy_with_logits(logits, masks).item()))
                dice, jaccard = _dice_jaccard(logits, masks)
                val_dice.append(dice)
                val_jaccard.append(jaccard)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_dice": float(np.mean(val_dice)),
            "val_jaccard": float(np.mean(val_jaccard)),
        }
        history.append(row)
        scheduler.step(row["val_loss"])
        print(
            f"epoch {epoch}/{epochs}  train_loss {row['train_loss']:.4f}  "
            f"val_loss {row['val_loss']:.4f}  val_dice {row['val_dice']:.4f}"
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_dice": row["val_dice"],
                    "val_jaccard": row["val_jaccard"],
                    "architecture": "unet_vgg11",
                    "image_size": NATIVE_HW[0],
                },
                output_checkpoint,
            )
    metrics_path = output_checkpoint.with_name("metrics_history.csv")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": str(output_checkpoint),
        "n_train": len(groups["train"]),
        "n_val": len(groups["val"]),
        "n_test": len(groups["test"]),
    }
