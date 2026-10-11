"""Train one condition's 2D (x, y) regressor."""

from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import (
    BACKBONE_CONVNEXT_TINY,
    IMAGE_SIZE_DEFAULT,
    LUMBAR_LEVELS,
)
from rsna2024_lumbar.landmarks.dataset import LandmarkSample, list_condition_samples, load_rgb_slice, split_samples
from rsna2024_lumbar.landmarks.models import build_regressor, landmark_loss, pixel_errors


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("pip install -e '.[landmarks]'") from exc
    return torch, DataLoader, Dataset


def _make_loader(samples: list[LandmarkSample], image_size: int, batch_size: int, shuffle: bool):
    torch, DataLoader, Dataset = _require_torch()

    class _DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            item = samples[idx]
            image = torch.from_numpy(load_rgb_slice(item.path, image_size))
            return {
                "image": image,
                "xy": torch.from_numpy(item.xy01),
                "visible": torch.from_numpy(item.visible),
                "width": item.width,
                "height": item.height,
            }

    return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, *, train: bool, optimizer=None, device):
    torch, _DataLoader, _Dataset = _require_torch()
    model.train(train)
    totals = {"loss": 0.0, "l1": 0.0, "mae_px": 0.0, "pck5": 0.0, "n": 0}
    for batch in loader:
        images = batch["image"].to(device)
        xy = batch["xy"].to(device)
        visible = batch["visible"].to(device)
        pred_xy, pred_vis = model(images)
        loss, parts = landmark_loss(pred_xy, pred_vis, xy, visible)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        err = pixel_errors(pred_xy, xy, visible, int(batch["width"][0]), int(batch["height"][0]))
        n = int(images.size(0))
        totals["loss"] += float(loss.detach()) * n
        totals["l1"] += parts["l1"] * n
        if err["mae_px"] == err["mae_px"]:
            totals["mae_px"] += err["mae_px"] * n
            totals["pck5"] += err["pck5"] * n
        totals["n"] += n
    count = max(totals["n"], 1)
    return {k: (totals[k] / count if k != "n" else totals[k]) for k in totals}


def train_condition(
    data_root: Path,
    output_dir: Path,
    condition: str,
    *,
    backbone: str = BACKBONE_CONVNEXT_TINY,
    image_size: int = IMAGE_SIZE_DEFAULT,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    pretrained: bool = False,
    max_studies: int | None = None,
) -> dict:
    torch, _DataLoader, _Dataset = _require_torch()
    print(f"listing {condition} samples under {data_root}", flush=True)
    samples = list_condition_samples(data_root, condition)
    if max_studies is not None:
        keep = sorted({s.study_id for s in samples})[: int(max_studies)]
        samples = [s for s in samples if s.study_id in set(keep)]
    if not samples:
        raise SystemExit(f"No landmark samples for {condition} under {data_root}.")
    parts = split_samples(samples, seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_regressor(backbone, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    best_val = float("inf")
    ckpt = output_dir / f"{condition}_{backbone}.pt"
    train_loader = _make_loader(parts["train"], image_size, batch_size, True)
    val_loader = _make_loader(parts["val"] or parts["train"], image_size, batch_size, False)
    print(
        f"split 70/15/15  train={len(parts['train'])} val={len(parts['val'])} "
        f"holdout={len(parts['holdout'])}  device={device}  backbone={backbone}",
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        train_m = _run_epoch(model, train_loader, train=True, optimizer=optimizer, device=device)
        val_m = _run_epoch(model, val_loader, train=False, device=device)
        row = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  train l1={train_m['l1']:.4f} mae_px={train_m['mae_px']:.1f} pck5={train_m['pck5']:.3f}  "
            f"val l1={val_m['l1']:.4f} mae_px={val_m['mae_px']:.1f} pck5={val_m['pck5']:.3f}",
            flush=True,
        )
        if val_m["l1"] < best_val:
            best_val = val_m["l1"]
            torch.save(
                {
                    "condition": condition,
                    "backbone": backbone,
                    "image_size": image_size,
                    "levels": list(LUMBAR_LEVELS),
                    "state_dict": model.state_dict(),
                    "val": val_m,
                },
                ckpt,
            )
    meta = {
        "condition": condition,
        "backbone": backbone,
        "image_size": image_size,
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_holdout": len(parts["holdout"]),
        "checkpoint": ckpt.as_posix(),
        "history": history,
        "split_seed": seed,
        "device": str(device),
    }
    (output_dir / f"{condition}_{backbone}_history.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
