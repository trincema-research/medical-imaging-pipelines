"""Train one condition's 2D (x, y) regressor."""

from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import (
    BACKBONE_CONVNEXT_TINY,
    IMAGE_SIZE_DEFAULT,
    LEVEL_DISPLAY,
    LUMBAR_LEVELS,
)
from rsna2024_lumbar.landmarks.dataset import (
    LandmarkSample,
    list_condition_samples,
    list_dense_slice_samples,
    load_rgb_slice,
    split_samples,
)
from rsna2024_lumbar.landmarks.models import build_regressor, coordinate_r2, landmark_loss, pixel_errors


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


def _fmt_r2(metrics: dict) -> str:
    return f"R2={metrics['r2']:.3f}  Rx={metrics['r2_x']:.3f}  Ry={metrics['r2_y']:.3f}"


def _fmt_level_r2(metrics: dict) -> str:
    parts: list[str] = []
    for level in LUMBAR_LEVELS:
        value = metrics.get(f"r2_{level}", float("nan"))
        parts.append(f"{LEVEL_DISPLAY[level]}={value:.3f}")
    return "  ".join(parts)


def _run_epoch(model, loader, *, train: bool, optimizer=None, device):
    torch, _DataLoader, _Dataset = _require_torch()
    model.train(train)
    totals = {"loss": 0.0, "l1": 0.0, "mae_px": 0.0, "pck5": 0.0, "n": 0}
    pred_chunks: list = []
    target_chunks: list = []
    vis_chunks: list = []
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
        pred_chunks.append(pred_xy.detach().cpu())
        target_chunks.append(xy.detach().cpu())
        vis_chunks.append(visible.detach().cpu())
    count = max(totals["n"], 1)
    out = {k: (totals[k] / count if k != "n" else totals[k]) for k in totals}
    if pred_chunks:
        out.update(coordinate_r2(torch.cat(pred_chunks), torch.cat(target_chunks), torch.cat(vis_chunks)))
    else:
        out.update({"r2": float("nan"), "r2_x": float("nan"), "r2_y": float("nan")})
    return out


def train_condition(
    data_root: Path,
    output_dir: Path,
    condition: str,
    *,
    backbone: str = BACKBONE_CONVNEXT_TINY,
    image_size: int = IMAGE_SIZE_DEFAULT,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    pretrained: bool = False,
    max_studies: int | None = None,
    all_slices: bool = False,
    dense_csv: Path | None = None,
) -> dict:
    torch, _DataLoader, _Dataset = _require_torch()
    print(f"listing {condition} samples under {data_root}", flush=True)
    if all_slices:
        if dense_csv is None:
            raise SystemExit("train --all-slices needs --dense-csv (run densify first).")
        samples = list_dense_slice_samples(data_root, condition, dense_csv)
    else:
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    history = []
    best_val = float("inf")
    ckpt = output_dir / f"{condition}_{backbone}.pt"
    train_loader = _make_loader(parts["train"], image_size, batch_size, True)
    val_loader = _make_loader(parts["val"] or parts["train"], image_size, batch_size, False)
    test_loader = _make_loader(parts["holdout"], image_size, batch_size, False) if parts["holdout"] else None
    levels = "  ".join(LEVEL_DISPLAY[level] for level in LUMBAR_LEVELS)
    print(
        f"one net, 5 level heads at once ({levels})  condition={condition}",
        flush=True,
    )
    print(
        f"split 70/15/15  train={len(parts['train'])} val={len(parts['val'])} "
        f"test={len(parts['holdout'])}  device={device}  backbone={backbone}  "
        f"lr={lr} wd={weight_decay}  epochs={epochs}",
        flush=True,
    )
    empty = {"r2": float("nan"), "r2_x": float("nan"), "r2_y": float("nan"), "mae_px": float("nan"), "pck5": float("nan"), "l1": float("nan"), "n": 0}
    for epoch in range(1, int(epochs) + 1):
        train_m = _run_epoch(model, train_loader, train=True, optimizer=optimizer, device=device)
        val_m = _run_epoch(model, val_loader, train=False, device=device)
        test_m = _run_epoch(model, test_loader, train=False, device=device) if test_loader else dict(empty)
        row = {"epoch": epoch, "train": train_m, "val": val_m, "test": test_m}
        history.append(row)
        print(f"epoch {epoch}/{epochs}  {condition}  5 levels at once", flush=True)
        for name, metrics in (("train", train_m), ("val", val_m), ("test", test_m)):
            print(
                f"  {name:<5}  {_fmt_r2(metrics)}  mae={metrics['mae_px']:.1f}px  "
                f"pck5={metrics['pck5']:.3f}  l1={metrics['l1']:.4f}",
                flush=True,
            )
        print(f"  train R2 by level  {_fmt_level_r2(train_m)}", flush=True)
        print(f"  val   R2 by level  {_fmt_level_r2(val_m)}", flush=True)
        print(f"  test  R2 by level  {_fmt_level_r2(test_m)}", flush=True)
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
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    def _eval(samples: list[LandmarkSample]) -> dict:
        if not samples:
            return {"r2": float("nan"), "r2_x": float("nan"), "r2_y": float("nan"), "n": 0}
        return _run_epoch(model, _make_loader(samples, image_size, batch_size, False), train=False, device=device)

    splits = {
        "train": _eval(parts["train"]),
        "val": _eval(parts["val"] or parts["train"]),
        "test": _eval(parts["holdout"]),
    }
    print("best checkpoint  train / val / test R2", flush=True)
    for name in ("train", "val", "test"):
        m = splits[name]
        print(f"  {name:<5}  {_fmt_r2(m)}  mae={m.get('mae_px', float('nan')):.1f}px  pck5={m.get('pck5', float('nan')):.3f}", flush=True)
        print(f"         R2 by level  {_fmt_level_r2(m)}", flush=True)
    meta = {
        "condition": condition,
        "backbone": backbone,
        "image_size": image_size,
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_holdout": len(parts["holdout"]),
        "checkpoint": ckpt.as_posix(),
        "history": history,
        "splits": splits,
        "lr": lr,
        "weight_decay": float(weight_decay),
        "split_seed": seed,
        "device": str(device),
        "all_slices": bool(all_slices),
        "dense_csv": str(dense_csv) if dense_csv else None,
    }
    (output_dir / f"{condition}_{backbone}_history.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
