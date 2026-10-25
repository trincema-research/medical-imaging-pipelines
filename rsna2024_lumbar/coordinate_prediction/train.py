"""Train one condition's 2D (x, y) regressor."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rsna2024_lumbar.coordinate_prediction.constants import (
    BACKBONE_CONVNEXT_TINY,
    IMAGE_SIZE_DEFAULT,
    LEVEL_DISPLAY,
    LUMBAR_LEVELS,
)
from rsna2024_lumbar.coordinate_prediction.dataset import (
    LandmarkSample,
    list_condition_samples,
    list_dense_slice_samples,
    load_rgb_slice,
    split_samples,
)
from rsna2024_lumbar.coordinate_prediction.models import (
    LEVEL_METRIC_KEYS,
    build_regressor,
    coordinate_metrics,
    empty_coordinate_metrics,
    landmark_loss,
)
from rsna2024_lumbar.coordinate_prediction.resume import (
    atomic_write_text,
    flatten_epoch_metrics,
    last_checkpoint_path,
    last_logged_epoch,
    next_epoch,
    reset_epoch_csvs,
    resume_hparams,
    resume_hparams_match,
    training_metrics_path,
    write_epoch_history_csvs,
)


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("pip install -e '.[coordinate_prediction]'") from exc
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


def _fmt_level_mae(metrics: dict) -> str:
    parts: list[str] = []
    nested = metrics.get("levels") or {}
    for level in LUMBAR_LEVELS:
        part = nested.get(level) or {}
        value = part.get("mae_px", metrics.get(f"mae_px_{level}", float("nan")))
        parts.append(f"{LEVEL_DISPLAY[level]}={value:.1f}px")
    return "  ".join(parts)


def metrics_table_rows(condition: str, splits: dict[str, dict], extra: dict | None = None) -> list[dict]:
    """One row per (split, overall|level) so later we can pick weak condition×level cells."""
    rows: list[dict] = []
    extras = dict(extra or {})
    for split, metrics in splits.items():
        base = {
            "condition": condition,
            "split": split,
            "n_images": int(metrics.get("n") or 0),
            **extras,
        }
        overall = {key: metrics.get(key, float("nan")) for key in LEVEL_METRIC_KEYS}
        rows.append({**base, "level": "overall", "level_display": "overall", **overall})
        nested = metrics.get("levels") or {}
        for level in LUMBAR_LEVELS:
            part = nested.get(level) or {key: metrics.get(f"{key}_{level}", float("nan")) for key in LEVEL_METRIC_KEYS}
            rows.append(
                {
                    **base,
                    "level": level,
                    "level_display": LEVEL_DISPLAY[level],
                    **{key: part.get(key, float("nan")) for key in LEVEL_METRIC_KEYS},
                }
            )
    return rows


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def write_split_metrics(
    output_dir: Path,
    condition: str,
    backbone: str,
    splits: dict[str, dict],
    extra: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition": condition,
        "backbone": backbone,
        "splits": splits,
        **dict(extra or {}),
    }
    json_path = output_dir / f"{condition}_{backbone}_metrics.json"
    csv_path = output_dir / f"{condition}_{backbone}_metrics.csv"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    rows = metrics_table_rows(condition, splits, extra={"backbone": backbone, **dict(extra or {})})
    fields = [
        "condition",
        "backbone",
        "split",
        "level",
        "level_display",
        "n_images",
        *LEVEL_METRIC_KEYS,
    ]
    extra_keys = [key for key in (rows[0].keys() if rows else []) if key not in fields]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*fields, *extra_keys], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if isinstance(value, float) and value != value else value)
                    for key, value in row.items()
                }
            )
    return {"json": json_path, "csv": csv_path, "n_rows": len(rows)}


def _run_epoch(model, loader, *, train: bool, optimizer=None, device):
    torch, _DataLoader, _Dataset = _require_torch()
    model.train(train)
    totals = {"loss": 0.0, "bce": 0.0, "n": 0}
    pred_chunks: list = []
    target_chunks: list = []
    vis_chunks: list = []
    width = height = None
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
        n = int(images.size(0))
        totals["loss"] += float(loss.detach()) * n
        totals["bce"] += float(parts["bce"]) * n
        totals["n"] += n
        width = int(batch["width"][0])
        height = int(batch["height"][0])
        pred_chunks.append(pred_xy.detach().cpu())
        target_chunks.append(xy.detach().cpu())
        vis_chunks.append(visible.detach().cpu())
    if not pred_chunks or width is None or height is None:
        return empty_coordinate_metrics()
    out = coordinate_metrics(torch.cat(pred_chunks), torch.cat(target_chunks), torch.cat(vis_chunks), width, height)
    count = max(totals["n"], 1)
    out["n"] = totals["n"]
    out["loss"] = totals["loss"] / count
    out["bce"] = totals["bce"] / count
    return out


def _atomic_torch_save(obj: dict, path: Path, torch) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_last_checkpoint(path: Path, map_location) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    torch, _DataLoader, _Dataset = _require_torch()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        print(f"could not read {path} ({exc}); starting from epoch 1", flush=True)
        return None
    return blob if isinstance(blob, dict) else None


def save_last_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_val: float,
    history: list,
    hparams: dict,
) -> Path:
    torch, _DataLoader, _Dataset = _require_torch()
    dest = Path(path)
    _atomic_torch_save(
        {
            "epoch": int(epoch),
            "best_val": float(best_val),
            "history": history,
            "hparams": hparams,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        dest,
        torch,
    )
    return dest


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
    resume: bool = True,
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
    last_path = last_checkpoint_path(output_dir, condition, backbone)
    hparams = resume_hparams(
        condition=condition,
        backbone=backbone,
        image_size=image_size,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        pretrained=pretrained,
        all_slices=all_slices,
    )
    metrics_csv = training_metrics_path(output_dir)
    start_epoch = 1
    if resume:
        last_blob = load_last_checkpoint(last_path, device)
        logged = last_logged_epoch(metrics_csv)
        if last_blob and resume_hparams_match(last_blob.get("hparams"), hparams):
            model.load_state_dict(last_blob["state_dict"])
            optimizer.load_state_dict(last_blob["optimizer"])
            history = list(last_blob.get("history") or [])
            best_val = float(last_blob.get("best_val") or float("inf"))
            start_epoch = next_epoch(max(int(last_blob.get("epoch") or 0), logged))
            print(f"resume {condition} {backbone} from epoch {start_epoch}/{epochs}", flush=True)
        elif last_blob:
            print(f"resume checkpoint hparams mismatch; starting {condition} from epoch 1", flush=True)
            reset_epoch_csvs(output_dir)
        elif logged:
            print(
                f"found {metrics_csv.name} through epoch {logged} but no last.pt; "
                f"retraining {condition} from epoch 1",
                flush=True,
            )
            reset_epoch_csvs(output_dir)
    else:
        reset_epoch_csvs(output_dir)
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
    if start_epoch > int(epochs):
        print(f"already finished {epochs} epochs; writing final metrics", flush=True)
    for epoch in range(start_epoch, int(epochs) + 1):
        train_m = _run_epoch(model, train_loader, train=True, optimizer=optimizer, device=device)
        val_m = _run_epoch(model, val_loader, train=False, device=device)
        test_m = _run_epoch(model, test_loader, train=False, device=device) if test_loader else empty_coordinate_metrics()
        row = {"epoch": epoch, "train": train_m, "val": val_m, "test": test_m}
        history.append(row)
        write_epoch_history_csvs(
            output_dir,
            flatten_epoch_metrics(epoch, {"train": train_m, "val": val_m, "test": test_m}),
        )
        print(f"epoch {epoch}/{epochs}  {condition}  5 levels at once", flush=True)
        for name, metrics in (("train", train_m), ("val", val_m), ("test", test_m)):
            print(
                f"  {name:<5}  {_fmt_r2(metrics)}  mae={metrics['mae_px']:.1f}px  "
                f"pck5={metrics['pck5']:.3f}  pck10={metrics.get('pck10', float('nan')):.3f}  "
                f"l1={metrics['l1']:.4f}  n_vis={int(metrics.get('n_visible') or 0)}",
                flush=True,
            )
            print(f"         R2 by level   {_fmt_level_r2(metrics)}", flush=True)
            print(f"         MAE by level  {_fmt_level_mae(metrics)}", flush=True)
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
        save_last_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val=best_val,
            history=history,
            hparams=hparams,
        )
    if ckpt.is_file():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
    model.eval()

    def _eval(samples: list[LandmarkSample]) -> dict:
        if not samples:
            return empty_coordinate_metrics()
        return _run_epoch(model, _make_loader(samples, image_size, batch_size, False), train=False, device=device)

    splits = {
        "train": _eval(parts["train"]),
        "val": _eval(parts["val"] or parts["train"]),
        "test": _eval(parts["holdout"]),
    }
    print("best checkpoint  train / val / test  (condition + each lumbar level)", flush=True)
    for name in ("train", "val", "test"):
        m = splits[name]
        print(
            f"  {name:<5}  {_fmt_r2(m)}  mae={m.get('mae_px', float('nan')):.1f}px  "
            f"pck5={m.get('pck5', float('nan')):.3f}  pck10={m.get('pck10', float('nan')):.3f}",
            flush=True,
        )
        print(f"         R2 by level   {_fmt_level_r2(m)}", flush=True)
        print(f"         MAE by level  {_fmt_level_mae(m)}", flush=True)
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
        "levels": list(LUMBAR_LEVELS),
    }
    written = write_split_metrics(
        output_dir,
        condition,
        backbone,
        splits,
        extra={
            "image_size": image_size,
            "lr": lr,
            "weight_decay": float(weight_decay),
            "pretrained": bool(pretrained),
            "all_slices": bool(all_slices),
        },
    )
    meta["metrics_json"] = written["json"].as_posix()
    meta["metrics_csv"] = written["csv"].as_posix()
    atomic_write_text(
        output_dir / f"{condition}_{backbone}_history.json",
        json.dumps(_jsonable(meta), indent=2),
    )
    return meta
