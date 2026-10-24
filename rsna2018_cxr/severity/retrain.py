#!/usr/bin/env python
"""
Train a binary severity ViT using a fixed train/val split from val_test_images.csv.

Use this to reproduce or retrain a specific NAS trial configuration while keeping
validation images exactly as listed in the CSV (e.g. trial_005 severity-1 NAS run).

Saves:
  - best_model_val{val_acc}_train{train_acc}.pt  (selected by --best-metric)
  - confusion_matrix_best_val.csv / .png at the epoch when best val accuracy is reached
  - training_history.csv, dataset_split.json, run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from rsna2018_cxr.severity import train_vit as tv
from rsna2018_cxr.severity import nas as vnas

NORMAL_CLASS = "normal"

SEVERITY_POSITIVE_CLASSES = {
    "severity_1": "severity_1_low_mild",
    "severity_2": "severity_2_moderate",
    "severity_3": "severity_3_severe",
}


def resolve_image_path(image_path: str, data_root: Path) -> Path:
    rel = image_path.replace("/workspace/", "").replace("\\", "/").lstrip("/")
    candidate = (data_root / rel).resolve()
    if candidate.is_file():
        return candidate
    stem = candidate.with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        alt = stem.with_suffix(ext)
        if alt.is_file():
            return alt.resolve()
    raise FileNotFoundError(f"Could not resolve image: {image_path}")


def build_binary_samples(
    data_root: Path,
    positive_class: str,
) -> Tuple[tv.SamplesListDataset, Dict[str, int], str]:
    class_to_idx = {NORMAL_CLASS: 0, positive_class: 1}

    if tv.is_severity_split_layout(data_root):
        samples_4c, class_to_idx_4c = tv.collect_severity_split_samples(data_root)
        idx_to_class_4c = {idx: cls for cls, idx in class_to_idx_4c.items()}
        binary_samples: List[Tuple[str, int]] = []
        for path, old_label in samples_4c:
            name = idx_to_class_4c[old_label]
            if name == NORMAL_CLASS:
                binary_samples.append((path, class_to_idx[NORMAL_CLASS]))
            elif name == positive_class:
                binary_samples.append((path, class_to_idx[positive_class]))
        layout = f"rsna_normal_vs_{positive_class}"
    else:
        raw = tv.ImageFolder(data_root)
        idx_to_class = {idx: cls for cls, idx in raw.class_to_idx.items()}
        binary_samples = []
        for path, old_label in raw.samples:
            name = idx_to_class[old_label]
            if name == NORMAL_CLASS:
                binary_samples.append((path, class_to_idx[NORMAL_CLASS]))
            elif name == positive_class:
                binary_samples.append((path, class_to_idx[positive_class]))
        layout = f"imagefolder_normal_vs_{positive_class}"

    if not binary_samples:
        raise FileNotFoundError(
            f"No images for classes {NORMAL_CLASS!r} and {positive_class!r} under {data_root}"
        )
    return tv.SamplesListDataset(binary_samples, class_to_idx), class_to_idx, layout


def build_dataloaders_from_split_csv(
    data_root: Path,
    split_csv: Path,
    positive_class: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    weights,
    balanced_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, Any]]:
    """Train/val/test loaders; val and test share indices from CSV val+test rows."""
    base_dataset, class_to_idx, layout = build_binary_samples(data_root, positive_class)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    df = pd.read_csv(split_csv)
    required = {"split", "image_path", "class_name"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(f"Split CSV missing columns: {missing_cols}")

    eval_stems: set[str] = set()
    for _, row in df.iterrows():
        if str(row["split"]).strip().lower() in {"val", "test"}:
            eval_stems.add(Path(str(row["image_path"])).stem)

    if not eval_stems:
        raise ValueError(f"No val/test rows in {split_csv}")

    train_indices: List[int] = []
    val_indices: List[int] = []
    unmatched_eval = set(eval_stems)

    for idx, (img_path, label) in enumerate(base_dataset.samples):
        stem = Path(img_path).stem
        if stem in eval_stems:
            val_indices.append(idx)
            unmatched_eval.discard(stem)
        else:
            train_indices.append(idx)

    if unmatched_eval:
        print(
            f"Warning: {len(unmatched_eval)} val/test stems in CSV not found in binary dataset "
            f"(e.g. wrong severity filter)."
        )

    train_tf, eval_tf = tv.build_transforms(image_size, weights)
    train_dataset = tv.SplitImageFolder(base_dataset, train_indices, train_tf)
    val_dataset = tv.SplitImageFolder(base_dataset, val_indices, eval_tf)
    test_dataset = tv.SplitImageFolder(base_dataset, val_indices, eval_tf)

    train_sampler = None
    if balanced_sampler and len(train_dataset.samples) > 0:
        class_counts = {i: 0 for i in class_to_idx.values()}
        for _, lab in train_dataset.samples:
            class_counts[lab] += 1
        sample_weights = [
            (1.0 / float(class_counts[lab])) if class_counts[lab] > 0 else 0.0
            for _, lab in train_dataset.samples
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    pin_memory = torch.cuda.is_available()
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
                pin_memory=pin_memory,
            )
        )

    def _count(indices: List[int]) -> Dict[str, int]:
        out = {idx_to_class[i]: 0 for i in idx_to_class}
        for j in indices:
            _, lab = base_dataset.samples[j]
            out[idx_to_class[lab]] += 1
        return out

    all_indices = list(range(len(base_dataset.samples)))
    split_summary: Dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "split_csv": str(split_csv.resolve()),
        "positive_class": positive_class,
        "dataset_layout": layout,
        "num_classes": len(class_to_idx),
        "class_to_idx": dict(class_to_idx),
        "total_files_loaded": len(base_dataset.samples),
        "files_per_class_before_split": dict(sorted(_count(all_indices).items())),
        "files_per_class_train": dict(sorted(_count(train_indices).items())),
        "files_per_class_val": dict(sorted(_count(val_indices).items())),
        "files_per_class_test": dict(sorted(_count(val_indices).items())),
        "val_fraction": None,
        "train_fraction_target": None,
        "train_split_size": len(train_indices),
        "val_split_size": len(val_indices),
        "test_split_size": len(val_indices),
        "val_and_test_identical": True,
        "split_rule": f"val/test membership from CSV; binary classes: {NORMAL_CLASS} vs {positive_class}",
    }
    return (*loaders, class_to_idx, split_summary)


def load_training_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def train(
    config: Dict[str, Any],
    data_root: Path,
    split_csv: Path,
    output_dir: Path,
    positive_class: str,
    best_metric: str,
    balanced_sampler: bool,
    show_progress: bool,
    log_interval: int,
    class_weight_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed = config.get("seed", 1234)
    tv.set_seed(int(set_seed))

    train_loader, val_loader, test_loader, class_to_idx, split_summary = (
        build_dataloaders_from_split_csv(
            data_root=data_root,
            split_csv=split_csv,
            positive_class=positive_class,
            image_size=int(config["image_size"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(config.get("num_workers", 4)),
            weights=None,
            balanced_sampler=balanced_sampler,
        )
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    tv.log_dataset_split(split_summary, output_dir / "dataset_split.json")
    shutil.copy2(split_csv, output_dir / "val_test_images.csv")

    model, weights = tv.build_model(
        config["variant"],
        num_classes=num_classes,
        freeze_backbone=config["freeze_backbone"],
        use_pretrained=config["use_pretrained"],
        head_depth=config["head_depth"],
        head_hidden_dim=config.get("head_hidden_dim"),
        head_dropout=config["head_dropout"],
        head_activation=config["head_activation"],
    )
    model = vnas.adapt_torchvision_vit_to_image_size(model, int(config["image_size"]))
    train_tf, eval_tf = tv.build_transforms(int(config["image_size"]), weights)
    train_loader.dataset.transform = train_tf
    val_loader.dataset.transform = eval_tf
    test_loader.dataset.transform = eval_tf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Split CSV: {split_csv}")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = vnas.create_optimizer(
        params,
        config["optimizer_type"],
        float(config["learning_rate"]),
        float(config["weight_decay"]),
    )
    scheduler = vnas.create_scheduler(
        optimizer, config["scheduler_type"], int(config["epochs"])
    )

    class_weight_tensor = None
    if class_weight_mode != "none":
        train_counts = split_summary["files_per_class_train"]
        counts = np.array(
            [float(train_counts.get(idx_to_class[i], 0)) for i in range(num_classes)],
            dtype=np.float64,
        )
        if np.any(counts <= 0):
            raise RuntimeError(f"Invalid train class counts: {counts.tolist()}")
        if class_weight_mode == "inverse":
            w = 1.0 / counts
        else:
            w = 1.0 / np.sqrt(counts)
        w = w * (len(w) / float(w.sum()))
        class_weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Class weights ({class_weight_mode}): {w.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    scaler = tv.create_grad_scaler(bool(config.get("amp", True)), device)

    epochs = int(config["epochs"])
    patience = config.get("patience")
    if patience is None:
        patience = epochs

    if best_metric == "val_loss":
        best_metric_value: Any = float("inf")
    elif best_metric == "val_f1_macro":
        best_metric_value = float("-inf")
    elif best_metric == "val_acc_then_train_acc":
        best_metric_value = (float("-inf"), float("-inf"))
    else:
        raise ValueError(f"Unknown best_metric: {best_metric}")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_model_file: Path | None = None
    best_val_preds = np.array([], dtype=np.int64)
    best_val_targets = np.array([], dtype=np.int64)

    history_csv = output_dir / "training_history.csv"
    history_initialized = False

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _ = tv.train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            show_batch_progress=show_progress,
            batch_progress_desc=f"Epoch {epoch}/{epochs} train",
            log_interval=log_interval,
        )
        val_loss, val_acc, val_preds, val_targets, _ = tv.evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_batch_progress=show_progress,
            batch_progress_desc=f"Epoch {epoch}/{epochs} val",
            log_interval=log_interval,
        )
        test_loss, test_acc, _, _, _ = tv.evaluate(
            model,
            test_loader,
            criterion,
            device,
            show_batch_progress=show_progress,
            batch_progress_desc=f"Epoch {epoch}/{epochs} test",
            log_interval=log_interval,
        )

        val_metrics = tv.compute_classification_metrics(val_targets, val_preds, num_classes)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr = (
            scheduler.get_last_lr()[0]
            if not isinstance(scheduler, ReduceLROnPlateau)
            else optimizer.param_groups[0]["lr"]
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
            "epoch_time_sec": time.time() - epoch_start,
        }
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        mode = "a" if history_initialized else "w"
        with history_csv.open(mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not history_initialized:
                writer.writeheader()
                history_initialized = True
            writer.writerow(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"| lr={lr:.2e} | {row['epoch_time_sec']:.1f}s",
            flush=True,
        )

        best_val_loss = min(best_val_loss, val_loss)

        if best_metric == "val_loss":
            selection_value = val_loss
            is_better = selection_value < best_metric_value
        elif best_metric == "val_f1_macro":
            selection_value = float(val_metrics["f1_macro"])
            is_better = selection_value > best_metric_value
        else:
            selection_value = (float(val_acc), float(train_acc))
            is_better = selection_value > best_metric_value

        if is_better:
            best_metric_value = selection_value
            best_epoch = epoch
            epochs_no_improve = 0
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets.copy()

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
                "class_to_idx": class_to_idx,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "positive_class": positive_class,
                "split_csv": str(split_csv.resolve()),
            }
            if best_model_file is not None and best_model_file.exists():
                best_model_file.unlink()
            acc_tag = f"val{val_acc:.4f}_train{train_acc:.4f}"
            best_model_file = output_dir / f"best_model_{acc_tag}.pt"
            torch.save(state, best_model_file)
            print(f"  -> Saved best model: {best_model_file.name}", flush=True)
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= int(patience):
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    if best_model_file is None:
        raise RuntimeError("No best model was saved; training may have failed.")

    class_names = [idx_to_class[i] for i in range(num_classes)]
    tv.save_confusion_matrix_artifacts(
        best_val_targets,
        best_val_preds,
        class_names,
        output_dir / "confusion_matrix_best_val.csv",
        output_dir / "confusion_matrix_best_val.png",
        f"Validation confusion matrix (best val acc, epoch {best_epoch})",
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {best_model_file}")
    print(f"Saved confusion_matrix_best_val.csv/png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trial-dir",
        type=Path,
        default=None,
        help="NAS trial folder containing config.json and val_test_images.csv.",
    )
    p.add_argument("--config-json", type=Path, default=None, help="Override training config JSON.")
    p.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="val_test_images.csv (default: <trial-dir>/val_test_images.csv).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="severity_split_jpg_dataset root (normal/ + pneumonia/).",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--severity",
        choices=list(SEVERITY_POSITIVE_CLASSES.keys()),
        default=None,
        help="Shortcut: severity_1 | severity_2 | severity_3 (sets --positive-class).",
    )
    p.add_argument(
        "--positive-class",
        type=str,
        default=None,
        help="Pneumonia positive class folder name, e.g. severity_1_low_mild.",
    )
    p.add_argument(
        "--best-metric",
        choices=["val_loss", "val_f1_macro", "val_acc_then_train_acc"],
        default="val_acc_then_train_acc",
    )
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument(
        "--class-weight-mode",
        choices=["none", "inverse", "sqrt_inverse"],
        default="none",
    )
    p.add_argument("--progress", action="store_true")
    p.add_argument("--log-interval", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.trial_dir is None and args.config_json is None:
        raise SystemExit("Provide --trial-dir and/or --config-json.")

    trial_dir = args.trial_dir.resolve() if args.trial_dir else None
    config_path = args.config_json or (trial_dir / "config.json" if trial_dir else None)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError("config.json not found; pass --config-json or --trial-dir.")

    split_csv = args.split_csv or (trial_dir / "val_test_images.csv" if trial_dir else None)
    if split_csv is None or not split_csv.is_file():
        raise FileNotFoundError("Split CSV not found; pass --split-csv or --trial-dir.")

    positive_class = args.positive_class
    if positive_class is None and args.severity is not None:
        positive_class = SEVERITY_POSITIVE_CLASSES[args.severity]
    if positive_class is None and trial_dir is not None:
        # Infer from trial folder name
        name = trial_dir.name.lower()
        if "severity1" in name or "sev1" in name:
            positive_class = SEVERITY_POSITIVE_CLASSES["severity_1"]
        elif "severity2" in name or "sev2" in name:
            positive_class = SEVERITY_POSITIVE_CLASSES["severity_2"]
        elif "severity3" in name or "sev3" in name:
            positive_class = SEVERITY_POSITIVE_CLASSES["severity_3"]
    if positive_class is None:
        raise SystemExit("Set --positive-class or --severity (or use a trial-dir path containing severity1/2/3).")

    config = load_training_config(config_path)
    data_root = args.data_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (trial_dir if trial_dir else Path("runs") / f"train_{positive_class}")
    )

    run_cfg = {
        "trial_dir": str(trial_dir) if trial_dir else None,
        "config_json": str(config_path),
        "split_csv": str(split_csv),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "positive_class": positive_class,
        "best_metric": args.best_metric,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"Training: {NORMAL_CLASS} vs {positive_class}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    train(
        config=config,
        data_root=data_root,
        split_csv=split_csv,
        output_dir=output_dir,
        positive_class=positive_class,
        best_metric=args.best_metric,
        balanced_sampler=args.balanced_sampler,
        show_progress=args.progress,
        log_interval=args.log_interval,
        class_weight_mode=args.class_weight_mode,
    )


if __name__ == "__main__":
    main()
