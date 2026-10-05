"""Run refit training from NAS best-config entries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from rsna2024_lumbar.ordinal.config import hyperparameters_dict, refit_slug
from rsna2024_lumbar.ordinal.paths import (
    ordinal_output_dir,
    refit_env,
    resolve_crops_root,
    resolve_data_root,
)
from rsna2024_lumbar.preprocessing.constants import CROP_POLICY_CENTERED


def build_train_command(
    entry: dict[str, Any],
    *,
    data_root: Path,
    crops_root: Path | None,
    epochs: int,
    output_dir: Path,
    save_checkpoints: bool = True,
    best_metric: str = "val_acc",
    max_studies: int | None = None,
    progress: bool = True,
    split_seed: int | None = 42,
) -> list[str]:
    hp = hyperparameters_dict(entry)
    condition = entry["condition"]
    model_type = hp.get("model_type") or entry.get("family")
    if not model_type:
        raise ValueError(f"entry missing model_type: {entry}")

    crop_policy = hp.get("crop_policy") or CROP_POLICY_CENTERED
    image_source = hp.get("image_source") or "png"
    if crops_root is None and image_source == "png":
        crops_root = resolve_crops_root(str(crop_policy), None)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.ordinal.train_entry",
        "--data-root",
        str(data_root),
        "--condition",
        condition,
        "--model-type",
        str(model_type),
        "--crop-policy",
        str(crop_policy),
        "--image-source",
        str(image_source),
        "--batch-size",
        str(int(hp.get("batch_size", 8))),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(float(hp.get("learning_rate", 3e-4))),
        "--weight-decay",
        str(float(hp.get("weight_decay", 0.05))),
        "--head-depth",
        str(int(hp.get("head_depth", 1))),
        "--head-dropout",
        str(float(hp.get("head_dropout", 0.0))),
        "--head-activation",
        str(hp.get("head_activation", "gelu")),
        "--output-dir",
        str(output_dir),
        "--best-metric",
        best_metric,
        *([] if not progress else ["--progress"]),
    ]

    variant = hp.get("variant") or hp.get("backbone_variant")
    if variant:
        cmd.extend(["--backbone-variant", str(variant)])

    input_layout = hp.get("input_layout")
    if input_layout and input_layout != "2d":
        cmd.extend(["--input-layout", str(input_layout)])

    if hp.get("image_width") is not None:
        cmd.extend(["--image-width", str(int(hp["image_width"]))])
    if hp.get("image_height") is not None:
        cmd.extend(["--image-height", str(int(hp["image_height"]))])

    if hp.get("head_hidden_dim") is not None:
        cmd.extend(["--head-hidden-dim", str(int(hp["head_hidden_dim"]))])

    if hp.get("freeze_backbone"):
        cmd.append("--freeze-backbone")
    if hp.get("use_pretrained") is False:
        cmd.append("--no-pretrained")
    if hp.get("amp"):
        cmd.append("--amp")

    if image_source == "png" and crops_root is not None:
        cmd.extend(["--crops-root", str(crops_root)])

    if save_checkpoints:
        cmd.append("--save-checkpoints")
    if max_studies is not None:
        cmd.extend(["--max-studies", str(max_studies)])
    if split_seed is not None:
        cmd.extend(["--split-seed", str(split_seed)])

    return cmd


def write_run_manifest(
    output_dir: Path,
    entry: dict[str, Any],
    *,
    command: list[str],
    source_config: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_best_config": str(source_config.resolve()),
        "nas_trial_id": entry.get("trial_id"),
        "condition": entry.get("condition"),
        "family": entry.get("family"),
        "hyperparameters": hyperparameters_dict(entry),
        "nas_metrics": entry.get("metrics"),
        "train_command": command,
    }
    with (output_dir / "refit_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def extract_best_epoch_summary(run_dir: Path) -> dict[str, Any] | None:
    csv_path = run_dir / "training_metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if "val_acc" in df.columns:
        idx = df["val_acc"].idxmax()
        row = df.loc[idx]
    else:
        row = df.iloc[-1]
    summary: dict[str, Any] = {"best_epoch": int(row.get("epoch", -1))}
    for col in df.columns:
        if col == "epoch":
            continue
        if any(
            col.startswith(p)
            for p in ("train_", "val_", "test_")
        ) and any(
            m in col
            for m in ("oa_", "omae_", "qwk_", "ser_", "_oa", "_omae", "_qwk", "_ser")
        ):
            summary[col] = row[col]
        elif col.endswith(("_oa_overall", "_omae_overall", "_qwk_overall", "_ser_overall")):
            summary[col] = row[col]
    return summary


def run_entry_refit(
    entry: dict[str, Any],
    *,
    source_config: Path,
    data_root: Path | None = None,
    crops_root: Path | None = None,
    epochs: int = 50,
    output_base: Path | None = None,
    max_studies: int | None = None,
    dry_run: bool = False,
) -> int:
    slug = refit_slug(entry)
    out = ordinal_output_dir(slug, output_base=output_base)
    cmd = build_train_command(
        entry,
        data_root=resolve_data_root(data_root),
        crops_root=crops_root,
        epochs=epochs,
        output_dir=out,
        max_studies=max_studies,
    )
    write_run_manifest(out, entry, command=cmd, source_config=source_config)
    print(f"Refit {slug} -> {out}")
    print(" ".join(cmd))
    if dry_run:
        return 0
    env = refit_env()
    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


def write_refit_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
