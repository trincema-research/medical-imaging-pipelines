"""Launch bundled training for one NAS best-config entry."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rsna2024_lumbar.perf_pipeline.config import hyperparameters_dict
from rsna2024_lumbar.perf_pipeline.paths import pipeline_env, resolve_crops_root, resolve_data_root
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
    seed: int | None = 42,
    early_stop_patience: int | None = None,
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
        "rsna2024_lumbar.perf_pipeline.train_entry",
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

    hidden = hp.get("head_hidden_dim")
    if hidden not in (None, ""):
        cmd.extend(["--head-hidden-dim", str(int(hidden))])

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
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if early_stop_patience is not None:
        cmd.extend(["--early-stop-patience", str(int(early_stop_patience))])

    return cmd


def write_run_manifest(
    output_dir: Path,
    entry: dict[str, Any],
    *,
    command: list[str],
    source_config: Path,
    repeat_index: int,
    split_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_best_config": str(source_config.resolve()),
        "repeat_index": repeat_index,
        "split_seed": split_seed,
        "nas_trial_id": entry.get("trial_id"),
        "condition": entry.get("condition"),
        "family": entry.get("family"),
        "hyperparameters": hyperparameters_dict(entry),
        "nas_metrics": entry.get("metrics"),
        "train_command": command,
    }
    with (output_dir / "pipeline_run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def run_training(
    entry: dict[str, Any],
    *,
    source_config: Path,
    output_dir: Path,
    data_root: Path | None = None,
    crops_root: Path | None = None,
    epochs: int = 50,
    max_studies: int | None = None,
    split_seed: int = 42,
    seed: int = 42,
    repeat_index: int = 1,
    dry_run: bool = False,
    early_stop_patience: int | None = None,
) -> int:
    cmd = build_train_command(
        entry,
        data_root=resolve_data_root(data_root),
        crops_root=crops_root,
        epochs=epochs,
        output_dir=output_dir,
        max_studies=max_studies,
        split_seed=split_seed,
        seed=seed,
        early_stop_patience=early_stop_patience,
    )
    write_run_manifest(
        output_dir,
        entry,
        command=cmd,
        source_config=source_config,
        repeat_index=repeat_index,
        split_seed=split_seed,
    )
    print(f"Pipeline run repeat_{repeat_index:02d} -> {output_dir}")
    if dry_run:
        print(" ".join(cmd))
        return 0
    completed = subprocess.run(cmd, env=pipeline_env(), check=False)
    return int(completed.returncode)
