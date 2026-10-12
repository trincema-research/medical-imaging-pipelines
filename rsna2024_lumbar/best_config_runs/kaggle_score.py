"""Score NAS best-config retrains with the official Kaggle metric."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from rsna2024_lumbar.best_config_runs.condition_specific import (
    ARCHITECTURE_LABELS,
    CONDITION_ABBREV,
)
from rsna2024_lumbar.best_config_runs.kaggle_metric import (
    ANY_SEVERE_SCALAR_OFFICIAL,
    ParticipantVisibleError,
    score_prediction_table,
)
from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

PREDICTION_SPLITS: tuple[tuple[str, str], ...] = (
    ("val", "validation"),
    ("test", "test"),
)
FAMILY_PREFIX = "nas_best_"


def default_score_root() -> Path:
    retrain = RESULTS_DIR / "retrain_pr"
    return retrain if retrain.is_dir() else RESULTS_DIR


def representation_from_stem(stem: str) -> str:
    if stem.endswith("_3d") or "3d" in stem.split("_"):
        return "2.5D-to-3D"
    return "2D"


def architecture_from_stem(stem: str) -> str:
    token = stem.replace("nas_best_", "")
    token = token.removesuffix("_2d").removesuffix("_3d")
    return ARCHITECTURE_LABELS.get(token, token)


def iter_family_dirs(output_base: Path) -> list[Path]:
    base = Path(output_base)
    if base.name.startswith(FAMILY_PREFIX):
        return [base]
    found = sorted(p for p in base.glob(f"{FAMILY_PREFIX}*") if p.is_dir())
    return found


def iter_repeat_dirs(family_dir: Path) -> list[Path]:
    repeats = sorted(family_dir.glob("*/repeat_*"))
    return [p for p in repeats if p.is_dir()]


def prediction_path(repeat_dir: Path, split: str) -> Path:
    return repeat_dir / f"predictions_{split}.csv"


def load_repeat_predictions(repeat_dir: Path, split: str) -> pd.DataFrame:
    path = prediction_path(repeat_dir, split)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    if "condition" not in df.columns or df["condition"].isna().all():
        df["condition"] = repeat_dir.parent.name
    return df


def argv_from_run_config(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    checkpoint: Path,
) -> list[str]:
    argv = [
        "--data-root",
        str(cfg["data_root"]),
        "--condition",
        str(cfg["condition"]),
        "--model-type",
        str(cfg["model_type"]),
        "--crop-policy",
        str(cfg.get("crop_policy") or "centered"),
        "--image-source",
        str(cfg.get("image_source") or "png"),
        "--batch-size",
        str(int(cfg.get("batch_size") or 8)),
        "--output-dir",
        str(output_dir),
        "--eval-checkpoint",
        str(checkpoint),
        "--num-workers",
        "0",
        "--seed",
        str(int(cfg.get("seed") or 42)),
        "--log-pipeline-metrics",
    ]
    if cfg.get("crops_root"):
        argv.extend(["--crops-root", str(cfg["crops_root"])])
    if cfg.get("image_width") not in (None, ""):
        argv.extend(["--image-width", str(int(cfg["image_width"]))])
    if cfg.get("image_height") not in (None, ""):
        argv.extend(["--image-height", str(int(cfg["image_height"]))])
    if cfg.get("input_layout") and cfg["input_layout"] != "2d":
        argv.extend(["--input-layout", str(cfg["input_layout"])])
    if cfg.get("backbone_variant"):
        argv.extend(["--backbone-variant", str(cfg["backbone_variant"])])
    if cfg.get("head_depth") not in (None, ""):
        argv.extend(["--head-depth", str(int(cfg["head_depth"]))])
    if cfg.get("head_hidden_dim") not in (None, ""):
        argv.extend(["--head-hidden-dim", str(int(cfg["head_hidden_dim"]))])
    if cfg.get("head_dropout") not in (None, ""):
        argv.extend(["--head-dropout", str(float(cfg["head_dropout"]))])
    if cfg.get("head_activation"):
        argv.extend(["--head-activation", str(cfg["head_activation"])])
    if cfg.get("volume_depth") not in (None, ""):
        argv.extend(["--volume-depth", str(int(cfg["volume_depth"]))])
    if cfg.get("split_seed") not in (None, ""):
        argv.extend(["--split-seed", str(int(cfg["split_seed"]))])
    if cfg.get("train_fraction") not in (None, ""):
        argv.extend(["--train-fraction", str(float(cfg["train_fraction"]))])
    if cfg.get("val_fraction") not in (None, ""):
        argv.extend(["--val-fraction", str(float(cfg["val_fraction"]))])
    if cfg.get("test_fraction") not in (None, ""):
        argv.extend(["--test-fraction", str(float(cfg["test_fraction"]))])
    if cfg.get("no_pretrained"):
        argv.append("--no-pretrained")
    if cfg.get("amp"):
        argv.append("--amp")
    if cfg.get("freeze_backbone"):
        argv.append("--freeze-backbone")
    return argv


def _import_trainer():
    from rsna2024_lumbar.best_config_runs.paths import pipeline_env
    from rsna2024_lumbar.nas.paths import BUNDLE_DIR, REPO_ROOT

    env = pipeline_env()
    os.environ.update(env)
    for path in (str(REPO_ROOT.resolve()), str(BUNDLE_DIR.resolve())):
        if path not in sys.path:
            sys.path.insert(0, path)
    import train_vit_lumbar as tv

    return tv


def eval_repeat_predictions(repeat_dir: Path) -> None:
    checkpoint = repeat_dir / "best_model.pt"
    config_path = repeat_dir / "run_config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    argv = argv_from_run_config(cfg, output_dir=repeat_dir, checkpoint=checkpoint)
    print(f"eval {repeat_dir.name} {repeat_dir.parent.name}", flush=True)
    _import_trainer().main(argv)


def ensure_family_predictions(family_dir: Path, *, eval_if_missing: bool) -> list[str]:
    warnings: list[str] = []
    for repeat_dir in iter_repeat_dirs(family_dir):
        missing = [
            split for split, _name in PREDICTION_SPLITS if not prediction_path(repeat_dir, split).is_file()
        ]
        if not missing:
            continue
        if not eval_if_missing:
            warnings.append(f"missing predictions in {repeat_dir}")
            continue
        if not (repeat_dir / "best_model.pt").is_file():
            warnings.append(f"no checkpoint in {repeat_dir}")
            continue
        eval_repeat_predictions(repeat_dir)
    return warnings


def _join_family_split(family_dir: Path, split: str, repeat_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        repeat_dir = family_dir / condition / repeat_name
        part = load_repeat_predictions(repeat_dir, split)
        if part.empty:
            continue
        part = part.copy()
        part["condition"] = condition
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def score_family(
    family_dir: Path,
    *,
    any_severe_scalar: float = ANY_SEVERE_SCALAR_OFFICIAL,
    repeats: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    stem = family_dir.name
    repeat_names = list(repeats) if repeats is not None else sorted(
        {p.name for p in iter_repeat_dirs(family_dir)}
    )
    out: list[dict[str, Any]] = []
    for repeat_name in repeat_names:
        for split, split_name in PREDICTION_SPLITS:
            pred = _join_family_split(family_dir, split, repeat_name)
            if pred.empty:
                continue
            try:
                parts = score_prediction_table(pred, any_severe_scalar=any_severe_scalar)
            except (ValueError, KeyError, ParticipantVisibleError) as exc:
                print(f"skip {stem} {repeat_name} {split_name}: {exc}")
                continue
            conditions = sorted(pred["condition"].astype(str).unique())
            out.append(
                {
                    "model_config": stem,
                    "architecture": architecture_from_stem(stem),
                    "representation": representation_from_stem(stem),
                    "repeat": repeat_name,
                    "split": split_name,
                    "n_conditions": len(conditions),
                    "conditions": ",".join(CONDITION_ABBREV.get(c, c) for c in conditions),
                    "complete_five": int(len(conditions) == len(CONDITIONS)),
                    **parts,
                }
            )
    return out


def write_kaggle_scores(
    output_base: Path,
    *,
    eval_if_missing: bool = True,
    any_severe_scalar: float = ANY_SEVERE_SCALAR_OFFICIAL,
    only: list[str] | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    wanted = {stem.replace(".json", "") for stem in only} if only else None
    for family_dir in iter_family_dirs(output_base):
        if wanted is not None and family_dir.name not in wanted:
            continue
        warnings.extend(ensure_family_predictions(family_dir, eval_if_missing=eval_if_missing))
        rows.extend(score_family(family_dir, any_severe_scalar=any_severe_scalar))
    if not rows:
        detail = "; ".join(warnings) if warnings else "no nas_best_* prediction tables"
        raise SystemExit(f"No Kaggle scores written: {detail}")
    out = Path(output_base) / "kaggle_metric_scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"Wrote {out} ({len(rows)} rows)")
    return out
