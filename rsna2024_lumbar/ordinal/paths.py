"""Default paths for ordinal refit runs."""

from __future__ import annotations

import os
from pathlib import Path

from rsna2024_lumbar.nas.paths import (
    BUNDLE_DIR,
    REPO_ROOT,
    YEAR_ROOT,
    default_crops_root,
    default_data_root,
)

ORDINAL_RESULTS_DIR = YEAR_ROOT / "data" / "ordinal_benchmark"
DEFAULT_BEST_CONFIG_DIR = YEAR_ROOT / "data" / "nas_compact" / "best_configs"


def ordinal_output_dir(entry_slug: str, *, output_base: Path | None = None) -> Path:
    base = output_base or ORDINAL_RESULTS_DIR
    return base / Path(entry_slug)


def bundle_train_script() -> Path:
    path = BUNDLE_DIR / "train_vit_lumbar.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Pack NAS bundle or sync training_bundle from legacy repo."
        )
    return path


def refit_env() -> dict[str, str]:
    env = os.environ.copy()
    bundle = str(BUNDLE_DIR.resolve())
    repo = str(REPO_ROOT.resolve())
    prev = env.get("PYTHONPATH", "")
    parts = [repo, bundle]
    if prev:
        parts.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["RSNA2024_ORDINAL_METRICS"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    return env


def resolve_data_root(data_root: Path | None) -> Path:
    return (data_root or default_data_root()).resolve()


def resolve_crops_root(crop_policy: str, crops_root: Path | None) -> Path:
    return (crops_root or default_crops_root(crop_policy)).resolve()
