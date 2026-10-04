"""Default data and training-bundle paths for cloud NAS runs."""

from __future__ import annotations

import os
from pathlib import Path

YEAR_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = YEAR_ROOT.parent
BUNDLE_DIR = NAS_DIR / "training_bundle"
CONFIGS_DIR = NAS_DIR / "configs"


def repo_root() -> Path:
    return REPO_ROOT


def default_data_root() -> Path:
    override = os.environ.get("RSNA2024_DATA_ROOT")
    if override:
        return Path(override).resolve()
    return (YEAR_ROOT / "data" / "raw").resolve()


def default_crops_root(crop_policy: str) -> Path:
    override = os.environ.get("RSNA2024_CROPS_ROOT")
    if override:
        return Path(override).resolve()
    return (YEAR_ROOT / "data" / "processed" / crop_policy).resolve()


def bundle_ready() -> bool:
    return (BUNDLE_DIR / "vit_nas_lumbar.py").is_file()


def config_file_for_family(config_name: str) -> Path:
    path = CONFIGS_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAS config: {path}")
    return path.resolve()
