"""Paths for NAS best-config retrain runs (``best_config_runs``)."""

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

RESULTS_DIR = YEAR_ROOT / "data" / "best_config_runs" / "results"
DEFAULT_BEST_CONFIG_DIR = YEAR_ROOT / "data" / "nas_compact" / "best_configs"
METRICS_ENV_VAR = "RSNA2024_BEST_CONFIG_RUNS_METRICS"
LEGACY_PERF_PIPELINE_ENV_VAR = "RSNA2024_PERF_PIPELINE_METRICS"
LEGACY_ORDINAL_ENV_VAR = "RSNA2024_ORDINAL_METRICS"
DEFAULT_REPEATS = 5
DEFAULT_SPLIT_SEEDS = (42, 142, 242, 342, 442)


def model_results_dir(config_stem: str, *, output_base: Path | None = None) -> Path:
    return (output_base or RESULTS_DIR) / config_stem


def run_output_dir(
    config_stem: str,
    condition: str,
    repeat_index: int,
    *,
    output_base: Path | None = None,
) -> Path:
    return model_results_dir(config_stem, output_base=output_base) / condition / f"repeat_{repeat_index:02d}"


def pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    bundle = str(BUNDLE_DIR.resolve())
    repo = str(REPO_ROOT.resolve())
    prev = env.get("PYTHONPATH", "")
    parts = [repo, bundle]
    if prev:
        parts.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env[METRICS_ENV_VAR] = "1"
    env[LEGACY_PERF_PIPELINE_ENV_VAR] = "1"
    env[LEGACY_ORDINAL_ENV_VAR] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    return env


def resolve_data_root(data_root: Path | None) -> Path:
    return (data_root or default_data_root()).resolve()


def resolve_crops_root(crop_policy: str, crops_root: Path | None) -> Path:
    return (crops_root or default_crops_root(crop_policy)).resolve()
