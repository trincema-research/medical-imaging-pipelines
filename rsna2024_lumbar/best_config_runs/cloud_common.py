"""Shared helpers for best_config_runs cloud pack / unpack."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rsna2024_lumbar.nas.bundle import default_legacy_root, iter_bundle_files, stage_training_bundle
from rsna2024_lumbar.nas.paths import BUNDLE_DIR, REPO_ROOT, YEAR_ROOT
from rsna2024_lumbar.best_config_runs.config import list_best_config_files
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS

MANIFEST_NAME = "best_config_runs_deploy_manifest.json"
CLOUD_RUN_NAME = "BEST_CONFIG_RUNS_CLOUD_RUN.txt"

CODE_PATHS = (
    YEAR_ROOT / "best_config_runs",
    YEAR_ROOT / "preprocessing",
    YEAR_ROOT / "nas",
    YEAR_ROOT / "__init__.py",
    REPO_ROOT / "common",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
    YEAR_ROOT / "nas" / "requirements-nas-cloud.txt",
    YEAR_ROOT / "data" / "best_config_runs" / "README.md",
)

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "runs", "training_bundle"}


def repo_root_from_extracted(extract_dir: Path) -> Path:
    extract_dir = extract_dir.resolve()
    if (extract_dir / "pyproject.toml").is_file():
        return extract_dir
    for candidate in extract_dir.iterdir():
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise SystemExit(
        f"No pyproject.toml under {extract_dir}. Unzip the best_config_runs cloud archive into an empty folder."
    )


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if SKIP_DIR_NAMES & set(rel_parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return files


def stage_bundle_if_needed(*, legacy_root: Path | None, skip_bundle: bool) -> list[Path]:
    if skip_bundle:
        return iter_bundle_files(BUNDLE_DIR)
    legacy = legacy_root or default_legacy_root()
    if legacy is not None:
        stage_training_bundle(legacy)
    elif not (BUNDLE_DIR / "train_vit_lumbar.py").is_file():
        raise SystemExit(
            "Training bundle missing (train_vit_lumbar.py). Pass --legacy-root or run pack without --skip-bundle "
            "on a machine with the legacy lumbar repo."
        )
    bundle_files = iter_bundle_files(BUNDLE_DIR)
    if not bundle_files:
        raise SystemExit("Training bundle directory is empty.")
    return bundle_files


def collect_pack_files(
    *,
    include_crops: bool,
    crop_policy: str,
    bundle_files: list[Path],
    best_config_dir: Path,
) -> list[Path]:
    files: list[Path] = []
    for path in CODE_PATHS:
        if not path.exists():
            continue
        files.extend(_iter_files(path))

    raw = YEAR_ROOT / "data" / "raw"
    for name in LABEL_CSVS:
        csv = raw / name
        if csv.is_file():
            files.append(csv)

    if not best_config_dir.is_dir():
        raise SystemExit(f"Missing NAS best configs: {best_config_dir}")
    for cfg in list_best_config_files(best_config_dir):
        files.append(cfg)

    if include_crops:
        crops = YEAR_ROOT / "data" / "processed" / crop_policy
        if not crops.is_dir():
            raise SystemExit(f"No crop cache at {crops}. Export crops or pack with --no-include-crops.")
        files.extend(_iter_files(crops))

    files.extend(bundle_files)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def arcname(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        bundle = BUNDLE_DIR.resolve()
        try:
            return path.resolve().relative_to(bundle).as_posix()
        except ValueError:
            return path.name


def build_manifest(
    *,
    include_crops: bool,
    crop_policy: str,
    file_count: int,
    repeats: int,
    epochs: int,
    early_stop_patience: int | None,
    best_config_dir: Path,
) -> dict[str, Any]:
    configs = [p.name for p in list_best_config_files(best_config_dir)]
    return {
        "schema": "best_config_runs_cloud_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "include_crops": include_crops,
        "crop_policy": crop_policy,
        "file_count": file_count,
        "repeats_default": repeats,
        "epochs_default": epochs,
        "early_stop_patience_default": early_stop_patience,
        "best_config_files": configs,
        "expected_result_rows": len(configs) * 5 * repeats,
    }


def cloud_run_text(
    *,
    repeats: int,
    epochs: int,
    early_stop_patience: int | None,
    include_crops: bool,
) -> str:
    patience_flag = (
        f" --early-stop-patience {early_stop_patience}"
        if early_stop_patience is not None
        else ""
    )
    return "\n".join(
        [
            "# Best-config runs — single-GPU cloud retrain",
            "",
            "# 1) Unpack (if you only copied the zip):",
            "python -m rsna2024_lumbar.best_config_runs.unpack_cloud --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud",
            "",
            "# 2) Install + train (from extracted repo root, directory with pyproject.toml):",
            "python -m rsna2024_lumbar.best_config_runs.deploy_cloud \\",
            f"  --repo-root . --repeats {repeats} --epochs {epochs}{patience_flag} \\",
            "  --skip-completed",
            "",
            "# Or unpack + deploy in one step:",
            "python -m rsna2024_lumbar.best_config_runs.deploy_cloud \\",
            f"  --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud --repeats {repeats} --epochs {epochs}{patience_flag}",
            "",
            f"Crops in zip: {'yes' if include_crops else 'no — need rsna2024_lumbar/data/processed/centered/'}",
            "Outputs: rsna2024_lumbar/data/best_config_runs/results/pipeline_results_all_models.csv",
            "",
        ]
    )


def read_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / MANIFEST_NAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
