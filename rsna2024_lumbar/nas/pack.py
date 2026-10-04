"""Build a cloud-deploy zip for lumbar NAS (code + training bundle + labels + optional PNG crops)."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from rsna2024_lumbar.nas.bundle import (
    default_legacy_root,
    iter_bundle_files,
    stage_training_bundle,
)
from rsna2024_lumbar.nas.families import FAMILIES, FAMILY_SPECS
from rsna2024_lumbar.nas.paths import BUNDLE_DIR, REPO_ROOT, YEAR_ROOT
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS

PROFILES: dict[str, dict[str, object]] = {
    "all": {
        "families": list(FAMILIES),
        "default_output": "lumbar_nas_all_cloud.zip",
        "include_crops_default": False,
    },
    "efficientnet_families": {
        "families": ["efficientnet", "efficientnet3d"],
        "default_output": "lumbar_efficientnet_nas_2d3d_cloud.zip",
        "include_crops_default": False,
    },
    "efficientnet_deploy": {
        "families": ["efficientnet"],
        "default_output": "lumbar_efficientnet_nas_deploy.zip",
        "include_crops_default": True,
    },
    "convnext_families": {
        "families": ["convnext", "convnext3d"],
        "default_output": "lumbar_convnext_nas_2d3d_cloud.zip",
        "include_crops_default": False,
    },
    "convnext_deploy": {
        "families": ["convnext"],
        "default_output": "lumbar_convnext_nas_deploy.zip",
        "include_crops_default": True,
    },
    "transformers": {
        "families": ["vit", "maxvit"],
        "default_output": "lumbar_nas_transformers_cloud.zip",
        "include_crops_default": False,
    },
}

CODE_PATHS = (
    YEAR_ROOT / "nas",
    YEAR_ROOT / "preprocessing",
    YEAR_ROOT / "__init__.py",
    REPO_ROOT / "common",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
)

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "runs", "training_bundle"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zip lumbar NAS code for cloud deploy.")
    p.add_argument("--profile", choices=sorted(PROFILES), default="efficientnet_deploy")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--include-crops",
        action="store_true",
        default=None,
        help="Pack data/processed/<crop-policy>/ (large). Profile default may enable this.",
    )
    p.add_argument(
        "--no-include-crops",
        action="store_true",
        help="Omit PNG crop cache even when the profile defaults to including crops.",
    )
    p.add_argument("--crop-policy", default="centered")
    p.add_argument(
        "--legacy-root",
        type=Path,
        default=None,
        help="Path to rsna-2024-lumbar-spine-degenerative-classification (vit_nas_lumbar.py). "
        "Default: LUMBAR_LEGACY_ROOT or common local paths.",
    )
    p.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Do not copy training scripts (cloud run will fail unless bundle already staged).",
    )
    return p.parse_args(argv)


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


def _resolve_include_crops(args: argparse.Namespace, profile: dict[str, object]) -> bool:
    if args.no_include_crops:
        return False
    if args.include_crops:
        return True
    return bool(profile.get("include_crops_default", False))


def collect_pack_files(
    *,
    include_crops: bool,
    crop_policy: str,
    bundle_files: list[Path],
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
    readme = YEAR_ROOT / "nas" / "README.md"
    if readme.is_file():
        files.append(readme)
    if include_crops:
        crops = YEAR_ROOT / "data" / "processed" / crop_policy
        if not crops.is_dir():
            raise SystemExit(f"No crop cache at {crops}. Export first or omit --include-crops.")
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


def cloud_run_text(families: list[str], *, include_crops: bool) -> str:
    family = families[0] if len(families) == 1 else families[0]
    extra = " --all-conditions" if len(families) == 1 else ""
    lines = [
        "# Unzip anywhere, then from the repo root (directory with pyproject.toml):",
        "",
        "pip install -e .",
        "pip install -r rsna2024_lumbar/nas/requirements-nas-cloud.txt",
        "",
        "# One command (EfficientNet 2D, 8 GPUs, all 5 conditions, centered PNG crops):",
        f"python -m rsna2024_lumbar.nas.deploy --family {family} --num-gpus 8 "
        f"--crop-policy centered --epochs 50 --early-stop-patience 5{extra}",
        "",
        "# Preflight only:",
        f"python -m rsna2024_lumbar.nas.deploy --family {family} --dry-run",
        "",
        "# Lower-level launcher (single condition):",
        f"python -m rsna2024_lumbar.nas.launch --family {family} --num-gpus 8 "
        "--condition spinal_canal_stenosis --spawn --wait --amp --progress "
        "--epochs 50 --early-stop-patience 5",
        "",
        f"Crops in zip: {'yes' if include_crops else 'no — export to rsna2024_lumbar/data/processed/centered/'}",
        "Labels: rsna2024_lumbar/data/raw/*.csv",
        "Training: rsna2024_lumbar/nas/training_bundle/vit_nas_lumbar.py",
        "",
    ]
    return "\n".join(lines)


def _arcname(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        bundle = BUNDLE_DIR.resolve()
        try:
            return path.resolve().relative_to(bundle).as_posix()
        except ValueError:
            return path.name


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    families = list(profile["families"])
    include_crops = _resolve_include_crops(args, profile)
    output = args.output or (REPO_ROOT / str(profile["default_output"]))
    output = output.resolve()

    bundle_files: list[Path] = []
    if not args.skip_bundle:
        legacy = args.legacy_root or default_legacy_root()
        if legacy is None:
            raise SystemExit(
                "Could not find legacy training repo (vit_nas_lumbar.py). "
                "Pass --legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification "
                "or set LUMBAR_LEGACY_ROOT."
            )
        stage_training_bundle(legacy)
        bundle_files = iter_bundle_files(BUNDLE_DIR)
        if not bundle_files:
            raise SystemExit("Training bundle staging produced no files.")

    files = collect_pack_files(
        include_crops=include_crops,
        crop_policy=args.crop_policy,
        bundle_files=bundle_files,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "families": families,
        "allowed_num_gpus": [1, 2, 4, 8],
        "include_crops": include_crops,
        "crop_policy": args.crop_policy,
        "training_bundle": not args.skip_bundle,
        "file_count": len(files),
        "family_output_bases": {name: FAMILY_SPECS[name].output_base for name in families},
    }
    repo_root = REPO_ROOT.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arc = _arcname(path, repo_root)
            zf.write(path, arc)
        zf.writestr("deploy_manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("CLOUD_RUN.txt", cloud_run_text(families, include_crops=include_crops))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")
    print(f"  profile: {args.profile}")
    print(f"  families: {', '.join(families)}")
    print(f"  files: {len(files)}")
    print(f"  crops: {'yes' if include_crops else 'no'}")
    print(f"  training bundle: {'yes' if bundle_files else 'no'}")


if __name__ == "__main__":
    main()
