"""Zip code, labels, NAS best configs, and optional PNG crops for cloud perf_pipeline runs."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from rsna2024_lumbar.nas.paths import REPO_ROOT
from rsna2024_lumbar.perf_pipeline.cloud_common import (
    CLOUD_RUN_NAME,
    MANIFEST_NAME,
    arcname,
    build_manifest,
    cloud_run_text,
    collect_pack_files,
    stage_bundle_if_needed,
)
from rsna2024_lumbar.perf_pipeline.paths import DEFAULT_BEST_CONFIG_DIR, DEFAULT_REPEATS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a zip for single-GPU perf_pipeline retrain on cloud (metrics + severity CSVs)."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "lumbar_perf_pipeline_cloud.zip",
        help="Output zip path.",
    )
    p.add_argument(
        "--no-include-crops",
        action="store_true",
        help="Omit PNG crops (default: include centered PNG cache).",
    )
    p.add_argument("--crop-policy", default="centered")
    p.add_argument(
        "--best-config-dir",
        type=Path,
        default=DEFAULT_BEST_CONFIG_DIR,
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Documented default for cloud deploy (default: 5).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Documented NAS-style early stopping for cloud deploy (default: 5).",
    )
    p.add_argument(
        "--legacy-root",
        type=Path,
        default=None,
        help="Legacy lumbar repo for staging train_vit_lumbar.py into the bundle.",
    )
    p.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Do not refresh training_bundle from legacy (use bundle already in tree).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    include_crops = not args.no_include_crops
    bundle_files = stage_bundle_if_needed(
        legacy_root=args.legacy_root,
        skip_bundle=args.skip_bundle,
    )
    files = collect_pack_files(
        include_crops=include_crops,
        crop_policy=args.crop_policy,
        bundle_files=bundle_files,
        best_config_dir=args.best_config_dir.resolve(),
    )
    manifest = build_manifest(
        include_crops=include_crops,
        crop_policy=args.crop_policy,
        file_count=len(files),
        repeats=args.repeats,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        best_config_dir=args.best_config_dir.resolve(),
    )
    repo_root = REPO_ROOT.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, arcname(path, repo_root))
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        zf.writestr(
            CLOUD_RUN_NAME,
            cloud_run_text(
                repeats=args.repeats,
                epochs=args.epochs,
                early_stop_patience=args.early_stop_patience,
                include_crops=include_crops,
            ),
        )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")
    print(f"  files: {len(files)}")
    print(f"  best configs: {len(manifest['best_config_files'])}")
    print(f"  crops: {'yes' if include_crops else 'no'}")
    print(f"  cloud defaults: repeats={args.repeats} epochs={args.epochs} patience={args.early_stop_patience}")


if __name__ == "__main__":
    main()
