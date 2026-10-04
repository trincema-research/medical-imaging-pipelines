"""One-shot cloud NAS: install deps, verify data, launch GPU shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rsna2024_lumbar.nas.paths import (
    REPO_ROOT,
    bundle_ready,
    default_crops_root,
    default_data_root,
)
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Install NAS dependencies and run rsna2024_lumbar.nas.launch on this machine.",
    )
    p.add_argument("--family", required=True)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--all-conditions", action="store_true")
    p.add_argument("--condition", default="spinal_canal_stenosis")
    p.add_argument("--crop-policy", default="centered")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--crops-root", type=Path, default=None)
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Preflight only; do not train.")
    return p.parse_args(argv)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def preflight(args: argparse.Namespace) -> None:
    if not bundle_ready():
        raise SystemExit(
            "Missing nas/training_bundle (vit_nas_lumbar.py). "
            "Build the deploy zip on a machine with the legacy lumbar repo:\n"
            "  python -m rsna2024_lumbar.nas.pack --profile efficientnet_deploy "
            "--legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification"
        )
    data_root = (args.data_root or default_data_root()).resolve()
    crops_root = (args.crops_root or default_crops_root(args.crop_policy)).resolve()
    for name in LABEL_CSVS:
        path = data_root / name
        if not path.is_file():
            raise SystemExit(f"Missing label CSV: {path}")
    if not crops_root.is_dir():
        raise SystemExit(
            f"Missing PNG crop cache: {crops_root}\n"
            "Pack with --include-crops or export crops into data/processed/<policy>/."
        )
    png_count = sum(1 for _ in crops_root.rglob("*.png"))
    if png_count < 1:
        raise SystemExit(f"No PNG files under {crops_root}")
    print(f"Data root:   {data_root}")
    print(f"Crops root:  {crops_root} ({png_count} PNG files indexed)")
    print(f"Bundle:      OK")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo = REPO_ROOT.resolve()
    preflight(args)
    if args.dry_run:
        print("Dry-run OK.")
        return
    if not args.skip_install:
        _run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo)
        req = repo / "rsna2024_lumbar" / "nas" / "requirements-nas-cloud.txt"
        if req.is_file():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=repo)
    launch_argv = [
        "--family",
        args.family,
        "--num-gpus",
        str(args.num_gpus),
        "--crop-policy",
        args.crop_policy,
        "--spawn",
        "--wait",
        "--epochs",
        str(args.epochs),
        "--progress",
        "--amp",
    ]
    if args.early_stop_patience is not None:
        launch_argv.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if args.data_root is not None:
        launch_argv.extend(["--data-root", str(args.data_root)])
    if args.crops_root is not None:
        launch_argv.extend(["--crops-root", str(args.crops_root)])
    if args.all_conditions:
        launch_argv.append("--all-conditions")
    else:
        launch_argv.extend(["--condition", args.condition])
    from rsna2024_lumbar.nas import launch

    launch.main(launch_argv)


if __name__ == "__main__":
    main()
