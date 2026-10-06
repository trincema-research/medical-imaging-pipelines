"""Install deps and run best_config_runs on one GPU (optionally after unpack)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rsna2024_lumbar.nas.paths import BUNDLE_DIR, default_crops_root, default_data_root
from rsna2024_lumbar.best_config_runs.cloud_common import read_manifest, repo_root_from_extracted
from rsna2024_lumbar.best_config_runs.paths import DEFAULT_BEST_CONFIG_DIR, DEFAULT_REPEATS, RESULTS_DIR
from rsna2024_lumbar.best_config_runs.unpack_cloud import unpack
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unpack (optional), install torch deps, run best_config_runs run-all (parallel GPUs)."
    )
    p.add_argument("--zip", type=Path, default=None, help="If set, unpack to --dest before training.")
    p.add_argument("--dest", type=Path, default=None, help="Unpack target when using --zip.")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root with pyproject.toml (default: cwd or parent after unpack).",
    )
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Match NAS early stopping (default: 5). Pass 0 to disable.",
    )
    p.add_argument("--crop-policy", default="centered")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--crops-root", type=Path, default=None)
    p.add_argument("--output-base", type=Path, default=None)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Preflight only; no pip or training.")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STEM",
        help="Optional subset of nas_best_* stems.",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        metavar="N",
        help="Shard run-all across GPUs: 1=sequential, 0=auto, or 2/4/8 (default: 8).",
    )
    return p.parse_args(argv)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def preflight(repo_root: Path, args: argparse.Namespace) -> None:
    data_root = (args.data_root or default_data_root()).resolve()
    crops_root = (args.crops_root or default_crops_root(args.crop_policy)).resolve()
    for name in LABEL_CSVS:
        path = data_root / name
        if not path.is_file():
            raise SystemExit(f"Missing label CSV: {path}")
    if not crops_root.is_dir():
        raise SystemExit(f"Missing PNG crops: {crops_root}")
    if not DEFAULT_BEST_CONFIG_DIR.is_dir():
        raise SystemExit(f"Missing best configs: {DEFAULT_BEST_CONFIG_DIR}")
    bundle_train = BUNDLE_DIR / "train_vit_lumbar.py"
    if not bundle_train.is_file():
        raise SystemExit(f"Missing training bundle: {bundle_train}")
    print(f"Repo root:  {repo_root}")
    print(f"Data root:  {data_root}")
    print(f"Crops root: {crops_root}")
    print(f"Results:    {args.output_base or RESULTS_DIR}")


def resolve_repo_root(args: argparse.Namespace) -> Path:
    if args.zip is not None:
        if args.dest is None:
            raise SystemExit("--dest is required when using --zip")
        return unpack(args.zip, args.dest, force=False)
    if args.repo_root is not None:
        return repo_root_from_extracted(args.repo_root)
    return repo_root_from_extracted(Path.cwd())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = resolve_repo_root(args)
    manifest = read_manifest(repo_root)
    if args.repeats == DEFAULT_REPEATS and manifest.get("repeats_default"):
        pass
    preflight(repo_root, args)
    if args.dry_run:
        print("Dry-run OK.")
        return

    if not args.skip_install:
        _run([sys.executable, "-m", "pip", "install", "-e", ".[best_config_runs]"], cwd=repo_root)
        req = repo_root / "rsna2024_lumbar" / "nas" / "requirements-nas-cloud.txt"
        if req.is_file():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=repo_root)

    patience: int | None = args.early_stop_patience
    if patience is not None and patience < 1:
        patience = None

    run_argv = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.best_config_runs",
        "run-all",
        "--repeats",
        str(args.repeats),
        "--epochs",
        str(args.epochs),
        "--data-root",
        str(args.data_root or default_data_root()),
        "--crops-root",
        str(args.crops_root or default_crops_root(args.crop_policy)),
    ]
    if patience is not None:
        run_argv.extend(["--early-stop-patience", str(patience)])
    if args.skip_completed:
        run_argv.append("--skip-completed")
    if args.output_base is not None:
        run_argv.extend(["--output-base", str(args.output_base)])
    if args.only:
        run_argv.extend(["--only", *args.only])
    if args.num_gpus != 1:
        run_argv.extend(["--num-gpus", str(args.num_gpus)])

    print(f"+ {' '.join(run_argv)}")
    completed = subprocess.run(run_argv, cwd=repo_root, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
