#!/usr/bin/env python3
"""
Lumbar best_config_runs on a cloud GPU — one script, no repo checkout required to unzip.

Typical workflow (zip + this file in /workspace):

  # 1) Extract
  python lumbar_best_config_runs.py unzip

  # 2) Train (installs deps, then run-all on 8 GPUs by default)
  python lumbar_best_config_runs.py run --repeats 5 --num-gpus 8

Or both steps:

  python lumbar_best_config_runs.py all --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

MANIFEST_NAME = "best_config_runs_deploy_manifest.json"
DEFAULT_ZIP = "lumbar_best_config_runs_cloud.zip"
DEFAULT_DEST = "perf_cloud"


def repo_root_from(extract_dir: Path) -> Path:
    extract_dir = extract_dir.resolve()
    if (extract_dir / "pyproject.toml").is_file():
        return extract_dir
    for child in extract_dir.iterdir():
        if child.is_dir() and (child / "pyproject.toml").is_file():
            return child.resolve()
    raise SystemExit(f"No pyproject.toml under {extract_dir}")


def cmd_unzip(zip_path: Path, dest: Path, *, force: bool) -> Path:
    zip_path = zip_path.resolve()
    dest = dest.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"Zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / "pyproject.toml").is_file():
        raise SystemExit(f"{dest} already has pyproject.toml — use --force or another --dest")
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    root = repo_root_from(dest)
    manifest = root / MANIFEST_NAME
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"  repeats (pack default): {data.get('repeats_default', '?')}")
        print(f"  expected CSV rows: {data.get('expected_result_rows', '?')}")
    print(f"Repo root: {root}")
    return root


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def cmd_run(
    repo_root: Path,
    *,
    repeats: int,
    epochs: int,
    patience: int,
    num_gpus: int,
    skip_completed: bool,
    skip_install: bool,
    seeds: list[int] | None = None,
    condition_specific_seeds: bool = False,
) -> None:
    repo_root = repo_root.resolve()
    if not (repo_root / "pyproject.toml").is_file():
        raise SystemExit(f"Not a repo root: {repo_root}")

    if not skip_install:
        _run([sys.executable, "-m", "pip", "install", "-e", ".[best_config_runs]"], cwd=repo_root)
        req = repo_root / "rsna2024_lumbar" / "nas" / "requirements-nas-cloud.txt"
        if req.is_file():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=repo_root)

    argv = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.best_config_runs",
        "run-all",
        "--repeats",
        str(repeats),
        "--epochs",
        str(epochs),
        "--early-stop-patience",
        str(patience),
        "--data-root",
        str(repo_root / "rsna2024_lumbar" / "data" / "raw"),
        "--crops-root",
        str(repo_root / "rsna2024_lumbar" / "data" / "processed" / "centered"),
    ]
    if skip_completed:
        argv.append("--skip-completed")
    if num_gpus != 1:
        argv.extend(["--num-gpus", str(num_gpus)])
    if condition_specific_seeds:
        argv.append("--condition-specific-seeds")
    elif seeds:
        argv.extend(["--seeds", *[str(s) for s in seeds]])

    completed = subprocess.run(argv, cwd=repo_root, check=False)
    raise SystemExit(completed.returncode)


def resolve_repo_root(dest: Path) -> Path:
    if (dest / "pyproject.toml").is_file():
        return dest.resolve()
    return repo_root_from(dest)


def main() -> None:
    here = Path.cwd()
    p = argparse.ArgumentParser(description="Unzip and run lumbar best_config_runs (parallel GPUs).")
    sub = p.add_subparsers(dest="command", required=True)

    u = sub.add_parser("unzip", help="Extract the cloud zip only.")
    u.add_argument("--zip", type=Path, default=Path(DEFAULT_ZIP))
    u.add_argument("--dest", type=Path, default=Path(DEFAULT_DEST))
    u.add_argument("--force", action="store_true")

    r = sub.add_parser("run", help="Install deps and start run-all (after unzip).")
    r.add_argument(
        "--repo-root",
        type=Path,
        default=Path(DEFAULT_DEST),
        help=f"Extracted tree (default: ./{DEFAULT_DEST})",
    )
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--epochs", type=int, default=50)
    r.add_argument("--early-stop-patience", type=int, default=5)
    r.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        metavar="N",
        help="Parallel GPUs for run-all (default: 8; use 1 for sequential).",
    )
    r.add_argument("--skip-completed", action="store_true", default=True)
    r.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")
    r.add_argument("--skip-install", action="store_true")
    r.add_argument("--seeds", type=int, nargs="+", default=None)
    r.add_argument(
        "--condition-specific-seeds",
        action="store_true",
        help="Use split seeds 42 43 44 45 46 (Beyond-Accuracy condition-specific protocol).",
    )

    a = sub.add_parser("all", help="unzip then run (same flags as run).")
    a.add_argument("--zip", type=Path, default=Path(DEFAULT_ZIP))
    a.add_argument("--dest", type=Path, default=Path(DEFAULT_DEST))
    a.add_argument("--force", action="store_true")
    a.add_argument("--repeats", type=int, default=5)
    a.add_argument("--epochs", type=int, default=50)
    a.add_argument("--early-stop-patience", type=int, default=5)
    a.add_argument("--num-gpus", type=int, default=8, metavar="N")
    a.add_argument("--skip-completed", action="store_true", default=True)
    a.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")
    a.add_argument("--skip-install", action="store_true")
    a.add_argument("--seeds", type=int, nargs="+", default=None)
    a.add_argument(
        "--condition-specific-seeds",
        action="store_true",
        help="Use split seeds 42 43 44 45 46 (Beyond-Accuracy condition-specific protocol).",
    )

    args = p.parse_args()
    os.chdir(here)

    if args.command == "unzip":
        cmd_unzip(args.zip, args.dest, force=args.force)
        return

    if args.command == "run":
        root = resolve_repo_root(args.repo_root)
        cmd_run(
            root,
            repeats=args.repeats,
            epochs=args.epochs,
            patience=args.early_stop_patience,
            num_gpus=args.num_gpus,
            skip_completed=args.skip_completed,
            skip_install=args.skip_install,
            seeds=args.seeds,
            condition_specific_seeds=args.condition_specific_seeds,
        )
        return

    if args.command == "all":
        cmd_unzip(args.zip, args.dest, force=args.force)
        root = resolve_repo_root(args.dest)
        cmd_run(
            root,
            repeats=args.repeats,
            epochs=args.epochs,
            patience=args.early_stop_patience,
            num_gpus=args.num_gpus,
            skip_completed=args.skip_completed,
            skip_install=args.skip_install,
            seeds=args.seeds,
            condition_specific_seeds=args.condition_specific_seeds,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
