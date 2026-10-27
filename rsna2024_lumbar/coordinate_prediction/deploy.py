"""One-shot cloud landmark NAS: install, preflight, shard GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rsna2024_lumbar.coordinate_prediction.constants import BATCH_SIZE_DEFAULT, CONDITIONS
from rsna2024_lumbar.coordinate_prediction.resume import wants_resume
from rsna2024_lumbar.coordinate_prediction.search import FAMILIES, FAMILY_FULL
from rsna2024_lumbar.nas.paths import REPO_ROOT, default_data_root
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install deps and run landmark NAS on this machine.")
    p.add_argument("--family", choices=list(FAMILIES), default=FAMILY_FULL)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--condition", choices=list(CONDITIONS), default="spinal_canal_stenosis")
    p.add_argument("--all-conditions", action="store_true")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Fixed batch size for every trial (default {BATCH_SIZE_DEFAULT}).",
    )
    p.add_argument("--max-studies", type=int, default=None)
    p.add_argument(
        "--start-trial",
        type=int,
        default=1,
        help="1-based first trial to run (inclusive). Remaining trials are resharded across GPUs.",
    )
    p.add_argument(
        "--end-trial",
        type=int,
        default=None,
        help="1-based last trial to run (inclusive). Default: last cell of the family grid.",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--skip-completed",
        action="store_true",
        help="Deprecated alias; resume is on by default.",
    )
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def preflight(data_root: Path) -> None:
    data_root = data_root.resolve()
    for name in LABEL_CSVS:
        path = data_root / name
        if not path.is_file():
            raise SystemExit(f"Missing label CSV: {path}")
    images = data_root / "train_images"
    if not images.is_dir():
        raise SystemExit(
            f"Missing {images}\n"
            "Pack with --include-images or copy train_images/ next to the label CSVs."
        )
    n_studies = sum(1 for p in images.iterdir() if p.is_dir())
    if n_studies < 1:
        raise SystemExit(f"No study folders under {images}")
    print(f"Data root: {data_root}")
    print(f"Studies:   {n_studies}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = REPO_ROOT.resolve()
    data_root = (args.data_root or default_data_root()).resolve()
    preflight(data_root)
    if args.dry_run:
        print("Dry-run OK.")
        return 0
    if not args.skip_install:
        print(f"+ {sys.executable} -m pip install -e .[coordinate_prediction]")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[coordinate_prediction]"], cwd=repo, check=True)
    cmd = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.coordinate_prediction.launch",
        "--num-gpus",
        str(args.num_gpus),
        "--data-root",
        str(data_root),
        "--spawn",
        "--wait",
        "--family",
        args.family,
    ]
    if args.all_conditions:
        cmd.append("--all-conditions")
    else:
        cmd.extend(["--condition", args.condition])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_studies is not None:
        cmd.extend(["--max-studies", str(args.max_studies)])
    if args.start_trial != 1:
        cmd.extend(["--start-trial", str(args.start_trial)])
    if args.end_trial is not None:
        cmd.extend(["--end-trial", str(args.end_trial)])
    if wants_resume(args):
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    if args.skip_completed:
        cmd.append("--skip-completed")
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
