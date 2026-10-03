"""
⬇ Download the Kaggle RSNA 2024 lumbar dump into data/raw/.

Not stored under data/raw/ itself — that folder is gitignored.

    pip install -e ".[kaggle]"
    python -m rsna2024_lumbar.data.download

Needs repo-root `.env` with `KAGGLE_API_TOKEN` (copy `.env.example`) and accepted rules:
    https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.utils.env import load_repo_env
from common.utils.kaggle import download_competition, find_partials, find_zips
from common.utils.log import data, die, disk_free, done, fmt_bytes, next_step, ok, start, step
from rsna2024_lumbar import TRACES_DIR
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS

KAGGLE_SLUG = "rsna-2024-lumbar-spine-degenerative-classification"
HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download RSNA 2024 lumbar files from Kaggle into data/raw/."
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW,
        help="Extract folder (default: rsna2024_lumbar/data/raw).",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if train.csv exists.")
    p.add_argument("--keep-zip", action="store_true", help="Leave the .zip after extract.")
    return p.parse_args(argv)


def raw_looks_complete(dest: Path) -> bool:
    return (dest / "train.csv").is_file() and (dest / "train_images").is_dir()


def require_extracted_layout(dest: Path) -> None:
    dest = Path(dest)
    present = {n: (dest / n).is_file() for n in LABEL_CSVS}
    images = (dest / "train_images").is_dir()
    data(layout={**present, "train_images/": images})
    missing = [n for n, ok_file in present.items() if not ok_file]
    if missing or not images:
        die(
            f"Download/unzip finished but layout is incomplete under {dest}: "
            f"missing {missing or ['train_images/']}."
        )


def _dest_snapshot(dest: Path) -> None:
    zips = find_zips(dest)
    partials = find_partials(dest)
    data(
        dest=dest,
        exists=dest.is_dir(),
        train_csv=(dest / "train.csv").is_file(),
        train_images=(dest / "train_images").is_dir(),
        zips=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in zips) or "(none)",
        partials=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in partials) or "(none)",
        disk_free=disk_free(dest),
    )


def main(argv: list[str] | None = None) -> None:
    start("rsna2024.data.download", traces_dir=TRACES_DIR)
    step("load repo-root .env")
    env_path = load_repo_env()
    data(env_file=env_path or "(none)")
    args = parse_args(argv)
    dest = args.dest.resolve()
    step("inspect dest")
    data(slug=KAGGLE_SLUG, force=args.force, keep_zip=args.keep_zip)
    dest.mkdir(parents=True, exist_ok=True)
    _dest_snapshot(dest)
    if raw_looks_complete(dest) and not args.force:
        ok(f"Already present: {dest}")
        next_step("python -m rsna2024_lumbar.data.validate")
        done("download (skipped)")
        return
    next_step("kaggle competitions download + unzip")
    download_competition(
        KAGGLE_SLUG,
        dest,
        force=args.force,
        unzip=True,
        delete_zip=not args.keep_zip,
    )
    step("verify extracted layout")
    _dest_snapshot(dest)
    require_extracted_layout(dest)
    ok("Download + unzip OK")
    next_step("python -m rsna2024_lumbar.data.validate")
    if args.keep_zip:
        next_step("python -m rsna2024_lumbar.data.unzip  (optional; zip kept)")
    done("download")


if __name__ == "__main__":
    main()
