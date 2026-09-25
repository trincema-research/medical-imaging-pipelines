"""
⬇ Download the Kaggle RSNA 2024 lumbar dump into data/raw/.

Not stored under data/raw/ itself — that folder is gitignored.

    pip install -e ".[kaggle]"
    python -m rsna2024_lumbar.data.download

Needs a Kaggle token (~/.kaggle/kaggle.json) and accepted competition rules:
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.utils.kaggle import download_competition
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dest = args.dest.resolve()
    if raw_looks_complete(dest) and not args.force:
        print(f"Already present: {dest}")
        print("Pass --force to download again.")
        return
    print(f"Kaggle: {KAGGLE_SLUG}")
    print(f"Dest:   {dest}")
    download_competition(
        KAGGLE_SLUG,
        dest,
        force=args.force,
        unzip=not args.keep_zip,
    )
    missing = [n for n in LABEL_CSVS if not (dest / n).is_file()]
    if missing or not (dest / "train_images").is_dir():
        raise SystemExit(
            f"Download finished but layout is incomplete under {dest}: missing {missing} "
            "or train_images/."
        )
    print("Ready for: python -m rsna2024_lumbar.preprocessing --data-root", dest)


if __name__ == "__main__":
    main()
