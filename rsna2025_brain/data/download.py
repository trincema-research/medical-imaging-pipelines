"""
⬇ Download the Kaggle RSNA 2025 brain dump into data/raw/.

    pip install -e ".[kaggle]"
    python -m rsna2025_brain.data.download

Uses the same repo-root `.env` `KAGGLE_API_TOKEN` as lumbar. Accept the rules first:
    https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common.utils.env import load_repo_env
from common.utils.kaggle import download_competition, find_partials, find_zips
from common.utils.log import data, die, disk_free, done, fmt_bytes, next_step, ok, start, step
from rsna2025_brain import TRACES_DIR
from rsna2025_brain.constants import ID_COLUMN, LABEL_CSVS, SERIES_DIRNAME

KAGGLE_SLUG = "rsna-intracranial-aneurysm-detection"
HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download RSNA 2025 brain files from Kaggle into data/raw/."
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW,
        help="Extract folder (default: rsna2025_brain/data/raw).",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if the dump looks complete.")
    p.add_argument("--keep-zip", action="store_true", help="Leave the .zip after extract.")
    return p.parse_args(argv)


def _n_label_series(dest: Path) -> int:
    with (dest / "train.csv").open(encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh)
        if not rows.fieldnames or ID_COLUMN not in rows.fieldnames:
            return 0
        return len({(row.get(ID_COLUMN) or "").strip() for row in rows if (row.get(ID_COLUMN) or "").strip()})


def raw_looks_complete(dest: Path) -> bool:
    """True only when on-disk series folders cover every SeriesInstanceUID in train.csv."""
    series = dest / SERIES_DIRNAME
    if not (dest / "train.csv").is_file() or not series.is_dir():
        return False
    n_dirs = sum(1 for p in series.iterdir() if p.is_dir())
    n_csv = _n_label_series(dest)
    return n_csv > 0 and n_dirs >= n_csv


def require_extracted_layout(dest: Path) -> None:
    dest = Path(dest)
    present = {n: (dest / n).is_file() for n in LABEL_CSVS}
    images = (dest / SERIES_DIRNAME).is_dir()
    data(layout={**present, f"{SERIES_DIRNAME}/": images})
    missing = [n for n, ok_file in present.items() if not ok_file]
    if missing or not images:
        die(
            f"Download/unzip finished but layout is incomplete under {dest}: "
            f"missing {missing or [f'{SERIES_DIRNAME}/']}."
        )


def _dest_snapshot(dest: Path) -> None:
    zips = find_zips(dest)
    partials = find_partials(dest)
    data(
        dest=dest,
        exists=dest.is_dir(),
        train_csv=(dest / "train.csv").is_file(),
        series=(dest / SERIES_DIRNAME).is_dir(),
        zips=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in zips) or "(none)",
        partials=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in partials) or "(none)",
        disk_free=disk_free(dest),
    )


def main(argv: list[str] | None = None) -> None:
    start("rsna2025.data.download", traces_dir=TRACES_DIR)
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
        next_step("python -m rsna2025_brain.data.validate")
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
    next_step("python -m rsna2025_brain.data.validate")
    if args.keep_zip:
        next_step("python -m rsna2025_brain.data.unzip  (optional; zip kept)")
    done("download")


if __name__ == "__main__":
    main()
