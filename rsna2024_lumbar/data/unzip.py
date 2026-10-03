"""
⬇ Unzip a Kaggle dump already sitting in data/raw/.

    python -m rsna2024_lumbar.data.unzip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.utils.kaggle import find_partials, find_zips, require_no_partials, unzip_competition
from common.utils.log import data, die, done, fmt_bytes, next_step, ok, start, step
from rsna2024_lumbar import TRACES_DIR
from rsna2024_lumbar.data.download import DEFAULT_RAW, require_extracted_layout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unzip RSNA 2024 lumbar files already downloaded into data/raw/."
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW,
        help="Folder that holds the .zip (default: rsna2024_lumbar/data/raw).",
    )
    p.add_argument("--keep-zip", action="store_true", help="Leave the .zip after extract.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    start("rsna2024.data.unzip", traces_dir=TRACES_DIR)
    args = parse_args(argv)
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    step("inspect dest")
    zips = find_zips(dest)
    partials = find_partials(dest)
    data(
        dest=dest,
        keep_zip=args.keep_zip,
        zips=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in zips) or "(none)",
        partials=", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in partials) or "(none)",
    )
    step("refuse leftover *.kaggle-partial")
    require_no_partials(dest)
    if not zips:
        step("no zip — check if already extracted")
        try:
            require_extracted_layout(dest)
        except SystemExit:
            die(
                f"Unzip failed: no .zip and no extracted dump under {dest}. "
                "Run python -m rsna2024_lumbar.data.download first."
            )
        ok(f"Already extracted: {dest}")
        next_step("python -m rsna2024_lumbar.data.validate")
        done("unzip (skipped)")
        return
    next_step(f"extract {len(zips)} zip(s)")
    unzip_competition(dest, delete_zip=not args.keep_zip)
    step("verify extracted layout")
    require_extracted_layout(dest)
    ok("Unzip OK")
    next_step("python -m rsna2024_lumbar.data.validate")
    done("unzip")


if __name__ == "__main__":
    main()
