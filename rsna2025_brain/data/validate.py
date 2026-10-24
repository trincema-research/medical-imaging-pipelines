"""
✔ Check a RSNA 2025 raw dump matches the official Kaggle layout.

    python -m rsna2025_brain.data.validate
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.utils.log import data, done, fail, next_step, ok, start, step, warn
from rsna2025_brain import TRACES_DIR
from rsna2025_brain.constants import (
    ID_COLUMN,
    LABEL_CSVS,
    LOCALIZER_COLUMNS,
    LOCATION_COLUMNS,
    METADATA_COLUMNS,
    PRESENCE_COLUMN,
    SERIES_DIRNAME,
    TARGET_COLUMNS,
)
from rsna2025_brain.data.download import DEFAULT_RAW


@dataclass
class RawReport:
    data_root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_series: int = 0
    n_localizer_rows: int = 0
    n_dicoms: int = 0
    n_series_dirs_ok: int = 0
    n_series_dirs_missing: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _need_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"{name}: missing columns {missing}"] if missing else []


def validate_raw(
    data_root: Path,
    *,
    max_series_checks: int | None = None,
    strict: bool = False,
) -> RawReport:
    root = Path(data_root).resolve()
    report = RawReport(data_root=root)
    step("check data-root exists")
    data(data_root=root, exists=root.is_dir())
    if not root.is_dir():
        report.errors.append(f"data-root does not exist: {root}")
        return report

    step("check required CSVs + series/")
    for name in LABEL_CSVS:
        present = (root / name).is_file()
        data(file=name, present=present, size=(root / name).stat().st_size if present else 0)
        if not present:
            report.errors.append(f"missing file: {name}")
    images = root / SERIES_DIRNAME
    data(series=images.is_dir())
    if not images.is_dir():
        report.errors.append(f"missing directory: {SERIES_DIRNAME}/")
    if report.errors:
        data(errors=len(report.errors))
        return report

    step("read label CSVs")
    labels = pd.read_csv(root / "train.csv")
    localizers = pd.read_csv(root / "train_localizers.csv")
    report.n_series = int(labels[ID_COLUMN].nunique()) if ID_COLUMN in labels.columns else 0
    report.n_localizer_rows = len(localizers)
    step("count DICOMs under series/")
    report.n_dicoms = sum(1 for _ in images.rglob("*.dcm"))
    data(
        series=report.n_series,
        localizer_rows=report.n_localizer_rows,
        dicoms=report.n_dicoms,
        label_cols=list(labels.columns)[:6],
    )

    report.errors.extend(_need_columns(labels, (*METADATA_COLUMNS, *TARGET_COLUMNS), "train.csv"))
    report.errors.extend(_need_columns(localizers, LOCALIZER_COLUMNS, "train_localizers.csv"))
    if report.n_series == 0:
        report.errors.append("train.csv: no series")
    if report.n_dicoms == 0:
        report.errors.append("series/: no .dcm files")

    if set(TARGET_COLUMNS).issubset(labels.columns):
        bad: set[str] = set()
        for col in TARGET_COLUMNS:
            vals = labels[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            bad.update(set(vals) - {"0", "1", "0.0", "1.0"})
        if bad:
            report.errors.append(f"train.csv: unexpected label values {sorted(bad)}")

    if "location" in localizers.columns:
        known = set(LOCATION_COLUMNS) | {PRESENCE_COLUMN}
        unknown = sorted(set(localizers["location"].dropna().astype(str)) - known)
        if unknown:
            report.warnings.append(f"unexpected location names in localizers: {unknown}")

    if report.errors:
        return report

    uids = labels[ID_COLUMN].astype(str).tolist()
    if max_series_checks is not None:
        uids = uids[:max_series_checks]
    step("map train.csv rows to series folders")
    data(series_checks=len(uids), strict=strict)
    for uid in uids:
        folder = images / uid
        if folder.is_dir() and any(folder.glob("*.dcm")):
            report.n_series_dirs_ok += 1
        else:
            report.n_series_dirs_missing += 1
    if report.n_series_dirs_missing:
        msg = f"{report.n_series_dirs_missing} train.csv series folder(s) missing under {SERIES_DIRNAME}/"
        if strict:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
    data(series_ok=report.n_series_dirs_ok, series_missing=report.n_series_dirs_missing)
    return report


def print_report(report: RawReport) -> None:
    step("report")
    data(f"data-root: {report.data_root}")
    data(f"series:    {report.n_series:,}")
    data(f"localizers:{report.n_localizer_rows:,}")
    data(f"dicoms:    {report.n_dicoms:,}")
    data(f"series->dir: {report.n_series_dirs_ok:,} ok / {report.n_series_dirs_missing:,} missing")
    for w in report.warnings:
        warn(f"warning: {w}")
    if report.ok:
        ok("OK")
    else:
        for e in report.errors:
            fail(f"error: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an RSNA 2025 brain raw dump.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    p.add_argument(
        "--max-series-checks",
        type=int,
        default=None,
        help="Only check the first N SeriesInstanceUID folders (faster on the full dump).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing series folders as errors (default: warning).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    start("rsna2025.data.validate", traces_dir=TRACES_DIR)
    args = parse_args(argv)
    data(data_root=args.data_root, max_series_checks=args.max_series_checks, strict=args.strict)
    next_step("validate_raw")
    report = validate_raw(
        args.data_root,
        max_series_checks=args.max_series_checks,
        strict=args.strict,
    )
    print_report(report)
    if not report.ok:
        next_step("fix the dump (re-download / unzip) then re-run validate")
        done("validate (failed)")
        raise SystemExit(1)
    next_step("dump is ready (train.csv + train_localizers.csv + series/)")
    done("validate")


if __name__ == "__main__":
    main()
