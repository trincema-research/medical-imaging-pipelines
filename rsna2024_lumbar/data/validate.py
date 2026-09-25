"""
✔ Check a RSNA 2024 raw dump matches what preprocess expects.

    python -m rsna2024_lumbar.data.validate
    python -m rsna2024_lumbar.data.validate --data-root rsna2024_lumbar/tests/fixtures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rsna2024_lumbar.data.download import DEFAULT_RAW
from rsna2024_lumbar.preprocessing.constants import (
    CONDITIONS,
    LABEL_CSVS,
    LEVEL_DISPLAY,
    LUMBAR_LEVELS,
    SEVERITY_CLASSES,
    label_column,
)

COORD_COLUMNS = ("study_id", "series_id", "instance_number", "condition", "level", "x", "y")
SERIES_COLUMNS = ("study_id", "series_id", "series_description")


@dataclass
class RawReport:
    data_root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_studies: int = 0
    n_coord_rows: int = 0
    n_series_rows: int = 0
    n_dicoms: int = 0
    n_coord_files_ok: int = 0
    n_coord_files_missing: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _need_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"{name}: missing columns {missing}"] if missing else []


def validate_raw(
    data_root: Path,
    *,
    max_coord_checks: int | None = None,
    strict: bool = False,
) -> RawReport:
    root = Path(data_root).resolve()
    report = RawReport(data_root=root)
    if not root.is_dir():
        report.errors.append(f"data-root does not exist: {root}")
        return report

    for name in LABEL_CSVS:
        if not (root / name).is_file():
            report.errors.append(f"missing file: {name}")
    images = root / "train_images"
    if not images.is_dir():
        report.errors.append("missing directory: train_images/")
    if report.errors:
        return report

    labels = pd.read_csv(root / "train.csv")
    coords = pd.read_csv(root / "train_label_coordinates.csv")
    series = pd.read_csv(root / "train_series_descriptions.csv")
    report.n_studies = int(labels["study_id"].nunique()) if "study_id" in labels.columns else 0
    report.n_coord_rows = len(coords)
    report.n_series_rows = len(series)
    report.n_dicoms = sum(1 for _ in images.rglob("*.dcm"))

    if "study_id" not in labels.columns:
        report.errors.append("train.csv: missing study_id")
    label_cols = [label_column(c, lv) for c in CONDITIONS for lv in LUMBAR_LEVELS]
    report.errors.extend(_need_columns(labels, tuple(label_cols), "train.csv"))
    report.errors.extend(_need_columns(coords, COORD_COLUMNS, "train_label_coordinates.csv"))
    report.errors.extend(_need_columns(series, SERIES_COLUMNS, "train_series_descriptions.csv"))

    if report.n_studies == 0:
        report.errors.append("train.csv: no studies")
    if report.n_dicoms == 0:
        report.errors.append("train_images/: no .dcm files")

    coord_names = {str(spec["coord_name"]) for spec in CONDITIONS.values()}
    if "condition" in coords.columns:
        unknown = sorted(set(coords["condition"].astype(str)) - coord_names)
        if unknown:
            report.warnings.append(f"unexpected condition names in coordinates: {unknown}")
    if "level" in coords.columns:
        known_levels = set(LEVEL_DISPLAY.values())
        unknown_lv = sorted(set(coords["level"].astype(str)) - known_levels)
        if unknown_lv:
            report.warnings.append(f"unexpected level names in coordinates: {unknown_lv}")

    if set(label_cols).issubset(labels.columns):
        bad = set()
        for col in label_cols:
            vals = labels[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            bad.update(set(vals) - set(SEVERITY_CLASSES))
        if bad:
            report.errors.append(f"train.csv: unexpected severity labels {sorted(bad)}")

    if report.errors:
        return report

    sample = coords
    if max_coord_checks is not None:
        sample = coords.head(max_coord_checks)
    for row in sample.itertuples(index=False):
        path = images / str(int(row.study_id)) / str(int(row.series_id)) / f"{int(row.instance_number)}.dcm"
        if path.is_file():
            report.n_coord_files_ok += 1
        else:
            report.n_coord_files_missing += 1
    if report.n_coord_files_missing:
        msg = (
            f"{report.n_coord_files_missing} coordinate row(s) point at a missing DICOM "
            "(preprocess skips those studies)"
        )
        if strict:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
    return report


def print_report(report: RawReport) -> None:
    print(f"data-root: {report.data_root}")
    print(f"studies:   {report.n_studies:,}")
    print(f"coords:    {report.n_coord_rows:,}")
    print(f"series:    {report.n_series_rows:,}")
    print(f"dicoms:    {report.n_dicoms:,}")
    print(f"coord->dcm: {report.n_coord_files_ok:,} ok / {report.n_coord_files_missing:,} missing")
    for w in report.warnings:
        print(f"warning: {w}")
    if report.ok:
        print("OK")
    else:
        for e in report.errors:
            print(f"error: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an RSNA 2024 lumbar raw dump.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    p.add_argument(
        "--max-coord-checks",
        type=int,
        default=None,
        help="Only check the first N coordinate paths (faster on the full dump).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing coordinate DICOMs as errors (default: warning).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_raw(
        args.data_root,
        max_coord_checks=args.max_coord_checks,
        strict=args.strict,
    )
    print_report(report)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
