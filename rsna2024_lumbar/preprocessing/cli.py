"""CLI: python -m rsna2024_lumbar.preprocessing ..."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.preprocessing.constants import CONDITIONS, CROP_POLICIES, CROP_POLICY_CENTERED
from rsna2024_lumbar.preprocessing.crops import crop_settings_for, describe_settings
from rsna2024_lumbar.preprocessing.export import export_crops, print_export_summary

HERE = Path(__file__).resolve().parents[1]
DEFAULT_RAW = HERE / "data" / "raw"
DEFAULT_PROCESSED = HERE / "data" / "processed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop RSNA 2024 lumbar DICOMs to a PNG cache the trainer can load."
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: data/processed/<crop-policy>/",
    )
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    p.add_argument("--condition", choices=list(CONDITIONS), default=None)
    p.add_argument(
        "--max-studies",
        type=int,
        default=None,
        metavar="N",
        help="Smoke-test prefix of train.csv (must be >= 1).",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_studies is not None and args.max_studies < 1:
        raise SystemExit("--max-studies must be >= 1.")
    output = args.output_root or (DEFAULT_PROCESSED / args.crop_policy)
    conditions = [args.condition] if args.condition else None
    sample = (conditions or list(CONDITIONS))[0]
    print(f"Crop policy: {args.crop_policy}")
    print(f"Example ({sample}): {describe_settings(crop_settings_for(sample, args.crop_policy))}")
    result = export_crops(
        args.data_root,
        output,
        crop_policy=args.crop_policy,
        conditions=conditions,
        max_studies=args.max_studies,
        skip_existing=args.skip_existing,
        progress=args.progress,
    )
    print_export_summary(result)
