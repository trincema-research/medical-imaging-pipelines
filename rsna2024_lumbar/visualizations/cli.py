"""CLI for publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import PATIENT_LEVEL_FIGURES_DIR
from rsna2024_lumbar.visualizations.plot_patient_level_results import (
    DEFAULT_REPEAT,
    DEFAULT_SUMMARY,
    generate_patient_level_figures,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate publication figures from harvested CSVs.")
    sub = parser.add_subparsers(dest="command", required=True)

    patient = sub.add_parser(
        "patient-level",
        help="Patient error-profile, exact vs O-MAE, and severe-error burden figures.",
    )
    patient.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="patient_level_summary.csv (or patient_summary_computed.csv).",
    )
    patient.add_argument(
        "--repeat-summary",
        type=Path,
        default=DEFAULT_REPEAT,
        help="patient_level_repeat_summary.csv (or patient_repeat_metrics_computed.csv).",
    )
    patient.add_argument(
        "--output-dir",
        type=Path,
        default=PATIENT_LEVEL_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )
    patient.add_argument("--split", default="validation")
    patient.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_DIR,
        help="Unused except to document the default harvest tree.",
    )
    args = parser.parse_args(argv)

    if args.command == "patient-level":
        generate_patient_level_figures(
            summary_path=args.summary,
            repeat_path=args.repeat_summary,
            output_dir=args.output_dir,
            split=args.split,
        )
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
