"""CLI for publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import (
    CONDITION_SPECIFIC_FIGURES_DIR,
    DATASET_DESCRIPTION_FIGURES_DIR,
    ORDINAL_FIGURES_DIR,
    PATIENT_LEVEL_FIGURES_DIR,
    SPINAL_LEVEL_FIGURES_DIR,
)
from rsna2024_lumbar.visualizations.plot_patient_level_results import (
    DEFAULT_REPEAT,
    DEFAULT_SUMMARY,
    generate_patient_level_figures,
)
from rsna2024_lumbar.visualizations.plot_spinal_level_results import (
    DEFAULT_MEANS,
    generate_spinal_level_figures,
)
from rsna2024_lumbar.visualizations.plot_ordinal_severity_results import (
    DEFAULT_MEANS as ORDINAL_DEFAULT_MEANS,
    generate_ordinal_severity_figures,
)
from rsna2024_lumbar.visualizations.plot_condition_specific_results import (
    DEFAULT_MEANS as CONDITION_DEFAULT_MEANS,
    generate_condition_specific_figures,
)
from rsna2024_lumbar.visualizations.plot_dataset_scale import (
    DEFAULT_COUNTS as DATASET_DEFAULT_COUNTS,
    DEFAULT_SCALE as DATASET_DEFAULT_SCALE,
    generate_dataset_scale_figures,
)
from rsna2024_lumbar.visualizations.plot_dicom_reuse import (
    DEFAULT_DUPLICATION,
    DEFAULT_PAIRWISE,
    DEFAULT_SUMMARY as DICOM_DEFAULT_SUMMARY,
    generate_dicom_reuse_figures,
)
from rsna2024_lumbar.visualizations.plot_dicom_resolutions import (
    DEFAULT_COUNTS as RESOLUTION_DEFAULT_COUNTS,
    generate_dicom_resolution_figures,
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
    spinal = sub.add_parser(
        "spinal-level",
        help="Spinal-level bottleneck, profile, overlay, and Macro-F1 heatmap figures.",
    )
    spinal.add_argument(
        "--means",
        type=Path,
        default=DEFAULT_MEANS,
        help="spinal_level_means.csv (or spinal_level_agg.csv).",
    )
    spinal.add_argument(
        "--output-dir",
        type=Path,
        default=SPINAL_LEVEL_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )
    spinal.add_argument("--split", default="validation")

    ordinal = sub.add_parser(
        "ordinal",
        help="Ordinal severity QWK vs O-MAE, metric heatmaps, 2D vs 2.5D, and SER burden figures.",
    )
    ordinal.add_argument(
        "--means",
        type=Path,
        default=ORDINAL_DEFAULT_MEANS,
        help="condition_specific_by_condition.csv (or ordinal_condition_means.csv).",
    )
    ordinal.add_argument(
        "--output-dir",
        type=Path,
        default=ORDINAL_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )
    ordinal.add_argument("--split", default="validation")

    condition = sub.add_parser(
        "condition-specific",
        help="Condition-specific reliability profiles, bottleneck, and uniformity figures.",
    )
    condition.add_argument(
        "--means",
        type=Path,
        default=CONDITION_DEFAULT_MEANS,
        help="condition_specific_by_condition.csv (or condspec_condition_means.csv).",
    )
    condition.add_argument(
        "--output-dir",
        type=Path,
        default=CONDITION_SPECIFIC_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )
    condition.add_argument("--split", default="validation")

    dataset = sub.add_parser(
        "dataset-scale",
        help="Dataset scale bars and condition × severity mix (usable crops).",
    )
    dataset.add_argument("--scale", type=Path, default=DATASET_DEFAULT_SCALE)
    dataset.add_argument("--counts", type=Path, default=DATASET_DEFAULT_COUNTS)
    dataset.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DESCRIPTION_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )

    reuse = sub.add_parser(
        "dicom-reuse",
        help="Stacked crops by within-condition DICOM reuse and cross-condition sharing.",
    )
    reuse.add_argument("--duplication", type=Path, default=DEFAULT_DUPLICATION)
    reuse.add_argument("--summary", type=Path, default=DICOM_DEFAULT_SUMMARY)
    reuse.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE)
    reuse.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DESCRIPTION_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
    )

    resolutions = sub.add_parser(
        "dicom-resolutions",
        help="DICOM resolution histograms by condition (series-matched slices).",
    )
    resolutions.add_argument("--counts", type=Path, default=RESOLUTION_DEFAULT_COUNTS)
    resolutions.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DESCRIPTION_FIGURES_DIR,
        help="Directory for PNG/PDF/CSV/JSON outputs.",
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
    if args.command == "spinal-level":
        generate_spinal_level_figures(
            means_path=args.means,
            output_dir=args.output_dir,
            split=args.split,
        )
        return 0
    if args.command == "ordinal":
        generate_ordinal_severity_figures(
            means_path=args.means,
            output_dir=args.output_dir,
            split=args.split,
        )
        return 0
    if args.command == "condition-specific":
        generate_condition_specific_figures(
            means_path=args.means,
            output_dir=args.output_dir,
            split=args.split,
        )
        return 0
    if args.command == "dataset-scale":
        generate_dataset_scale_figures(
            scale_path=args.scale,
            counts_path=args.counts,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "dicom-reuse":
        generate_dicom_reuse_figures(
            duplication_path=args.duplication,
            summary_path=args.summary,
            pairwise_path=args.pairwise,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "dicom-resolutions":
        generate_dicom_resolution_figures(
            counts_path=args.counts,
            output_dir=args.output_dir,
        )
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
