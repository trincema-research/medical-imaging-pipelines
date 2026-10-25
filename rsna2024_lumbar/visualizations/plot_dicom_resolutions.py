"""DICOM resolution histograms by condition.

Reads inventory CSVs only. Count labels use thousands separators for 5+ digit
integers and leave four-digit values ungrouped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsna2024_lumbar.visualizations import DATASET_DESCRIPTION_FIGURES_DIR, YEAR_ROOT
from rsna2024_lumbar.visualizations.plot_dataset_scale import (
    CONDITION_ORDER,
    CONDITION_SHORT,
    _save_figure,
    _style,
    format_count,
    save_metadata,
)

DEFAULT_COUNTS = YEAR_ROOT / "data" / "dataset_description" / "dicom_resolution_counts_by_condition.csv"
TOP_N = 12
CONDITION_COLORS: dict[str, str] = {
    "spinal_canal_stenosis": "#2980B9",
    "left_neural_foraminal_narrowing": "#27AE60",
    "right_neural_foraminal_narrowing": "#16A085",
    "left_subarticular_stenosis": "#E67E22",
    "right_subarticular_stenosis": "#D35400",
}
CONDITION_KEYWORDS: dict[str, str] = {
    "spinal_canal_stenosis": "sagittal t2",
    "left_neural_foraminal_narrowing": "sagittal t1",
    "right_neural_foraminal_narrowing": "sagittal t1",
    "left_subarticular_stenosis": "axial t2",
    "right_subarticular_stenosis": "axial t2",
}


def _resolution_sort_key(label: str) -> tuple[int, int, int]:
    width, height = map(int, label.lower().replace("×", "x").split("x"))
    return (-(width * height), -width, -height)


def load_resolution_counts(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"condition", "resolution", "count"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(raw["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(raw['condition']))}")
    return raw


def top_resolution_labels(counts: pd.DataFrame, *, top_n: int = TOP_N) -> list[str]:
    totals = counts.groupby("resolution")["count"].sum()
    labels = [str(label) for label in totals.nlargest(top_n).index]
    return sorted(labels, key=_resolution_sort_key)


def plot_resolutions_by_condition(counts: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    top_labels = top_resolution_labels(counts)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, condition in enumerate(CONDITION_ORDER):
        ax = axes_flat[idx]
        sub = counts[counts["condition"] == condition]
        n = int(sub["count"].sum())
        lookup = {str(row.resolution): int(row.count) for row in sub.itertuples()}
        values = [lookup.get(label, 0) for label in top_labels]
        other = n - sum(values)
        plot_labels = [label.replace("x", "×") for label in top_labels]
        plot_values = list(values)
        if other > 0:
            plot_labels.append("other")
            plot_values.append(other)

        y = np.arange(len(plot_labels))
        ax.barh(y, plot_values, color=CONDITION_COLORS[condition], height=0.72)
        ax.set_yticks(y, plot_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(
            f"{CONDITION_SHORT[condition]}  ({format_count(n)} files)\n[{CONDITION_KEYWORDS[condition]}]",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.25)
        if plot_values:
            xmax = max(plot_values)
            for i, val in enumerate(plot_values):
                if val <= 0:
                    continue
                pct = 100.0 * val / n if n else 0
                ax.text(
                    val + xmax * 0.01,
                    i,
                    f"{format_count(val)}\n{pct:.0f}%",
                    va="center",
                    fontsize=7,
                )

    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.05,
        0.6,
        "Assignment:\nseries description keywords\nfrom train_series_descriptions.csv\n\n"
        "Note: LFN/RFN share sagittal T1;\nLSS/RSS share axial T2\n(often similar resolution mixes).",
        fontsize=10,
        transform=axes_flat[-1].transAxes,
        va="top",
    )
    fig.suptitle(
        "DICOM resolutions by condition (series-matched slices)",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("Number of DICOM files")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "histogram_dicom_resolutions_by_condition")


def generate_dicom_resolution_figures(
    *,
    counts_path: Path = DEFAULT_COUNTS,
    output_dir: Path = DATASET_DESCRIPTION_FIGURES_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = load_resolution_counts(Path(counts_path))
    source = output_dir / "dicom_resolution_counts_by_condition.csv"
    counts.to_csv(source, index=False)
    png, pdf = plot_resolutions_by_condition(counts, output_dir)
    meta = save_metadata(
        output_dir,
        stem="histogram_dicom_resolutions_by_condition",
        extra={
            "script_name": "plot_dicom_resolutions.py",
            "source_counts": str(Path(counts_path).resolve()),
            "n_rows": int(len(counts)),
        },
    )
    print(
        "\nDICOM RESOLUTION FIGURE GENERATION COMPLETE\n\n"
        f"Counts:\n{counts_path}\n\n"
        "Generated:\n"
        "- Histogram of DICOM resolutions by condition\n\n"
        "Count labels use commas only for 5+ digit integers.\n"
    )
    return {"resolutions_png": png, "resolutions_pdf": pdf, "meta": meta, "source_counts": source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot DICOM resolutions by condition from inventory CSVs.")
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DESCRIPTION_FIGURES_DIR)
    args = parser.parse_args(argv)
    generate_dicom_resolution_figures(counts_path=args.counts, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
