"""Dataset scale and condition × severity mix figure.

Reads inventory CSVs only. Count labels use thousands separators for 5+ digit
integers and leave four-digit values ungrouped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsna2024_lumbar.visualizations import DATASET_DESCRIPTION_FIGURES_DIR, YEAR_ROOT

CONDITION_ORDER: tuple[str, ...] = (
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
)
CONDITION_SHORT: dict[str, str] = {
    "spinal_canal_stenosis": "SCS",
    "left_neural_foraminal_narrowing": "LFN",
    "right_neural_foraminal_narrowing": "RFN",
    "left_subarticular_stenosis": "LSS",
    "right_subarticular_stenosis": "RSS",
}
SEVERITY_CLASSES: tuple[str, ...] = ("Normal/Mild", "Moderate", "Severe")
SEVERITY_COLORS: dict[str, str] = {
    "Normal/Mild": "#4C9F70",
    "Moderate": "#E8A838",
    "Severe": "#C0392B",
}

DEFAULT_SCALE = YEAR_ROOT / "data" / "dataset_description" / "dataset_scale.csv"
DEFAULT_COUNTS = YEAR_ROOT / "data" / "dataset_description" / "counts_condition_x_severity.csv"
SCALE_KEYS: tuple[str, ...] = (
    "raw_dicom_files",
    "unique_dicom_files_total",
    "usable_dicom_crops_total",
    "label_cells_total",
)


def format_count(value: int | float) -> str:
    """Group thousands only for 5+ digit integers (10,000+). Four-digit values stay ungrouped."""
    number = int(round(float(value)))
    if abs(number) >= 10_000:
        return f"{number:,}"
    return str(number)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def save_metadata(output_dir: Path, *, stem: str, extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "generated_from_real_results": True,
        "script_name": "plot_dataset_scale.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "count_grouping": "thousands separators for 5+ digit integers only",
    }
    if extra:
        payload.update(extra)
    path = output_dir / f"{stem}_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_scale(path: Path) -> dict[str, int]:
    raw = pd.read_csv(path)
    if not {"metric", "count"}.issubset(raw.columns):
        raise ValueError(f"{path} must have columns metric,count")
    scale = {str(row.metric): int(row.count) for row in raw.itertuples(index=False)}
    missing = [key for key in SCALE_KEYS if key not in scale]
    if missing:
        raise ValueError(f"{path} missing metrics: {missing}")
    return scale


def load_condition_severity(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"condition", *SEVERITY_CLASSES}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(raw["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(raw['condition']))}")
    return raw


def condition_severity_matrix(counts_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(CONDITION_ORDER), len(SEVERITY_CLASSES)), dtype=float)
    for i, condition in enumerate(CONDITION_ORDER):
        row = counts_df[counts_df["condition"] == condition]
        if row.empty:
            continue
        for j, severity in enumerate(SEVERITY_CLASSES):
            counts[i, j] = float(row[severity].iloc[0])
    row_totals = counts.sum(axis=1, keepdims=True)
    row_pct = np.divide(counts, row_totals, where=row_totals > 0, out=np.zeros_like(counts)) * 100.0
    return counts, row_pct


def _plot_file_hierarchy_ax(ax: plt.Axes, scale: dict[str, int]) -> None:
    bar_labels = ["Raw DICOMs", "Unique coord DICOMs", "Usable crops", "Label cells"]
    bar_vals = [scale[key] for key in SCALE_KEYS]
    ax.bar(bar_labels, bar_vals, color=["#34495E", "#2980B9", "#27AE60", "#8E44AD"])
    ax.set_ylabel("Count")
    ax.set_title("From ~147k files to training labels")
    ax.tick_params(axis="x", rotation=18)
    for i, val in enumerate(bar_vals):
        ax.text(
            i,
            val + max(bar_vals) * 0.01,
            format_count(val),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.grid(axis="y", alpha=0.25)


def _plot_severity_mix_ax(ax: plt.Axes, counts: np.ndarray) -> None:
    x = np.arange(len(CONDITION_ORDER))
    bar_width = 0.36
    offsets = (-0.18, 0.0, 0.18)
    for j, severity in enumerate(SEVERITY_CLASSES):
        bars = ax.bar(
            x + offsets[j],
            counts[:, j],
            bar_width,
            label=severity,
            color=SEVERITY_COLORS[severity],
        )
        for bar, count in zip(bars, counts[:, j]):
            height = bar.get_height()
            if height <= 0:
                continue
            text_color = "white" if j > 0 else "#1a1a1a"
            if height >= 800:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    format_count(count),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(counts.max(), 1) * 0.02,
                    format_count(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#1a1a1a",
                )
    ax.set_xticks(x, [CONDITION_SHORT[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Usable DICOM crops")
    ax.set_ylim(0, counts.max() * 1.12 if counts.size else 1)
    ax.set_title("Severity mix per condition\n(5 lumbar levels pooled)")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)


def plot_scale_and_severity_combo(
    counts_df: pd.DataFrame,
    scale: dict[str, int],
    output_dir: Path,
) -> tuple[Path, Path]:
    _style()
    counts, _ = condition_severity_matrix(counts_df)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    _plot_file_hierarchy_ax(ax_left, scale)
    _plot_severity_mix_ax(ax_right, counts)
    fig.suptitle(
        "RSNA 2024 lumbar spine — dataset scale & severity mix",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "dataset_scale_and_severity_combo")


def generate_dataset_scale_figures(
    *,
    scale_path: Path = DEFAULT_SCALE,
    counts_path: Path = DEFAULT_COUNTS,
    output_dir: Path = DATASET_DESCRIPTION_FIGURES_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = load_scale(Path(scale_path))
    counts_df = load_condition_severity(Path(counts_path))
    source_scale = output_dir / "dataset_scale.csv"
    source_counts = output_dir / "counts_condition_x_severity.csv"
    pd.DataFrame({"metric": list(scale), "count": [scale[k] for k in scale]}).to_csv(source_scale, index=False)
    counts_df.to_csv(source_counts, index=False)
    png, pdf = plot_scale_and_severity_combo(counts_df, scale, output_dir)
    meta = save_metadata(
        output_dir,
        stem="dataset_scale_and_severity_combo",
        extra={
            "source_scale": str(Path(scale_path).resolve()),
            "source_counts": str(Path(counts_path).resolve()),
            "scale": scale,
        },
    )
    print(
        "\nDATASET SCALE FIGURE GENERATION COMPLETE\n\n"
        f"Scale:\n{scale_path}\n"
        f"Counts:\n{counts_path}\n\n"
        "Generated:\n"
        "- Dataset scale & severity mix combo\n\n"
        "Count labels use commas only for 5+ digit integers.\n"
    )
    return {
        "combo_png": png,
        "combo_pdf": pdf,
        "meta": meta,
        "source_scale": source_scale,
        "source_counts": source_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot dataset scale and severity mix from inventory CSVs.")
    parser.add_argument("--scale", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DESCRIPTION_FIGURES_DIR)
    args = parser.parse_args(argv)
    generate_dataset_scale_figures(scale_path=args.scale, counts_path=args.counts, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
