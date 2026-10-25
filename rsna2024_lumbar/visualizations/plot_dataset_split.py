"""Actual vs target train/val/test split fractions by condition.

Reads inventory CSVs only. Count labels use thousands separators for 5+ digit
integers and leave four-digit values ungrouped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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

DEFAULT_SPLITS = YEAR_ROOT / "data" / "dataset_description" / "dataset_split_all_conditions.csv"
SPLIT_ORDER = ("train", "val", "test")
SPLIT_COLORS = {
    "train": "#2E86AB",
    "val": "#E8A838",
    "test": "#C0392B",
}
SPLIT_ROW_BG = {
    "train": "#d6ebf5",
    "val": "#fde8c8",
    "test": "#f5d0cc",
}
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_split_summaries(path: Path) -> list[dict[str, Any]]:
    raw = pd.read_csv(path)
    required = {"condition", "split", "studies_usable", "dicom_crops", "fraction_actual", "fraction_target"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(raw["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(raw['condition']))}")
    summaries: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        sub = raw[raw["condition"] == condition]
        rec: dict[str, Any] = {"condition_key": condition}
        for row in sub.itertuples(index=False):
            split_name = str(row.split)
            rec[f"studies_usable_{split_name}"] = int(row.studies_usable)
            rec[f"dicom_crops_{split_name}"] = int(row.dicom_crops)
            rec[f"{split_name}_fraction_actual"] = float(row.fraction_actual)
            rec[f"{split_name}_fraction_target"] = float(row.fraction_target)
        summaries.append(rec)
    return summaries


def _style_split_table(table, *, caption: str, ax_tbl) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e0e0")
        cell.set_linewidth(0.6)
        cell.get_text().set_color("#1a1a1a")
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        elif col == -1:
            split_name = SPLIT_ORDER[row - 1]
            cell.set_facecolor(SPLIT_ROW_BG[split_name])
            cell.set_edgecolor(SPLIT_COLORS[split_name])
            cell.set_linewidth(1.4)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
    ax_tbl.text(
        0.5,
        -0.06,
        caption,
        transform=ax_tbl.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )


def plot_split_fraction_check(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[Path, Path]:
    _style()
    n = len(summaries)
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.55], hspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")

    x = np.arange(n, dtype=float)
    width = 0.26
    targets = [train_fraction, val_fraction, test_fraction]
    for i, split_name in enumerate(SPLIT_ORDER):
        actuals = [s[f"{split_name}_fraction_actual"] for s in summaries]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            [100 * a for a in actuals],
            width=width,
            color=SPLIT_COLORS[split_name],
            label=f"{split_name} (actual)",
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.axhline(
            100 * targets[i],
            color=SPLIT_COLORS[split_name],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        for bar, actual in zip(bars, actuals):
            extra = 6.5 if split_name == "test" else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2 + extra,
                f"{100 * actual:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1a1a1a",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": SPLIT_COLORS[split_name],
                    "linewidth": 1.0,
                    "alpha": 0.96,
                },
            )

    ax.set_xticks(x, [CONDITION_SHORT[s["condition_key"]] for s in summaries], fontsize=11)
    ax.set_ylabel("Share of usable studies (%)")
    ax.set_ylim(0, 88)
    ax.set_title(
        f"Actual vs target split fractions "
        f"(dashed = target {train_fraction:.0%}/{val_fraction:.0%}/{test_fraction:.0%})",
        fontsize=12,
        pad=10,
    )
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_xlim(-0.55, n - 0.45)

    col_labels = [CONDITION_SHORT[s["condition_key"]] for s in summaries]
    cell_text: list[list[str]] = []
    for split_name in SPLIT_ORDER:
        row = []
        for s in summaries:
            crops = int(s[f"dicom_crops_{split_name}"])
            studies = int(s[f"studies_usable_{split_name}"])
            row.append(f"{format_count(crops)} crops\n{format_count(studies)} studies")
        cell_text.append(row)

    table = ax_tbl.table(
        cellText=cell_text,
        rowLabels=list(SPLIT_ORDER),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    _style_split_table(
        table,
        caption="Usable crops = training images (one cropped patch per lumbar level per study)",
        ax_tbl=ax_tbl,
    )
    return _save_figure(fig, output_dir, "dataset_split_fraction_check")


def generate_dataset_split_figures(
    *,
    splits_path: Path = DEFAULT_SPLITS,
    output_dir: Path = DATASET_DESCRIPTION_FIGURES_DIR,
    train_fraction: float = DEFAULT_FRACTIONS["train"],
    val_fraction: float = DEFAULT_FRACTIONS["val"],
    test_fraction: float = DEFAULT_FRACTIONS["test"],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_split_summaries(Path(splits_path))
    source = output_dir / "dataset_split_all_conditions.csv"
    pd.read_csv(splits_path).to_csv(source, index=False)
    png, pdf = plot_split_fraction_check(
        summaries,
        output_dir,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    meta = save_metadata(
        output_dir,
        stem="dataset_split_fraction_check",
        extra={
            "script_name": "plot_dataset_split.py",
            "source_splits": str(Path(splits_path).resolve()),
            "n_conditions": int(len(summaries)),
        },
    )
    print(
        "\nDATASET SPLIT FIGURE GENERATION COMPLETE\n\n"
        f"Splits:\n{splits_path}\n\n"
        "Generated:\n"
        "- Actual vs target split fractions\n\n"
        "Count labels use commas only for 5+ digit integers.\n"
    )
    return {"split_png": png, "split_pdf": pdf, "meta": meta, "source_splits": source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot actual vs target split fractions from inventory CSVs.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DESCRIPTION_FIGURES_DIR)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_FRACTIONS["train"])
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_FRACTIONS["val"])
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_FRACTIONS["test"])
    args = parser.parse_args(argv)
    generate_dataset_split_figures(
        splits_path=args.splits,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
