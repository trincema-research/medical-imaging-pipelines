"""Stacked crops by within-condition DICOM reuse, plus cross-condition sharing.

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

MAX_SHOWN_MULT = 5
DEFAULT_DUPLICATION = YEAR_ROOT / "data" / "dataset_description" / "dicom_duplication_by_condition.csv"
DEFAULT_SUMMARY = YEAR_ROOT / "data" / "dataset_description" / "dicom_cross_condition_summary.csv"
DEFAULT_PAIRWISE = YEAR_ROOT / "data" / "dataset_description" / "dicom_cross_condition_pairwise.csv"

MULT_COLORS = {
    "1": "#4C9F70",
    "2": "#F1C40F",
    "3": "#E67E22",
    "4": "#E74C3C",
    f"{MAX_SHOWN_MULT}+": "#8E44AD",
    str(MAX_SHOWN_MULT): "#8E44AD",
}
# Equal columns would be ~0.33 each; Value is ~60% narrower so Meaning can wrap.
MATH_COL_WIDTHS = (0.42, 0.13, 0.45)


def _bin_multiplicity(mult: dict[int, int]) -> tuple[list[str], list[int]]:
    labels = [str(i) for i in range(1, MAX_SHOWN_MULT)] + [f"{MAX_SHOWN_MULT}+"]
    values = [0] * len(labels)
    for m, count in mult.items():
        if m < MAX_SHOWN_MULT:
            values[m - 1] += count
        else:
            values[-1] += count
    if all(m <= MAX_SHOWN_MULT for m in mult) and MAX_SHOWN_MULT in mult:
        labels[-1] = str(MAX_SHOWN_MULT)
    return labels, values


def load_duplication(path: Path) -> dict[str, dict[str, Any]]:
    raw = pd.read_csv(path)
    required = {"condition", "multiplicity", "n_unique_dicoms", "n_coord_rows_from_these"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(raw["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(raw['condition']))}")
    results: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_ORDER:
        sub = raw[raw["condition"] == condition]
        crop_by_mult = {int(row.multiplicity): int(row.n_coord_rows_from_these) for row in sub.itertuples()}
        unique_by_mult = {int(row.multiplicity): int(row.n_unique_dicoms) for row in sub.itertuples()}
        results[condition] = {
            "n_coord_rows": int(sum(crop_by_mult.values())),
            "n_unique_dicoms": int(sum(unique_by_mult.values())),
            "crop_counts_by_multiplicity": {str(k): v for k, v in crop_by_mult.items()},
            "multiplicity_counts": {str(k): v for k, v in unique_by_mult.items()},
        }
    return results


def load_overlap(summary_path: Path, pairwise_path: Path) -> dict[str, Any]:
    summary_raw = pd.read_csv(summary_path)
    if not {"metric", "value"}.issubset(summary_raw.columns):
        raise ValueError(f"{summary_path} must have columns metric,value")
    summary = {str(row.metric): int(row.value) for row in summary_raw.itertuples(index=False)}
    pairwise = {key.removeprefix("pairwise_"): val for key, val in summary.items() if key.startswith("pairwise_")}
    hist = {}
    prefix, suffix = "dicoms_in_", "_conditions"
    for key, val in summary.items():
        if key.startswith(prefix) and key.endswith(suffix):
            hist[key[len(prefix) : -len(suffix)]] = val
    per = {}
    for condition, short in CONDITION_SHORT.items():
        per[condition] = {
            "condition_short": short,
            "n_unique_dicoms": summary[f"{short}_unique"],
            "n_coord_rows": summary[f"{short}_coord_rows"],
        }
    pairwise_raw = pd.read_csv(pairwise_path)
    if "condition" not in pairwise_raw.columns:
        raise ValueError(f"{pairwise_path} missing condition column")
    return {
        "per_condition": per,
        "pairwise_overlap": pairwise,
        "n_conditions_hist": hist,
        "global_unique_dicoms": summary["global_unique_dicoms"],
        "sum_per_condition_uniques": summary["sum_per_condition_uniques"],
        "double_counted_extras": summary["double_counted_extras"],
        "check_sum_minus_extras_equals_global": (
            summary["sum_per_condition_uniques"] - summary["double_counted_extras"] == summary["global_unique_dicoms"]
        ),
        "pairwise_table": pairwise_raw,
    }


def reconciliation_table_rows(overlap: dict[str, Any]) -> list[list[str]]:
    pairwise = overlap["pairwise_overlap"]
    return [
        [
            "Sum of per-condition unique DICOMs",
            format_count(overlap["sum_per_condition_uniques"]),
            "adds SCS+LFN+RFN+LSS+RSS uniques\n(double-counts shared files)",
        ],
        [
            "− extras from multi-condition DICOMs",
            format_count(overlap["double_counted_extras"]),
            "each file in k conditions\ncontributes (k−1) extras",
        ],
        [
            "= Global unique coord-referenced DICOMs",
            format_count(overlap["global_unique_dicoms"]),
            "matches check" if overlap["check_sum_minus_extras_equals_global"] else "MISMATCH",
        ],
        [
            "Largest cross-condition overlap",
            f"LSS&RSS =\n{format_count(pairwise.get('LSS&RSS', 0))}",
            f"LFN&RFN = {format_count(pairwise.get('LFN&RFN', 0))}\nSCS shares 0 with others",
        ],
    ]


def _style_reconciliation_table(table) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.get_text().set_color("#1a1a1a")
        if 0 <= col < len(MATH_COL_WIDTHS):
            cell.set_width(MATH_COL_WIDTHS[col])
        if col == 1:
            cell.get_text().set_ha("center")
        else:
            cell.get_text().set_ha("left")
        cell.get_text().set_va("center")
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        elif row == 3:
            cell.set_facecolor("#eafaf1")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")


def plot_crops_from_duplication(
    results: dict[str, dict[str, Any]],
    overlap: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    _style()
    conditions = list(CONDITION_ORDER)
    mult_labels = [str(i) for i in range(1, MAX_SHOWN_MULT)] + [f"{MAX_SHOWN_MULT}+"]
    fig = plt.figure(figsize=(13.5, 12.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 1.4, 1.9], hspace=0.48, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax_pair = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_math = fig.add_subplot(gs[2, :])
    ax_pair.axis("off")
    ax_hist.axis("off")
    ax_math.axis("off")

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    binned: list[dict[str, int]] = []
    display_labels = list(mult_labels)
    for condition in conditions:
        crops = {int(k): v for k, v in results[condition]["crop_counts_by_multiplicity"].items()}
        labels, values = _bin_multiplicity(crops)
        display_labels = labels
        binned.append(dict(zip(labels, values)))

    ymax = max(float(results[c]["n_coord_rows"]) for c in conditions) or 1.0
    external_labels: list[tuple[int, float, str, str]] = []
    for label in display_labels:
        vals = np.array([b.get(label, 0) for b in binned], dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=f"{label}× within condition",
            color=MULT_COLORS.get(label, "#7F8C8D"),
            edgecolor="white",
            width=0.7,
        )
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v <= 0:
                continue
            total = float(results[conditions[i]]["n_coord_rows"] or 1)
            pct = 100.0 * v / total
            text_inline = f"{format_count(v)} ({pct:.0f}%)"
            text_stacked = f"{format_count(v)}\n{pct:.0f}%"
            if v / ymax < 0.09:
                external_labels.append((i, b + v / 2, text_inline, MULT_COLORS.get(label, "#7F8C8D")))
            else:
                ax.text(
                    i,
                    b + v / 2,
                    text_stacked,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if label not in ("1",) else "#1a1a1a",
                )
        bottoms += vals

    by_cond: dict[int, list[tuple[float, str, str]]] = {}
    for i, y, text, color in external_labels:
        by_cond.setdefault(i, []).append((y, text, color))
    for i, items in by_cond.items():
        items.sort(key=lambda t: t[0])
        if len(items) >= 2:
            span = max(items[-1][0] - items[0][0], ymax * 0.12)
            step = span / max(len(items) - 1, 1)
            base = items[0][0] - span * 0.15
            ys = [base + k * step for k in range(len(items))]
        else:
            ys = [items[0][0]]
        for (y_seg, text, color), y_lab in zip(items, ys):
            ax.annotate(
                text,
                xy=(i - 0.35, y_seg),
                xytext=(i - 0.62, y_lab),
                ha="right",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1a1a1a",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 1.1,
                    "alpha": 0.96,
                },
            )

    ax.set_ylim(0, ymax * 1.04)
    ax.set_xlim(-1.15, len(conditions) - 0.35)
    short_labels = []
    for c in conditions:
        u = results[c]["n_unique_dicoms"]
        n = results[c]["n_coord_rows"]
        short_labels.append(f"{CONDITION_SHORT[c]}\n{format_count(u)} unique\n{format_count(n)} coords")
    ax.set_xticks(x, short_labels, fontsize=8.5)
    ax.set_ylabel("Coordinate rows (≈ crops)", fontsize=11)
    ax.set_title(
        "Within-condition DICOM reuse (stacked coord rows)\n"
        "This panel does NOT show sharing across conditions — see tables below",
        fontsize=12,
    )
    ax.legend(title="Source DICOM reused", loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    shorts = [CONDITION_SHORT[c] for c in conditions]
    matrix = np.zeros((len(conditions), len(conditions)), dtype=int)
    pairwise = overlap["pairwise_overlap"]
    per = overlap["per_condition"]
    for i, a in enumerate(shorts):
        for j, b in enumerate(shorts):
            if i == j:
                matrix[i, j] = int(per[conditions[i]]["n_unique_dicoms"])
            else:
                key = f"{a}&{b}" if f"{a}&{b}" in pairwise else f"{b}&{a}"
                matrix[i, j] = int(pairwise.get(key, 0))

    cell_text = [[format_count(matrix[i, j]) for j in range(len(conditions))] for i in range(len(conditions))]
    table = ax_pair.table(cellText=cell_text, rowLabels=shorts, colLabels=shorts, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.get_text().set_color("#1a1a1a")
        if r == 0 or c == -1:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        elif r == c:
            cell.set_facecolor("#e8f4fc")
        else:
            val = matrix[r - 1, c]
            if val > 0:
                cell.set_facecolor("#fdebd0")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white")
    ax_pair.set_title(
        "Unique DICOMs shared across conditions\n(diagonal = unique within that condition)",
        fontsize=10,
        pad=8,
    )

    hist = overlap["n_conditions_hist"]
    hist_rows = [[f"{k} condition(s)", format_count(v)] for k, v in hist.items()]
    hist_rows.append(["Global unique DICOMs", format_count(overlap["global_unique_dicoms"])])
    hist_table = ax_hist.table(
        cellText=hist_rows,
        colLabels=["DICOM appears in", "Count"],
        loc="center",
        cellLoc="center",
    )
    hist_table.auto_set_font_size(False)
    hist_table.set_fontsize(9)
    hist_table.scale(1.1, 1.85)
    for (r, c), cell in hist_table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.get_text().set_color("#1a1a1a")
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        elif r == len(hist_rows):
            cell.set_facecolor("#eafaf1")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
    ax_hist.set_title("How many conditions share each DICOM file", fontsize=10, pad=8)

    math_table = ax_math.table(
        cellText=reconciliation_table_rows(overlap),
        colLabels=["Reconciliation", "Value", "Meaning"],
        loc="center",
        cellLoc="left",
    )
    _style_reconciliation_table(math_table)
    ax_math.set_title(
        "Why per-condition uniques do not add up to the global unique count",
        fontsize=10,
        pad=18,
    )
    fig.suptitle(
        "DICOM reuse: within-condition multiplicity + across-condition sharing",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    return _save_figure(fig, output_dir, "stacked_crops_by_dicom_reuse")


def generate_dicom_reuse_figures(
    *,
    duplication_path: Path = DEFAULT_DUPLICATION,
    summary_path: Path = DEFAULT_SUMMARY,
    pairwise_path: Path = DEFAULT_PAIRWISE,
    output_dir: Path = DATASET_DESCRIPTION_FIGURES_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_duplication(Path(duplication_path))
    overlap = load_overlap(Path(summary_path), Path(pairwise_path))
    source_dup = output_dir / "dicom_duplication_by_condition.csv"
    source_sum = output_dir / "dicom_cross_condition_summary.csv"
    source_pair = output_dir / "dicom_cross_condition_pairwise.csv"
    pd.read_csv(duplication_path).to_csv(source_dup, index=False)
    pd.read_csv(summary_path).to_csv(source_sum, index=False)
    pd.read_csv(pairwise_path).to_csv(source_pair, index=False)
    png, pdf = plot_crops_from_duplication(results, overlap, output_dir)
    meta = save_metadata(
        output_dir,
        stem="stacked_crops_by_dicom_reuse",
        extra={
            "script_name": "plot_dicom_reuse.py",
            "source_duplication": str(Path(duplication_path).resolve()),
            "source_summary": str(Path(summary_path).resolve()),
            "source_pairwise": str(Path(pairwise_path).resolve()),
        },
    )
    print(
        "\nDICOM REUSE FIGURE GENERATION COMPLETE\n\n"
        f"Duplication:\n{duplication_path}\n"
        f"Summary:\n{summary_path}\n\n"
        "Generated:\n"
        "- Stacked crops by DICOM reuse\n\n"
        "Count labels use commas only for 5+ digit integers.\n"
    )
    return {
        "reuse_png": png,
        "reuse_pdf": pdf,
        "meta": meta,
        "source_duplication": source_dup,
        "source_summary": source_sum,
        "source_pairwise": source_pair,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot stacked crops by DICOM reuse from inventory CSVs.")
    parser.add_argument("--duplication", type=Path, default=DEFAULT_DUPLICATION)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DESCRIPTION_FIGURES_DIR)
    args = parser.parse_args(argv)
    generate_dicom_reuse_figures(
        duplication_path=args.duplication,
        summary_path=args.summary,
        pairwise_path=args.pairwise,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
