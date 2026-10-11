"""Publication figures for patient-level evaluation results.

Reads harvested CSVs only. No manuscript values are hard-coded into the plots.
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

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import PATIENT_LEVEL_FIGURES_DIR

ARCH_ORDER: tuple[str, ...] = ("ViT", "MaxViT", "ConvNeXt", "EfficientNet")
REP_ORDER: tuple[str, ...] = ("2D", "2.5D-to-3D")
PLOT_SPLIT = "validation"

R0_COLOR = "#16A34A"
R1_COLOR = "#FFD400"
R2_COLOR = "#EF4444"
COLOR_2D = "#FACC15"
COLOR_25D = "#38BDF8"
ARCH_COLORS: dict[str, str] = {
    "ViT": "#2563EB",
    "MaxViT": "#EA580C",
    "ConvNeXt": "#16A34A",
    "EfficientNet": "#DC2626",
}
REP_MARKERS: dict[str, str] = {"2D": "o", "2.5D-to-3D": "^"}

GUIDE_EXPECTED_REPEATS: dict[tuple[str, str], int] = {
    ("ViT", "2D"): 5,
    ("MaxViT", "2D"): 5,
    ("ConvNeXt", "2D"): 5,
    ("EfficientNet", "2D"): 5,
    ("ViT", "2.5D-to-3D"): 5,
    ("MaxViT", "2.5D-to-3D"): 5,
    ("ConvNeXt", "2.5D-to-3D"): 4,
    ("EfficientNet", "2.5D-to-3D"): 4,
}

SUMMARY_ALIASES: dict[str, tuple[str, ...]] = {
    "arch": ("arch", "architecture"),
    "rep": ("rep", "representation"),
    "split": ("split",),
    "n_repeats": ("n_repeats",),
    "exact_mean": ("exact_mean", "patient_macro_exact_rate_mean"),
    "exact_std": ("exact_std", "patient_macro_exact_rate_std"),
    "omae_mean": ("omae_mean", "patient_macro_omae_mean"),
    "omae_std": ("omae_std", "patient_macro_omae_std"),
    "r0_mean": ("r0_mean", "patient_exact_profile_rate_mean"),
    "r0_std": ("r0_std", "patient_exact_profile_rate_std"),
    "r1_mean": ("r1_mean", "patient_adjacent_only_rate_mean"),
    "r1_std": ("r1_std", "patient_adjacent_only_rate_std"),
    "r2_mean": ("r2_mean", "patient_severe_error_rate_mean"),
    "r2_std": ("r2_std", "patient_severe_error_rate_std"),
}

REPEAT_ALIASES: dict[str, tuple[str, ...]] = {
    "arch": ("arch", "architecture"),
    "rep": ("rep", "representation"),
    "split": ("split",),
    "repeat": ("repeat", "repeat_index"),
    "patient_macro_exact": ("patient_macro_exact", "patient_macro_exact_rate"),
    "patient_macro_omae": ("patient_macro_omae",),
    "exact_profile_rate": ("exact_profile_rate", "patient_exact_profile_rate"),
    "adjacent_only_rate": ("adjacent_only_rate", "patient_adjacent_only_rate"),
    "severe_profile_rate": ("severe_profile_rate", "patient_severe_error_rate"),
}

PLOT_SOURCE_COLUMNS: tuple[str, ...] = (
    "arch",
    "rep",
    "n_repeats",
    "exact_mean",
    "exact_std",
    "omae_mean",
    "omae_std",
    "r0_mean",
    "r0_std",
    "r1_mean",
    "r1_std",
    "r2_mean",
    "r2_std",
)

DEFAULT_SUMMARY = RESULTS_DIR / "protocol_patient_level" / "patient_level_summary.csv"
DEFAULT_REPEAT = RESULTS_DIR / "protocol_patient_level" / "patient_level_repeat_summary.csv"

FOOTER = (
    "Means and SDs are computed across available repeated runs. "
    "n is the number of completed repeats in the harvested CSV."
)


def _resolve_column(df: pd.DataFrame, names: tuple[str, ...], *, label: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"{label}: none of {names} found in columns {list(df.columns)}")


def _canonicalize(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    out = pd.DataFrame()
    for dest, names in aliases.items():
        src = _resolve_column(df, names, label=dest)
        out[dest] = df[src]
    return out


def load_data(
    summary_path: Path,
    repeat_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not repeat_path.is_file():
        raise FileNotFoundError(repeat_path)
    summary = _canonicalize(pd.read_csv(summary_path), SUMMARY_ALIASES)
    repeats = _canonicalize(pd.read_csv(repeat_path), REPEAT_ALIASES)
    summary["n_repeats"] = pd.to_numeric(summary["n_repeats"], errors="coerce")
    for col in (
        "exact_mean",
        "exact_std",
        "omae_mean",
        "omae_std",
        "r0_mean",
        "r0_std",
        "r1_mean",
        "r1_std",
        "r2_mean",
        "r2_std",
    ):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in (
        "patient_macro_exact",
        "patient_macro_omae",
        "exact_profile_rate",
        "adjacent_only_rate",
        "severe_profile_rate",
    ):
        repeats[col] = pd.to_numeric(repeats[col], errors="coerce")
    return summary, repeats


def validate_repeat_summary_consistency(
    summary: pd.DataFrame,
    repeats: pd.DataFrame,
    *,
    split: str = PLOT_SPLIT,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> pd.DataFrame:
    split_repeats = repeats[repeats["split"] == split].copy()
    if split_repeats.empty:
        raise ValueError(f"No repeat rows for split={split}")
    recomputed = (
        split_repeats.groupby(["arch", "rep"], sort=False)
        .agg(
            exact_mean=("patient_macro_exact", "mean"),
            exact_std=("patient_macro_exact", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            omae_mean=("patient_macro_omae", "mean"),
            omae_std=("patient_macro_omae", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            r0_mean=("exact_profile_rate", "mean"),
            r0_std=("exact_profile_rate", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            r1_mean=("adjacent_only_rate", "mean"),
            r1_std=("adjacent_only_rate", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            r2_mean=("severe_profile_rate", "mean"),
            r2_std=("severe_profile_rate", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )
    summary_split = summary[summary["split"] == split].copy()
    merged = recomputed.merge(
        summary_split,
        on=["arch", "rep"],
        suffixes=("_recomputed", "_file"),
        how="outer",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", ["arch", "rep", "_merge"]]
        raise ValueError(f"Summary/repeat architecture-representation mismatch:\n{missing}")

    checks = (
        "exact_mean",
        "exact_std",
        "omae_mean",
        "omae_std",
        "r0_mean",
        "r0_std",
        "r1_mean",
        "r1_std",
        "r2_mean",
        "r2_std",
        "n_repeats",
    )
    for col in checks:
        left = pd.to_numeric(merged[f"{col}_recomputed"], errors="coerce")
        right = pd.to_numeric(merged[f"{col}_file"], errors="coerce")
        if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            delta = (left - right).abs()
            raise ValueError(
                f"STOP plotting: {col} mismatch between repeat-level recompute and summary.\n"
                f"{merged[['arch', 'rep', f'{col}_recomputed', f'{col}_file']].to_string(index=False)}\n"
                f"max abs delta={float(delta.max())}"
            )
    return recomputed


def validate_profile_sums(summary: pd.DataFrame, *, split: str = PLOT_SPLIT) -> None:
    rows = summary[summary["split"] == split]
    for _, row in rows.iterrows():
        profile = float(row["r0_mean"]) + float(row["r1_mean"]) + float(row["r2_mean"])
        if not np.isclose(profile, 1.0, atol=1e-6):
            raise ValueError(
                f"STOP: R0+R1+R2 != 1 for {row['arch']} {row['rep']}: {profile}"
            )
        for key in ("r0_mean", "r1_mean", "r2_mean", "exact_mean"):
            value = float(row[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"STOP: {key} out of range for {row['arch']} {row['rep']}: {value}")
        if not 0.0 <= float(row["omae_mean"]) <= 2.0:
            raise ValueError(f"STOP: omae_mean out of range for {row['arch']} {row['rep']}")


def validate_plot_scope(summary: pd.DataFrame, *, split: str = PLOT_SPLIT) -> pd.DataFrame:
    validation = summary[summary["split"] == split].copy()
    if set(validation["arch"]) != set(ARCH_ORDER):
        raise ValueError(f"Unexpected architectures: {sorted(set(validation['arch']))}")
    if set(validation["rep"]) != set(REP_ORDER):
        raise ValueError(f"Unexpected representations: {sorted(set(validation['rep']))}")
    if len(validation) != 8:
        raise ValueError(f"Expected 8 architecture-representation rows, got {len(validation)}")
    return validation


def warn_repeat_counts(validation: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for _, row in validation.iterrows():
        key = (str(row["arch"]), str(row["rep"]))
        expected = GUIDE_EXPECTED_REPEATS.get(key)
        got = int(row["n_repeats"])
        if expected is not None and got != expected:
            msg = (
                f"{key[0]} {key[1]}: harvested n_repeats={got} "
                f"(guide expected {expected}). Using harvested count."
            )
            warnings.append(msg)
            print(f"WARNING: {msg}")
    return warnings


def save_plot_source_csv(validation: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "patient_level_plot_source_validation.csv"
    ordered = validation[list(PLOT_SOURCE_COLUMNS)].sort_values(["rep", "arch"])
    print(ordered.to_string(index=False))
    ordered.to_csv(path, index=False)
    return path


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def save_metadata(
    output_dir: Path,
    *,
    stem: str,
    summary_path: Path,
    repeat_path: Path,
    split: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "source_file": str(summary_path),
        "source_repeat_file": str(repeat_path),
        "split": split,
        "generated_from_real_results": True,
        "architectures": list(ARCH_ORDER),
        "representations": list(REP_ORDER),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_name": "plot_patient_level_results.py",
    }
    if extra:
        payload.update(extra)
    path = output_dir / f"{stem}_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


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


def _model_label(arch: str, rep: str, n_repeats: int) -> str:
    return f"{arch} ({rep}, n={int(n_repeats)})"


def plot_error_profile_distribution(validation: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True)
    panels = (("A.  2D models", "2D"), ("B.  2.5D-to-3D models", "2.5D-to-3D"))
    for ax, (title, rep) in zip(axes, panels):
        sub = validation[validation["rep"] == rep].set_index("arch").loc[list(ARCH_ORDER)]
        y = np.arange(len(ARCH_ORDER))
        left = np.zeros(len(ARCH_ORDER))
        series = (
            ("r0_mean", R0_COLOR),
            ("r1_mean", R1_COLOR),
            ("r2_mean", R2_COLOR),
        )
        for col, color in series:
            widths = 100.0 * sub[col].to_numpy(dtype=float)
            ax.barh(y, widths, left=left, color=color, edgecolor="white", height=0.62)
            for yi, width, start in zip(y, widths, left):
                label = f"{width:.2f}%"
                if width >= 6.5:
                    ax.text(
                        start + width / 2,
                        yi,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#111827",
                    )
                elif width >= 2.5:
                    ax.text(
                        start + width + 0.4,
                        yi,
                        label,
                        ha="left",
                        va="center",
                        fontsize=7,
                        color="#7F1D1D",
                    )
            left = left + widths
        ax.set_yticks(y)
        ax.set_yticklabels(list(ARCH_ORDER))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Patients (%)")
        ax.set_title(title, loc="left")
        ax.invert_yaxis()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=R0_COLOR, label="R0: all patient predictions exact"),
        plt.Rectangle((0, 0), 1, 1, color=R1_COLOR, label="R1: only adjacent-grade errors"),
        plt.Rectangle((0, 0), 1, 1, color=R2_COLOR, label="R2: at least one severe error"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Patient-Level Error Profiles (Validation Set)")
    fig.text(
        0.5,
        0.005,
        "R2 = patient contains at least one Normal/Mild ↔ Severe error. " + FOOTER,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    return _save_figure(fig, output_dir, "patient_error_profile_distribution")


def plot_exact_vs_omae_tradeoff(validation: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    x_min = float(validation["exact_mean"].min() - validation["exact_std"].max()) - 0.01
    x_max = float(validation["exact_mean"].max() + validation["exact_std"].max()) + 0.015
    y_min = float(validation["omae_mean"].min() - validation["omae_std"].max()) - 0.01
    y_max = float(validation["omae_mean"].max() + validation["omae_std"].max()) + 0.015
    shade_top = float(np.mean([y_min, y_max]))
    ax.axhspan(y_min, shade_top, xmin=0.55, color="#DCFCE7", alpha=0.45, zorder=0)
    ax.text(
        x_max - 0.002,
        shade_top - 0.004,
        "Ideal direction:\nhigher exactness, lower O-MAE",
        ha="right",
        va="top",
        fontsize=8,
        color="#166534",
        zorder=1,
    )
    offsets = {
        ("ViT", "2D"): (-78, 10),
        ("MaxViT", "2D"): (8, -22),
        ("ConvNeXt", "2D"): (10, 8),
        ("EfficientNet", "2D"): (10, 10),
        ("ViT", "2.5D-to-3D"): (10, 8),
        ("MaxViT", "2.5D-to-3D"): (10, -14),
        ("ConvNeXt", "2.5D-to-3D"): (10, 8),
        ("EfficientNet", "2.5D-to-3D"): (10, -12),
    }

    def _plot_row(row: pd.Series, *, zorder: int) -> None:
        arch = str(row["arch"])
        rep = str(row["rep"])
        ax.errorbar(
            row["exact_mean"],
            row["omae_mean"],
            xerr=row["exact_std"],
            yerr=row["omae_std"],
            fmt=REP_MARKERS[rep],
            color=ARCH_COLORS[arch],
            ecolor=ARCH_COLORS[arch],
            elinewidth=1.0,
            capsize=3,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=zorder,
        )
        ax.annotate(
            f"{arch} ({rep})",
            (row["exact_mean"], row["omae_mean"]),
            textcoords="offset points",
            xytext=offsets.get((arch, rep), (8, 8)),
            fontsize=8,
            zorder=zorder + 1,
        )

    on_top = (validation["arch"] == "MaxViT") & (validation["rep"] == "2D")
    for _, row in validation.loc[~on_top].iterrows():
        _plot_row(row, zorder=3)
    for _, row in validation.loc[on_top].iterrows():
        _plot_row(row, zorder=6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Patient Exact (higher is better)")
    ax.set_ylabel("Patient O-MAE (lower is better)")
    ax.set_title("Patient Exact vs Patient O-MAE Trade-Off (Validation Set)")
    arch_handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", label=arch)
        for arch, color in ARCH_COLORS.items()
    ]
    rep_handles = [
        plt.Line2D([0], [0], marker=marker, color="#111827", linestyle="", label=rep)
        for rep, marker in REP_MARKERS.items()
    ]
    legend1 = ax.legend(handles=arch_handles, title="Architecture", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=rep_handles, title="Representation", loc="upper right")
    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _save_figure(fig, output_dir, "patient_exact_vs_omae_tradeoff")


def plot_severe_error_burden(validation: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    plot_df = validation.sort_values("r2_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    y = np.arange(len(plot_df))
    colors = [COLOR_2D if rep == "2D" else COLOR_25D for rep in plot_df["rep"]]
    widths = 100.0 * plot_df["r2_mean"].to_numpy(dtype=float)
    xerr = 100.0 * plot_df["r2_std"].to_numpy(dtype=float)
    ax.barh(y, widths, color=colors, edgecolor="#111827", linewidth=0.6, height=0.68, xerr=xerr, capsize=3)
    labels = [
        _model_label(str(row["arch"]), str(row["rep"]), int(row["n_repeats"]))
        for _, row in plot_df.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Patients with at least one severe error (%)")
    ax.set_title("Severe-Error Patient Burden\nValidation set")
    xmax = float((widths + xerr).max()) + 3.5
    ax.set_xlim(0, xmax)
    for yi, width, err in zip(y, widths, xerr):
        ax.text(width + err + 0.35, yi, f"{width:.2f}%", va="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_2D, edgecolor="#111827", label="2D"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_25D, edgecolor="#111827", label="2.5D-to-3D"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _save_figure(fig, output_dir, "severe_error_patient_burden")


def generate_patient_level_figures(
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    repeat_path: Path = DEFAULT_REPEAT,
    output_dir: Path = PATIENT_LEVEL_FIGURES_DIR,
    split: str = PLOT_SPLIT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, repeats = load_data(Path(summary_path), Path(repeat_path))
    validate_repeat_summary_consistency(summary, repeats, split=split)
    validate_profile_sums(summary, split=split)
    validation = validate_plot_scope(summary, split=split)
    count_warnings = warn_repeat_counts(validation)
    source_csv = save_plot_source_csv(validation, output_dir)

    profile_png, profile_pdf = plot_error_profile_distribution(validation, output_dir)
    tradeoff_png, tradeoff_pdf = plot_exact_vs_omae_tradeoff(validation, output_dir)
    burden_png, burden_pdf = plot_severe_error_burden(validation, output_dir)

    extra = {
        "n_configurations": int(len(validation)),
        "run_counts": {
            f"{row.arch} {row.rep}": int(row.n_repeats) for row in validation.itertuples(index=False)
        },
        "repeat_count_warnings": count_warnings,
    }
    meta_profile = save_metadata(
        output_dir,
        stem="patient_error_profile_distribution",
        summary_path=Path(summary_path),
        repeat_path=Path(repeat_path),
        split=split,
        extra=extra,
    )
    meta_tradeoff = save_metadata(
        output_dir,
        stem="patient_exact_vs_omae_tradeoff",
        summary_path=Path(summary_path),
        repeat_path=Path(repeat_path),
        split=split,
        extra=extra,
    )
    meta_burden = save_metadata(
        output_dir,
        stem="severe_error_patient_burden",
        summary_path=Path(summary_path),
        repeat_path=Path(repeat_path),
        split=split,
        extra=extra,
    )

    counts = {
        f"{row.arch} {row.rep}": int(row.n_repeats) for row in validation.itertuples(index=False)
    }
    print(
        "\nPATIENT-LEVEL FIGURE GENERATION COMPLETE\n\n"
        f"Source summary:\n{summary_path}\n\n"
        f"Source repeated runs:\n{repeat_path}\n\n"
        f"Split:\n{split}\n\n"
        "Configurations:\n8\n\n"
        "Run completeness:\n"
        f"ViT 2D: {counts.get('ViT 2D', 0)}\n"
        f"MaxViT 2D: {counts.get('MaxViT 2D', 0)}\n"
        f"ConvNeXt 2D: {counts.get('ConvNeXt 2D', 0)}\n"
        f"EfficientNet 2D: {counts.get('EfficientNet 2D', 0)}\n"
        f"ViT 2.5D-to-3D: {counts.get('ViT 2.5D-to-3D', 0)}\n"
        f"MaxViT 2.5D-to-3D: {counts.get('MaxViT 2.5D-to-3D', 0)}\n"
        f"ConvNeXt 2.5D-to-3D: {counts.get('ConvNeXt 2.5D-to-3D', 0)}\n"
        f"EfficientNet 2.5D-to-3D: {counts.get('EfficientNet 2.5D-to-3D', 0)}\n\n"
        "Generated:\n"
        "- Patient Error-Profile Distribution\n"
        "- Patient Exactness–Ordinal Error Trade-Off\n"
        "- Severe-Error Patient Burden\n\n"
        "All plotted values were read from computed experimental result files.\n"
        "No manuscript values or manually entered metrics were used.\n"
    )
    return {
        "plot_source": source_csv,
        "profile_png": profile_png,
        "profile_pdf": profile_pdf,
        "profile_meta": meta_profile,
        "tradeoff_png": tradeoff_png,
        "tradeoff_pdf": tradeoff_pdf,
        "tradeoff_meta": meta_tradeoff,
        "burden_png": burden_png,
        "burden_pdf": burden_pdf,
        "burden_meta": meta_burden,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot patient-level evaluation figures from harvested CSVs.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--repeat-summary", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--output-dir", type=Path, default=PATIENT_LEVEL_FIGURES_DIR)
    parser.add_argument("--split", default=PLOT_SPLIT)
    args = parser.parse_args(argv)
    generate_patient_level_figures(
        summary_path=args.summary,
        repeat_path=args.repeat_summary,
        output_dir=args.output_dir,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
