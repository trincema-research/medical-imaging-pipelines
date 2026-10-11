"""Publication figures for spinal-level evaluation results.

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
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import SPINAL_LEVEL_FIGURES_DIR

ARCH_ORDER: tuple[str, ...] = ("ViT", "MaxViT", "ConvNeXt", "EfficientNet")
REP_ORDER: tuple[str, ...] = ("2D", "2.5D-to-3D")
LEVEL_ORDER: tuple[str, ...] = ("L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1")
LEVEL_LABELS: dict[str, str] = {
    "L1_L2": "L1/L2",
    "L2_L3": "L2/L3",
    "L3_L4": "L3/L4",
    "L4_L5": "L4/L5",
    "L5_S1": "L5/S1",
}
PLOT_SPLIT = "validation"
ARCH_COLORS: dict[str, str] = {
    "ViT": "#1f77b4",
    "MaxViT": "#ff7f0e",
    "ConvNeXt": "#2ca02c",
    "EfficientNet": "#d62728",
}
WORST_COLOR = "#EF4444"
BEST_COLOR = "#16A34A"
# Cream → orange → maroon, matching the manuscript Macro-F1 heatmap palette.
MACRO_F1_CMAP = LinearSegmentedColormap.from_list(
    "macro_f1_oranges",
    (
        "#FFF7EC",
        "#FEE8C8",
        "#FDD49E",
        "#FDBB84",
        "#FC8D59",
        "#EF6548",
        "#D7301F",
        "#B30000",
        "#7F0000",
    ),
)

DEFAULT_MEANS = RESULTS_DIR / "protocol_spinal_level" / "spinal_level_means.csv"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "arch",
        "rep",
        "level",
        "f1_mean",
        "f1_std",
        "qwk_mean",
        "qwk_std",
        "omae_mean",
        "omae_std",
        "ser_mean",
        "ser_std",
    }
)

METRIC_ALIASES: dict[str, tuple[str, str]] = {
    "acc": ("accuracy_mean", "accuracy_std"),
    "precision": ("precision_macro_mean", "precision_macro_std"),
    "recall": ("recall_macro_mean", "recall_macro_std"),
    "f1": ("f1_macro_mean", "f1_macro_std"),
    "oa": ("oa_mean", "oa_std"),
    "omae": ("omae_mean", "omae_std"),
    "qwk": ("qwk_mean", "qwk_std"),
    "ser": ("ser_mean", "ser_std"),
}


def canonical_level_key(raw: str) -> str:
    text = str(raw or "").strip().upper().replace("/", "_").replace("-", "_").replace(" ", "_")
    if text in LEVEL_ORDER:
        return text
    raise ValueError(f"Unknown spinal level: {raw!r}")


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


def save_metadata(
    output_dir: Path,
    *,
    stem: str,
    source_file: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "source_file": str(source_file),
        "generated_from_real_results": True,
        "split": PLOT_SPLIT,
        "architectures": list(ARCH_ORDER),
        "representations": list(REP_ORDER),
        "spinal_levels": [LEVEL_LABELS[k] for k in LEVEL_ORDER],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_name": "plot_spinal_level_results.py",
    }
    if extra:
        payload.update(extra)
    path = output_dir / f"{stem}_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _resolve_col(df: pd.DataFrame, names: tuple[str, ...], *, label: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"{label}: none of {names} found in {list(df.columns)}")


def load_and_validate_data(means_path: Path, *, split: str = PLOT_SPLIT) -> pd.DataFrame:
    if not means_path.is_file():
        raise FileNotFoundError(means_path)
    raw = pd.read_csv(means_path)
    arch_col = _resolve_col(raw, ("arch", "architecture"), label="arch")
    rep_col = _resolve_col(raw, ("rep", "representation"), label="rep")
    level_col = _resolve_col(raw, ("level", "spinal_level"), label="level")
    work = raw.copy()
    if "split" in work.columns:
        work = work[work["split"] == split].copy()
    if work.empty:
        raise ValueError(f"No rows for split={split} in {means_path}")
    work["arch"] = work[arch_col]
    work["rep"] = work[rep_col]
    work["level"] = work[level_col].map(canonical_level_key)

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["arch", "rep", "level"], dropna=False, sort=False)
    for (arch, rep, level), group in grouped:
        rec: dict[str, Any] = {"arch": arch, "rep": rep, "level": level}
        for dest, (mean_col, std_col) in METRIC_ALIASES.items():
            if mean_col not in group.columns:
                continue
            vals = pd.to_numeric(group[mean_col], errors="coerce")
            rec[f"{dest}_mean"] = float(vals.mean())
            if std_col in group.columns:
                stds = pd.to_numeric(group[std_col], errors="coerce")
                rec[f"{dest}_std"] = float(stds.mean()) if stds.notna().any() else float("nan")
            else:
                rec[f"{dest}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(rec)
    df = pd.DataFrame(rows)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("NaNs in required spinal-level metrics")
    if set(df["arch"]) != set(ARCH_ORDER):
        raise ValueError(f"Unexpected architectures: {sorted(set(df['arch']))}")
    if set(df["rep"]) != set(REP_ORDER):
        raise ValueError(f"Unexpected representations: {sorted(set(df['rep']))}")
    if set(df["level"]) != set(LEVEL_ORDER):
        raise ValueError(f"Unexpected spinal levels: {sorted(set(df['level']))}")
    if len(df) != 40:
        raise ValueError(f"Expected 40 architecture-representation-level rows, got {len(df)}")
    for _, row in df.iterrows():
        if not 0.0 <= float(row["f1_mean"]) <= 1.0:
            raise ValueError(f"f1_mean out of range: {row['arch']} {row['rep']} {row['level']}")
        if not 0.0 <= float(row["omae_mean"]) <= 2.0:
            raise ValueError(f"omae_mean out of range: {row['arch']} {row['rep']} {row['level']}")
        if not 0.0 <= float(row["ser_mean"]) <= 1.0:
            raise ValueError(f"ser_mean out of range: {row['arch']} {row['rep']} {row['level']}")
        if not -1.0 <= float(row["qwk_mean"]) <= 1.0:
            raise ValueError(f"qwk_mean out of range: {row['arch']} {row['rep']} {row['level']}")
    level_rank = {k: i for i, k in enumerate(LEVEL_ORDER)}
    arch_rank = {k: i for i, k in enumerate(ARCH_ORDER)}
    rep_rank = {k: i for i, k in enumerate(REP_ORDER)}
    df["_a"] = df["arch"].map(arch_rank)
    df["_r"] = df["rep"].map(rep_rank)
    df["_l"] = df["level"].map(level_rank)
    return df.sort_values(["_r", "_a", "_l"], kind="mergesort").drop(columns=["_a", "_r", "_l"]).reset_index(drop=True)


def save_plot_source_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "spinal_level_plot_source.csv"
    df.to_csv(path, index=False)
    return path


def compute_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arch in ARCH_ORDER:
        for rep in REP_ORDER:
            sub = df[(df["arch"] == arch) & (df["rep"] == rep)]
            worst = sub.loc[sub["f1_mean"].idxmin()]
            best = sub.loc[sub["f1_mean"].idxmax()]
            rows.append(
                {
                    "arch": arch,
                    "rep": rep,
                    "worst_level": worst["level"],
                    "worst_f1": float(worst["f1_mean"]),
                    "best_level": best["level"],
                    "best_f1": float(best["f1_mean"]),
                    "spread": float(best["f1_mean"] - worst["f1_mean"]),
                }
            )
    return pd.DataFrame(rows)


def compute_representation_level_means(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["rep", "level"], as_index=False)[["f1_mean", "qwk_mean", "omae_mean", "ser_mean"]]
        .mean()
    )
    if len(agg) != 10:
        raise ValueError(f"Expected 10 representation-level rows, got {len(agg)}")
    return agg


def plot_best_worst_bottleneck(bottlenecks: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True)
    for ax, rep in zip(axes, REP_ORDER):
        sub = bottlenecks[bottlenecks["rep"] == rep].set_index("arch").loc[list(ARCH_ORDER)]
        y = np.arange(len(ARCH_ORDER))
        ax.hlines(y, sub["worst_f1"], sub["best_f1"], color="#9CA3AF", linewidth=2.0, zorder=1)
        ax.scatter(sub["worst_f1"], y, color=WORST_COLOR, s=46, zorder=2, label="Worst")
        ax.scatter(sub["best_f1"], y, color=BEST_COLOR, s=46, zorder=2, label="Best")
        for yi, row in enumerate(sub.itertuples(index=False)):
            ax.text(
                row.worst_f1 - 0.008,
                yi,
                f"{LEVEL_LABELS[row.worst_level]}, {row.worst_f1:.3f}",
                ha="right",
                va="center",
                fontsize=7.5,
                color=WORST_COLOR,
            )
            ax.text(
                row.best_f1 + 0.008,
                yi,
                f"{LEVEL_LABELS[row.best_level]}, {row.best_f1:.3f}",
                ha="left",
                va="center",
                fontsize=7.5,
                color=BEST_COLOR,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([f"{arch} ({rep})" for arch in ARCH_ORDER])
        ax.set_xlabel("Validation Macro-F1")
        ax.set_title(rep, loc="left")
        ax.set_xlim(max(0.0, float(sub["worst_f1"].min()) - 0.16), min(1.0, float(sub["best_f1"].max()) + 0.16))
        ax.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WORST_COLOR, markersize=8, label="Worst"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BEST_COLOR, markersize=8, label="Best"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Best–Worst Spinal-Level Bottleneck Analysis (Validation Set)")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return _save_figure(fig, output_dir, "best_worst_spinal_level_bottleneck")


def plot_spinal_profiles(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    x = np.arange(len(LEVEL_ORDER))
    specs = (
        (0, "f1_mean", "Macro-F1"),
        (1, "qwk_mean", "QWK"),
    )
    for row_i, metric, ylabel in specs:
        for col_i, rep in enumerate(REP_ORDER):
            ax = axes[row_i][col_i]
            sub = df[df["rep"] == rep]
            for arch in ARCH_ORDER:
                series = sub[sub["arch"] == arch].set_index("level").loc[list(LEVEL_ORDER)]
                ax.plot(
                    x,
                    series[metric],
                    color=ARCH_COLORS[arch],
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    label=arch,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([LEVEL_LABELS[k] for k in LEVEL_ORDER])
            if col_i == 0:
                ax.set_ylabel(ylabel)
            if row_i == 0:
                ax.set_title(rep, loc="left")
            if row_i == 1:
                ax.set_xlabel("Spinal level")
    handles = [
        plt.Line2D([0], [0], color=color, marker="o", label=arch)
        for arch, color in ARCH_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Spinal-Level Performance Profiles")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    return _save_figure(fig, output_dir, "spinal_level_profile_plots")


def plot_anatomical_multi_metric_overlay(agg: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 5.8),
        gridspec_kw={"width_ratios": [0.55, 1.25, 1.25]},
    )
    metrics = (
        ("f1_mean", "Macro-F1 ↑", True),
        ("qwk_mean", "QWK ↑", True),
        ("omae_mean", "O-MAE ↓", False),
        ("ser_mean", "SER ↓", False),
    )
    n_levels = len(LEVEL_ORDER)
    spine_ax = axes[0]
    spine_ax.plot([0.5, 0.5], [-0.4, n_levels - 0.6], color="#6B7280", linewidth=7, solid_capstyle="round")
    for yi, key in enumerate(LEVEL_ORDER):
        spine_ax.scatter(0.5, yi, s=560, color="#F3F4F6", edgecolor="#374151", linewidth=1.4, zorder=2)
        spine_ax.text(0.5, yi, LEVEL_LABELS[key], ha="center", va="center", fontsize=8, zorder=3)
    spine_ax.set_xlim(0.0, 1.0)
    spine_ax.set_ylim(-0.55, n_levels - 0.45)
    spine_ax.invert_yaxis()
    spine_ax.axis("off")
    spine_ax.set_title("Lumbar levels", loc="left")

    for ax, rep in zip(axes[1:], REP_ORDER):
        sub = agg[agg["rep"] == rep].set_index("level").loc[list(LEVEL_ORDER)]
        table = np.array([[float(sub.loc[level, name]) for name, _, _ in metrics] for level in LEVEL_ORDER])
        for j, (_, _, hib) in enumerate(metrics):
            col_vals = table[:, j]
            lo, hi = float(col_vals.min()), float(col_vals.max())
            span = hi - lo if hi > lo else 1.0
            normed = (col_vals - lo) / span
            if not hib:
                normed = 1.0 - normed
            colors = cm.YlGn(0.22 + 0.70 * normed)
            for i, (value, color) in enumerate(zip(col_vals, colors)):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.48, i - 0.42),
                        0.96,
                        0.84,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.1,
                    )
                )
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8.5)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(-0.55, n_levels - 0.45)
        ax.invert_yaxis()
        ax.set_xticks(range(4))
        ax.set_xticklabels([label for _, label, _ in metrics], fontsize=8)
        ax.set_yticks([])
        ax.set_title(rep, loc="left")
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Anatomically Grounded Multi-Metric Performance Overlay")
    fig.text(
        0.5,
        0.01,
        "Values are means across ViT, MaxViT, ConvNeXt, and EfficientNet. "
        "Cell color is scaled within each metric column (green = more favorable).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return _save_figure(fig, output_dir, "anatomically_grounded_multi_metric_overlay")


def _heatmap_text_color(value: float, vmin: float, vmax: float) -> str:
    span = vmax - vmin
    t = (value - vmin) / span if span else 0.0
    return "#FFFFFF" if t >= 0.68 else "#111827"


def plot_macro_f1_heatmaps(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    _style()
    vmin = float(df["f1_mean"].min())
    vmax = float(df["f1_mean"].max())
    paths: dict[str, Path] = {}
    stems = {
        "2D": "spinal_level_macro_f1_heatmap_2d",
        "2.5D-to-3D": "spinal_level_macro_f1_heatmap_25d_to_3d",
    }
    cmap = MACRO_F1_CMAP
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    last_im = None
    for ax, rep in zip(axes, REP_ORDER):
        pivot = (
            df[df["rep"] == rep]
            .pivot(index="arch", columns="level", values="f1_mean")
            .reindex(index=list(ARCH_ORDER), columns=list(LEVEL_ORDER))
        )
        last_im = ax.imshow(pivot.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(ARCH_ORDER)):
            for j in range(len(LEVEL_ORDER)):
                value = float(pivot.iloc[i, j])
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_heatmap_text_color(value, vmin, vmax),
                )
        ax.set_xticks(range(len(LEVEL_ORDER)))
        ax.set_xticklabels([LEVEL_LABELS[k] for k in LEVEL_ORDER])
        ax.set_yticks(range(len(ARCH_ORDER)))
        ax.set_yticklabels(list(ARCH_ORDER))
        ax.set_xlabel("Spinal level")
        ax.set_ylabel("Architecture")
        ax.set_title(f"Spinal-Level Macro-F1 Heatmap ({rep})")
        single = plt.figure(figsize=(6.4, 4.0))
        sax = single.add_subplot(111)
        sax.imshow(pivot.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(ARCH_ORDER)):
            for j in range(len(LEVEL_ORDER)):
                value = float(pivot.iloc[i, j])
                sax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_heatmap_text_color(value, vmin, vmax),
                )
        sax.set_xticks(range(len(LEVEL_ORDER)))
        sax.set_xticklabels([LEVEL_LABELS[k] for k in LEVEL_ORDER])
        sax.set_yticks(range(len(ARCH_ORDER)))
        sax.set_yticklabels(list(ARCH_ORDER))
        sax.set_xlabel("Spinal level")
        sax.set_ylabel("Architecture")
        sax.set_title(f"Spinal-Level Macro-F1 Heatmap ({rep})")
        cbar_s = single.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=sax, fraction=0.046)
        cbar_s.set_label("Macro-F1 (higher is better)")
        single.tight_layout()
        png, pdf = _save_figure(single, output_dir, stems[rep])
        paths[f"{rep}_png"] = png
        paths[f"{rep}_pdf"] = pdf
    cbar_ax = fig.add_axes((0.93, 0.18, 0.015, 0.64))
    fig.colorbar(last_im, cax=cbar_ax).set_label("Macro-F1 (higher is better)")
    fig.subplots_adjust(left=0.07, right=0.91, wspace=0.28, top=0.86, bottom=0.16)
    combined_png, combined_pdf = _save_figure(fig, output_dir, "spinal_level_macro_f1_profiles")
    paths["combined_png"] = combined_png
    paths["combined_pdf"] = combined_pdf
    return paths


def generate_spinal_level_figures(
    *,
    means_path: Path = DEFAULT_MEANS,
    output_dir: Path = SPINAL_LEVEL_FIGURES_DIR,
    split: str = PLOT_SPLIT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_data(Path(means_path), split=split)
    source = save_plot_source_csv(df, output_dir)
    bottlenecks = compute_bottlenecks(df)
    bottleneck_path = output_dir / "spinal_level_bottleneck_source.csv"
    bottlenecks.to_csv(bottleneck_path, index=False)
    overlay = compute_representation_level_means(df)
    overlay_path = output_dir / "spinal_level_anatomical_overlay_source.csv"
    overlay.to_csv(overlay_path, index=False)

    bottleneck_png, bottleneck_pdf = plot_best_worst_bottleneck(bottlenecks, output_dir)
    profile_png, profile_pdf = plot_spinal_profiles(df, output_dir)
    overlay_png, overlay_pdf = plot_anatomical_multi_metric_overlay(overlay, output_dir)
    heatmaps = plot_macro_f1_heatmaps(df, output_dir)

    extra = {"n_rows": int(len(df)), "vmin_f1": float(df["f1_mean"].min()), "vmax_f1": float(df["f1_mean"].max())}
    meta_bottleneck = save_metadata(output_dir, stem="best_worst_spinal_level_bottleneck", source_file=Path(means_path), extra=extra)
    meta_profiles = save_metadata(output_dir, stem="spinal_level_profile_plots", source_file=Path(means_path), extra=extra)
    meta_overlay = save_metadata(
        output_dir,
        stem="anatomically_grounded_multi_metric_overlay",
        source_file=Path(means_path),
        extra=extra,
    )
    meta_heat = save_metadata(output_dir, stem="spinal_level_macro_f1_profiles", source_file=Path(means_path), extra=extra)

    print(
        "\nSPINAL-LEVEL FIGURE GENERATION COMPLETE\n\n"
        f"Source:\n{means_path}\n\n"
        "Configurations:\n"
        "4 architectures × 2 representations × 5 levels = 40 rows\n\n"
        "Generated:\n"
        "- Best–Worst Spinal-Level Bottleneck Analysis\n"
        "- Spinal-Level Profile Plots\n"
        "- Anatomically Grounded Multi-Metric Performance Overlay\n"
        "- Spinal-Level Macro-F1 Profiles\n\n"
        "All figure values were loaded from computed experiment results.\n"
        "No manuscript values or illustrative metrics were used.\n"
    )
    return {
        "plot_source": source,
        "bottleneck_source": bottleneck_path,
        "overlay_source": overlay_path,
        "bottleneck_png": bottleneck_png,
        "bottleneck_pdf": bottleneck_pdf,
        "profile_png": profile_png,
        "profile_pdf": profile_pdf,
        "overlay_png": overlay_png,
        "overlay_pdf": overlay_pdf,
        "heatmap_2d_png": heatmaps["2D_png"],
        "heatmap_25d_png": heatmaps["2.5D-to-3D_png"],
        "meta_bottleneck": meta_bottleneck,
        "meta_profiles": meta_profiles,
        "meta_overlay": meta_overlay,
        "meta_heat": meta_heat,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot spinal-level evaluation figures from harvested CSVs.")
    parser.add_argument("--means", type=Path, default=DEFAULT_MEANS)
    parser.add_argument("--output-dir", type=Path, default=SPINAL_LEVEL_FIGURES_DIR)
    parser.add_argument("--split", default=PLOT_SPLIT)
    args = parser.parse_args(argv)
    generate_spinal_level_figures(means_path=args.means, output_dir=args.output_dir, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
