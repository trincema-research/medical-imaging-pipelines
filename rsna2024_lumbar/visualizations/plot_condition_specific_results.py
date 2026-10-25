"""Publication figures for condition-specific evaluation results.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import CONDITION_SPECIFIC_FIGURES_DIR

ARCH_ORDER: tuple[str, ...] = ("ViT", "MaxViT", "ConvNeXt", "EfficientNet")
REP_ORDER: tuple[str, ...] = ("2D", "2.5D-to-3D")
CONDITION_ORDER: tuple[str, ...] = ("SCS", "LFN", "RFN", "LSS", "RSS")
PLOT_SPLIT = "validation"

ARCH_COLORS: dict[str, str] = {
    "ViT": "#1F77B4",
    "MaxViT": "#FF7F0E",
    "ConvNeXt": "#2CA02C",
    "EfficientNet": "#D62728",
}
REP_MARKERS: dict[str, str] = {"2D": "o", "2.5D-to-3D": "^"}
CONDITION_COLORS: dict[str, str] = {
    "SCS": "#4C78A8",
    "LFN": "#F2CF5B",
    "RFN": "#E45756",
    "LSS": "#72B7B2",
    "RSS": "#B279A2",
}

DEFAULT_MEANS = RESULTS_DIR / "protocol_condition_specific" / "condition_specific_by_condition.csv"

REQUIRED_COLUMNS: frozenset[str] = frozenset({"arch", "rep", "condition", "f1_mean", "f1_std"})
OPTIONAL_METRICS: tuple[str, ...] = ("acc", "precision", "recall", "oa", "omae", "qwk", "ser")

LONG_METRIC_ALIASES: dict[str, str] = {
    "f1": "f1",
    "f1_macro_overall": "f1",
    "accuracy_overall": "acc",
    "acc": "acc",
    "precision_macro_overall": "precision",
    "precision": "precision",
    "recall_macro_overall": "recall",
    "recall": "recall",
    "oa": "oa",
    "oa_overall": "oa",
    "omae": "omae",
    "omae_overall": "omae",
    "qwk": "qwk",
    "qwk_overall": "qwk",
    "ser": "ser",
    "ser_overall": "ser",
}


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
        "metric": "Macro-F1",
        "architectures": list(ARCH_ORDER),
        "representations": list(REP_ORDER),
        "conditions": list(CONDITION_ORDER),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_name": "plot_condition_specific_results.py",
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


def _canonical_condition(raw: str) -> str:
    aliases = {
        "SCS": "SCS",
        "SPINAL_CANAL_STENOSIS": "SCS",
        "LFN": "LFN",
        "LEFT_NEURAL_FORAMINAL_NARROWING": "LFN",
        "RFN": "RFN",
        "RIGHT_NEURAL_FORAMINAL_NARROWING": "RFN",
        "LSS": "LSS",
        "LEFT_SUBARTICULAR_STENOSIS": "LSS",
        "RSS": "RSS",
        "RIGHT_SUBARTICULAR_STENOSIS": "RSS",
    }
    key = str(raw or "").strip().upper().replace(" ", "_").replace("-", "_")
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown condition: {raw!r}")


def _from_wide(raw: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = raw.copy()
    if "split" in work.columns:
        work = work[work["split"] == split].copy()
    arch_col = _resolve_col(work, ("arch", "architecture"), label="arch")
    rep_col = _resolve_col(work, ("rep", "representation"), label="rep")
    cond_col = _resolve_col(work, ("condition",), label="condition")
    out = pd.DataFrame(
        {
            "arch": work[arch_col],
            "rep": work[rep_col],
            "condition": work[cond_col].map(_canonical_condition),
        }
    )
    dests = ("f1",) + OPTIONAL_METRICS
    for dest in dests:
        mean_names = (f"{dest}_mean", f"{dest}_macro_overall_mean", f"{dest}_overall_mean", dest)
        try:
            mean_col = _resolve_col(work, mean_names, label=f"{dest}_mean")
        except ValueError:
            if dest == "f1":
                raise
            continue
        out[f"{dest}_mean"] = pd.to_numeric(work[mean_col], errors="coerce")
        std_names = (f"{dest}_std", f"{dest}_macro_overall_std", f"{dest}_overall_std")
        std_col = next((name for name in std_names if name in work.columns), None)
        out[f"{dest}_std"] = pd.to_numeric(work[std_col], errors="coerce") if std_col else 0.0
    if "n_runs" in work.columns:
        out["n_runs"] = pd.to_numeric(work["n_runs"], errors="coerce")
    elif "n_repeats" in work.columns:
        out["n_runs"] = pd.to_numeric(work["n_repeats"], errors="coerce")
    else:
        out["n_runs"] = np.nan
    return out


def _from_long(raw: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = raw.copy()
    if "split" in work.columns:
        work = work[work["split"] == split].copy()
    arch_col = _resolve_col(work, ("arch", "architecture"), label="arch")
    rep_col = _resolve_col(work, ("rep", "representation"), label="rep")
    cond_col = _resolve_col(work, ("condition",), label="condition")
    metric_col = _resolve_col(work, ("metric",), label="metric")
    mean_col = _resolve_col(work, ("mean",), label="mean")
    std_col = _resolve_col(work, ("std",), label="std")
    work["arch"] = work[arch_col]
    work["rep"] = work[rep_col]
    work["condition"] = work[cond_col].map(_canonical_condition)
    work["_metric"] = work[metric_col].map(lambda x: LONG_METRIC_ALIASES.get(str(x).strip().lower(), ""))
    wanted = {"f1", *OPTIONAL_METRICS}
    work = work[work["_metric"].isin(wanted)].copy()
    if work.empty:
        raise ValueError(f"No condition-specific metric rows for split={split}")
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["arch", "rep", "condition"], dropna=False, sort=False)
    for (arch, rep, condition), group in grouped:
        rec: dict[str, Any] = {"arch": arch, "rep": rep, "condition": condition}
        for dest in ("f1", *OPTIONAL_METRICS):
            sub = group[group["_metric"] == dest]
            if sub.empty:
                rec[f"{dest}_mean"] = float("nan")
                rec[f"{dest}_std"] = float("nan")
                continue
            rec[f"{dest}_mean"] = float(pd.to_numeric(sub[mean_col], errors="coerce").iloc[0])
            rec[f"{dest}_std"] = float(pd.to_numeric(sub[std_col], errors="coerce").iloc[0])
        if "n_runs" in group.columns:
            rec["n_runs"] = int(pd.to_numeric(group["n_runs"], errors="coerce").iloc[0])
        elif "n_repeats" in group.columns:
            rec["n_runs"] = int(pd.to_numeric(group["n_repeats"], errors="coerce").iloc[0])
        else:
            rec["n_runs"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def load_and_validate_data(means_path: Path, *, split: str = PLOT_SPLIT) -> pd.DataFrame:
    if not means_path.is_file():
        raise FileNotFoundError(means_path)
    raw = pd.read_csv(means_path)
    if "metric" in raw.columns and "mean" in raw.columns:
        df = _from_long(raw, split=split)
    else:
        df = _from_wide(raw, split=split)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("NaNs in required condition-specific metrics")
    if set(df["arch"]) != set(ARCH_ORDER):
        raise ValueError(f"Unexpected architectures: {sorted(set(df['arch']))}")
    if set(df["rep"]) != set(REP_ORDER):
        raise ValueError(f"Unexpected representations: {sorted(set(df['rep']))}")
    if set(df["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(df['condition']))}")
    if len(df) != 40:
        raise ValueError(f"Expected 40 architecture-representation-condition rows, got {len(df)}")
    if not df["f1_mean"].between(0, 1).all():
        raise ValueError("f1_mean out of range [0, 1]")
    if "qwk_mean" in df.columns and df["qwk_mean"].notna().any() and not df["qwk_mean"].dropna().between(-1, 1).all():
        raise ValueError("qwk_mean out of range [-1, 1]")
    if "omae_mean" in df.columns and df["omae_mean"].notna().any() and not df["omae_mean"].dropna().between(0, 2).all():
        raise ValueError("omae_mean out of range [0, 2]")
    if "ser_mean" in df.columns and df["ser_mean"].notna().any() and not df["ser_mean"].dropna().between(0, 1).all():
        raise ValueError("ser_mean out of range [0, 1]")
    arch_rank = {k: i for i, k in enumerate(ARCH_ORDER)}
    rep_rank = {k: i for i, k in enumerate(REP_ORDER)}
    cond_rank = {k: i for i, k in enumerate(CONDITION_ORDER)}
    df["_a"] = df["arch"].map(arch_rank)
    df["_r"] = df["rep"].map(rep_rank)
    df["_c"] = df["condition"].map(cond_rank)
    return df.sort_values(["_r", "_a", "_c"], kind="mergesort").drop(columns=["_a", "_r", "_c"]).reset_index(drop=True)


def incomplete_groups(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "n_runs" not in df.columns:
        return []
    rows = df[pd.to_numeric(df["n_runs"], errors="coerce") < 5]
    return [
        {"arch": row.arch, "rep": row.rep, "condition": row.condition, "n_runs": int(row.n_runs)}
        for row in rows.itertuples(index=False)
    ]


def compute_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rep in REP_ORDER:
        for arch in ARCH_ORDER:
            sub = df[(df["rep"] == rep) & (df["arch"] == arch)]
            worst = sub.loc[sub["f1_mean"].idxmin()]
            best = sub.loc[sub["f1_mean"].idxmax()]
            rows.append(
                {
                    "rep": rep,
                    "arch": arch,
                    "worst_condition": worst["condition"],
                    "worst_f1": float(worst["f1_mean"]),
                    "best_condition": best["condition"],
                    "best_f1": float(best["f1_mean"]),
                    "condition_range": float(best["f1_mean"] - worst["f1_mean"]),
                    "condition_sd": float(sub["f1_mean"].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def plot_reliability_profiles(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    x = np.arange(len(CONDITION_ORDER))
    for ax, rep in zip(axes, REP_ORDER):
        for arch in ARCH_ORDER:
            sub = df[(df["rep"] == rep) & (df["arch"] == arch)].set_index("condition").reindex(list(CONDITION_ORDER))
            ax.errorbar(
                x,
                sub["f1_mean"].to_numpy(dtype=float),
                yerr=sub["f1_std"].to_numpy(dtype=float),
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=3,
                color=ARCH_COLORS[arch],
                label=arch,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(CONDITION_ORDER))
        ax.set_xlabel("Degenerative condition")
        ax.set_title(rep, fontweight="bold")
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Validation Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Architecture",
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Condition-Specific Reliability Profiles", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    return _save_figure(fig, output_dir, "condition_specific_reliability_profiles")


def plot_reliability_radar(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    angles = np.linspace(0, 2 * np.pi, len(CONDITION_ORDER), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), subplot_kw={"projection": "polar"})
    for ax, rep in zip(axes, REP_ORDER):
        for arch in ARCH_ORDER:
            sub = df[(df["rep"] == rep) & (df["arch"] == arch)].set_index("condition").reindex(list(CONDITION_ORDER))
            values = sub["f1_mean"].to_numpy(dtype=float)
            values = np.concatenate([values, [values[0]]])
            ax.plot(angles, values, linewidth=2, color=ARCH_COLORS[arch], label=arch)
            ax.scatter(angles, values, s=25, color=ARCH_COLORS[arch])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(CONDITION_ORDER))
        ax.set_ylim(0, 1)
        ax.set_title(rep, fontweight="bold", pad=16)
    axes[1].legend(bbox_to_anchor=(1.28, 1.12), frameon=False)
    fig.suptitle("Condition-Specific Macro-F1 Reliability Profiles", fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "condition_specific_reliability_radar")


def plot_bottleneck_analysis(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    plot_df = summary.copy()
    plot_df["label"] = plot_df["arch"] + " (" + plot_df["rep"] + ")"
    y = np.arange(len(plot_df))
    colors = [CONDITION_COLORS[c] for c in plot_df["worst_condition"]]
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.barh(y, plot_df["worst_f1"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Worst-condition validation Macro-F1")
    ax.set_title("Condition-Specific Bottleneck Analysis", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.axhline(3.5, color="0.45", linewidth=1)
    xmax = float(plot_df["worst_f1"].max())
    ax.set_xlim(0.0, xmax + 0.12)
    for i, row in plot_df.iterrows():
        ax.text(
            row["worst_f1"] + 0.008,
            i,
            f"{row['worst_condition']}  {row['worst_f1']:.3f}",
            va="center",
            fontsize=9.5,
        )
    handles = [Patch(facecolor=CONDITION_COLORS[c], edgecolor="black", label=c) for c in CONDITION_ORDER]
    ax.legend(
        handles=handles,
        title="Worst condition",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return _save_figure(fig, output_dir, "condition_specific_bottleneck_analysis")


def plot_bottleneck_uniformity(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    offsets = {
        ("ViT", "2D"): (-78, 10),
        ("MaxViT", "2D"): (8, -16),
        ("ConvNeXt", "2D"): (8, 8),
        ("EfficientNet", "2D"): (-100, 10),
        ("ViT", "2.5D-to-3D"): (8, 8),
        ("MaxViT", "2.5D-to-3D"): (8, -16),
        ("ConvNeXt", "2.5D-to-3D"): (8, 8),
        ("EfficientNet", "2.5D-to-3D"): (8, -16),
    }
    for row in summary.itertuples(index=False):
        ax.scatter(
            row.condition_range,
            row.worst_f1,
            s=170,
            marker=REP_MARKERS[row.rep],
            color=ARCH_COLORS[row.arch],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{row.arch} ({row.rep})",
            (row.condition_range, row.worst_f1),
            xytext=offsets.get((row.arch, row.rep), (6, 5)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Condition Macro-F1 Range ↓")
    ax.set_ylabel("Worst-Condition Macro-F1 ↑")
    ax.set_title("Condition-Specific Bottleneck and Uniformity", fontweight="bold")
    ax.grid(alpha=0.25)
    arch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ARCH_COLORS[arch],
            markeredgecolor="black",
            markersize=8,
            label=arch,
        )
        for arch in ARCH_ORDER
    ]
    legend1 = ax.legend(handles=arch_handles, title="Architecture", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.add_artist(legend1)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=8, label="2D"),
            Line2D([0], [0], marker="^", linestyle="None", color="black", markersize=8, label="2.5D-to-3D"),
        ],
        title="Representation",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        frameon=False,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "condition_specific_bottleneck_uniformity")


def plot_macro_f1_heatmap(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    matrix_rows = []
    row_labels = []
    for rep in REP_ORDER:
        for arch in ARCH_ORDER:
            sub = df[(df["rep"] == rep) & (df["arch"] == arch)].set_index("condition").reindex(list(CONDITION_ORDER))
            matrix_rows.append(sub["f1_mean"].to_numpy(dtype=float))
            row_labels.append(f"{arch} ({rep})")
    data = np.vstack(matrix_rows)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels(list(CONDITION_ORDER))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    vmin = float(data.min())
    vmax = float(data.max())
    span = vmax - vmin
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            t = (float(data[i, j]) - vmin) / span if span else 0.0
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#FFFFFF" if t >= 0.72 else "#111827",
            )
        min_j = int(np.argmin(data[i]))
        ax.add_patch(Rectangle((min_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
    ax.axhline(3.5, color="white", linewidth=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Validation Macro-F1")
    ax.set_title("Condition-Specific Validation Macro-F1", fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "condition_specific_macro_f1_heatmap")


def generate_condition_specific_figures(
    *,
    means_path: Path = DEFAULT_MEANS,
    output_dir: Path = CONDITION_SPECIFIC_FIGURES_DIR,
    split: str = PLOT_SPLIT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_data(Path(means_path), split=split)
    source = output_dir / "condition_specific_plot_source.csv"
    df.to_csv(source, index=False)
    bottlenecks = compute_bottlenecks(df)
    bottleneck_path = output_dir / "condition_specific_bottleneck_uniformity_source.csv"
    bottlenecks.to_csv(bottleneck_path, index=False)
    missing = incomplete_groups(df)
    extra = {"n_rows": int(len(df)), "incomplete_groups": missing, "n_runs_min": None if df["n_runs"].isna().all() else int(df["n_runs"].min())}

    profile_png, profile_pdf = plot_reliability_profiles(df, output_dir)
    radar_png, radar_pdf = plot_reliability_radar(df, output_dir)
    bottleneck_png, bottleneck_pdf = plot_bottleneck_analysis(bottlenecks, output_dir)
    uniformity_png, uniformity_pdf = plot_bottleneck_uniformity(bottlenecks, output_dir)
    heatmap_png, heatmap_pdf = plot_macro_f1_heatmap(df, output_dir)

    meta_profiles = save_metadata(output_dir, stem="condition_specific_reliability_profiles", source_file=Path(means_path), extra=extra)
    meta_uniform = save_metadata(output_dir, stem="condition_specific_bottleneck_uniformity", source_file=Path(means_path), extra=extra)
    meta_bottleneck = save_metadata(output_dir, stem="condition_specific_bottleneck_analysis", source_file=Path(means_path), extra=extra)
    meta_radar = save_metadata(output_dir, stem="condition_specific_reliability_radar", source_file=Path(means_path), extra=extra)
    meta_heat = save_metadata(output_dir, stem="condition_specific_macro_f1_heatmap", source_file=Path(means_path), extra=extra)

    print(
        "\nCONDITION-SPECIFIC FIGURE GENERATION COMPLETE\n\n"
        f"Source:\n{means_path}\n\n"
        "Expected aggregate rows:\n"
        "40\n\n"
        "Architectures:\n"
        "ViT\nMaxViT\nConvNeXt\nEfficientNet\n\n"
        "Representations:\n"
        "2D\n2.5D-to-3D\n\n"
        "Conditions:\n"
        "SCS\nLFN\nRFN\nLSS\nRSS\n\n"
        "Generated:\n"
        "- Condition-Specific Reliability Profiles\n"
        "- Condition-Specific Bottleneck Analysis\n"
        "- Condition-Specific Bottleneck and Uniformity\n"
        "- Optional Macro-F1 Heatmap\n"
        "- Optional Radar Profiles\n\n"
        "All plotted values were loaded from experimental result files.\n"
        "No manuscript or AI-generated numerical values were used.\n"
    )
    return {
        "plot_source": source,
        "bottleneck_source": bottleneck_path,
        "profile_png": profile_png,
        "profile_pdf": profile_pdf,
        "radar_png": radar_png,
        "radar_pdf": radar_pdf,
        "bottleneck_png": bottleneck_png,
        "bottleneck_pdf": bottleneck_pdf,
        "uniformity_png": uniformity_png,
        "uniformity_pdf": uniformity_pdf,
        "heatmap_png": heatmap_png,
        "heatmap_pdf": heatmap_pdf,
        "meta_profiles": meta_profiles,
        "meta_uniform": meta_uniform,
        "meta_bottleneck": meta_bottleneck,
        "meta_radar": meta_radar,
        "meta_heat": meta_heat,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot condition-specific evaluation figures from harvested CSVs.")
    parser.add_argument("--means", type=Path, default=DEFAULT_MEANS)
    parser.add_argument("--output-dir", type=Path, default=CONDITION_SPECIFIC_FIGURES_DIR)
    parser.add_argument("--split", default=PLOT_SPLIT)
    args = parser.parse_args(argv)
    generate_condition_specific_figures(means_path=args.means, output_dir=args.output_dir, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
