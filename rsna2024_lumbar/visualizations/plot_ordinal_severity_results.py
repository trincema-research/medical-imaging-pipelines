"""Publication figures for ordinal severity evaluation results.

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
from matplotlib.patches import Rectangle

from rsna2024_lumbar.best_config_runs.paths import RESULTS_DIR
from rsna2024_lumbar.visualizations import ORDINAL_FIGURES_DIR

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
REP_COLORS: dict[str, str] = {"2D": "#2F80C9", "2.5D-to-3D": "#FF7A1A"}
GAIN_COLOR = "#16A34A"
DEGRADATION_COLOR = "#EF4444"

DEFAULT_MEANS = RESULTS_DIR / "condition_specific_by_condition.csv"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "architecture",
        "representation",
        "condition",
        "oa_mean",
        "oa_std",
        "omae_mean",
        "omae_std",
        "qwk_mean",
        "qwk_std",
        "ser_mean",
        "ser_std",
    }
)

LONG_METRIC_ALIASES: dict[str, str] = {
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
        "metrics": ["OA", "O-MAE", "QWK", "SER"],
        "architectures": list(ARCH_ORDER),
        "representations": list(REP_ORDER),
        "conditions": list(CONDITION_ORDER),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_name": "plot_ordinal_severity_results.py",
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
    text = str(raw or "").strip()
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
    key = text.upper().replace(" ", "_").replace("-", "_")
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown condition: {raw!r}")


def _from_wide(raw: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = raw.copy()
    if "split" in work.columns:
        work = work[work["split"] == split].copy()
    arch_col = _resolve_col(work, ("architecture", "arch"), label="architecture")
    rep_col = _resolve_col(work, ("representation", "rep"), label="representation")
    cond_col = _resolve_col(work, ("condition",), label="condition")
    out = pd.DataFrame(
        {
            "architecture": work[arch_col],
            "representation": work[rep_col],
            "condition": work[cond_col].map(_canonical_condition),
        }
    )
    for dest in ("oa", "omae", "qwk", "ser"):
        mean_col = _resolve_col(work, (f"{dest}_mean", f"{dest}_overall_mean", dest), label=f"{dest}_mean")
        out[f"{dest}_mean"] = pd.to_numeric(work[mean_col], errors="coerce")
        std_names = (f"{dest}_std", f"{dest}_overall_std")
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
    arch_col = _resolve_col(work, ("architecture", "arch"), label="architecture")
    rep_col = _resolve_col(work, ("representation", "rep"), label="representation")
    cond_col = _resolve_col(work, ("condition",), label="condition")
    metric_col = _resolve_col(work, ("metric",), label="metric")
    mean_col = _resolve_col(work, ("mean",), label="mean")
    std_col = _resolve_col(work, ("std",), label="std")
    work["architecture"] = work[arch_col]
    work["representation"] = work[rep_col]
    work["condition"] = work[cond_col].map(_canonical_condition)
    work["_metric"] = work[metric_col].map(lambda x: LONG_METRIC_ALIASES.get(str(x).strip().lower(), ""))
    work = work[work["_metric"].isin(("oa", "omae", "qwk", "ser"))].copy()
    if work.empty:
        raise ValueError(f"No OA/O-MAE/QWK/SER rows for split={split}")
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["architecture", "representation", "condition"], dropna=False, sort=False)
    for (arch, rep, condition), group in grouped:
        rec: dict[str, Any] = {"architecture": arch, "representation": rep, "condition": condition}
        for dest in ("oa", "omae", "qwk", "ser"):
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
        raise ValueError("NaNs in required ordinal metrics")
    if set(df["architecture"]) != set(ARCH_ORDER):
        raise ValueError(f"Unexpected architectures: {sorted(set(df['architecture']))}")
    if set(df["representation"]) != set(REP_ORDER):
        raise ValueError(f"Unexpected representations: {sorted(set(df['representation']))}")
    if set(df["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"Unexpected conditions: {sorted(set(df['condition']))}")
    if len(df) != 40:
        raise ValueError(f"Expected 40 architecture-representation-condition rows, got {len(df)}")
    if not df["oa_mean"].between(0, 1).all():
        raise ValueError("oa_mean out of range [0, 1]")
    if not df["omae_mean"].between(0, 2).all():
        raise ValueError("omae_mean out of range [0, 2]")
    if not df["qwk_mean"].between(-1, 1).all():
        raise ValueError("qwk_mean out of range [-1, 1]")
    if not df["ser_mean"].between(0, 1).all():
        raise ValueError("ser_mean out of range [0, 1]")
    arch_rank = {k: i for i, k in enumerate(ARCH_ORDER)}
    rep_rank = {k: i for i, k in enumerate(REP_ORDER)}
    cond_rank = {k: i for i, k in enumerate(CONDITION_ORDER)}
    df["_a"] = df["architecture"].map(arch_rank)
    df["_r"] = df["representation"].map(rep_rank)
    df["_c"] = df["condition"].map(cond_rank)
    return df.sort_values(["_r", "_a", "_c"], kind="mergesort").drop(columns=["_a", "_r", "_c"]).reset_index(drop=True)


def compute_architecture_means(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["representation", "architecture"], as_index=False)
        .agg(
            oa_mean=("oa_mean", "mean"),
            omae_mean=("omae_mean", "mean"),
            qwk_mean=("qwk_mean", "mean"),
            ser_mean=("ser_mean", "mean"),
            n_runs=("n_runs", "min") if "n_runs" in df.columns else ("oa_mean", "size"),
        )
    )
    if len(agg) != 8:
        raise ValueError(f"Expected 8 architecture-representation rows, got {len(agg)}")
    return agg


def save_plot_source_csvs(df: pd.DataFrame, arch: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    condition_path = output_dir / "ordinal_plot_source_condition_level.csv"
    arch_path = output_dir / "ordinal_plot_source_architecture_means.csv"
    df.to_csv(condition_path, index=False)
    arch.to_csv(arch_path, index=False)
    return condition_path, arch_path


def _leaders(arch: pd.DataFrame) -> dict[str, str]:
    leaders: dict[str, str] = {}
    for rep in REP_ORDER:
        sub = arch[arch["representation"] == rep]
        best_qwk = str(sub.loc[sub["qwk_mean"].idxmax(), "architecture"])
        best_omae = str(sub.loc[sub["omae_mean"].idxmin(), "architecture"])
        best_oa = str(sub.loc[sub["oa_mean"].idxmax(), "architecture"])
        best_ser = str(sub.loc[sub["ser_mean"].idxmin(), "architecture"])
        if best_qwk == best_omae == best_oa == best_ser:
            leaders[rep] = best_qwk
    return leaders


def plot_qwk_vs_omae_tradeoff(arch: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    offsets = {
        ("ViT", "2D"): (9, 10),
        ("MaxViT", "2D"): (10, 16),
        ("ConvNeXt", "2D"): (12, 8),
        ("EfficientNet", "2D"): (-88, -12),
        ("ViT", "2.5D-to-3D"): (8, 8),
        ("MaxViT", "2.5D-to-3D"): (8, 10),
        ("ConvNeXt", "2.5D-to-3D"): (8, 8),
        ("EfficientNet", "2.5D-to-3D"): (8, -16),
    }
    plot_rows = list(arch.itertuples(index=False))
    plot_rows.sort(key=lambda row: (row.architecture == "MaxViT" and row.representation == "2D"))
    for row in plot_rows:
        zorder = 6 if row.architecture == "MaxViT" and row.representation == "2D" else 3
        ax.scatter(
            row.qwk_mean,
            row.omae_mean,
            s=180,
            marker=REP_MARKERS[row.representation],
            color=ARCH_COLORS[row.architecture],
            edgecolor="black",
            linewidth=0.8,
            zorder=zorder,
        )
        ax.annotate(
            f"{row.architecture} ({row.representation})",
            (row.qwk_mean, row.omae_mean),
            xytext=offsets.get((row.architecture, row.representation), (7, 5)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Quadratic Weighted Kappa (QWK) ↑")
    ax.set_ylabel("Ordinal Mean Absolute Error (O-MAE) ↓")
    ax.set_title("Ordinal Performance Trade-Off: QWK vs O-MAE", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.axvspan(0.65, max(0.82, float(arch["qwk_mean"].max()) + 0.02), alpha=0.07, color="#1F77B4")
    ax.axhspan(0.00, 0.17, alpha=0.05, color="#16A34A")
    ax.text(0.67, 0.04, "Ideal region\nhigh QWK, low O-MAE", fontsize=10, va="bottom")
    arch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ARCH_COLORS[arch_name],
            markeredgecolor="black",
            markersize=9,
            label=arch_name,
        )
        for arch_name in ARCH_ORDER
    ]
    rep_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=8, label="2D"),
        Line2D([0], [0], marker="^", linestyle="None", color="black", markersize=8, label="2.5D-to-3D"),
    ]
    legend1 = ax.legend(handles=arch_handles, title="Architecture", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=rep_handles, title="Representation", loc="upper left", bbox_to_anchor=(1.01, 0.55))
    fig.tight_layout()
    return _save_figure(fig, output_dir, "ordinal_qwk_vs_omae_tradeoff")


def _cell_text_color(value: float, vmin: float, vmax: float, *, invert: bool = False) -> str:
    span = vmax - vmin
    t = (value - vmin) / span if span else 0.0
    if invert:
        t = 1.0 - t
    return "#FFFFFF" if t >= 0.72 else "#111827"


def plot_ordinal_metric_heatmaps(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    metrics = (
        ("oa_mean", "OA ↑", "YlGnBu", False),
        ("omae_mean", "O-MAE ↓", "Greens_r", True),
        ("qwk_mean", "QWK ↑", "YlGnBu", False),
        ("ser_mean", "SER ↓", "RdYlGn_r", False),
    )
    fig = plt.figure(figsize=(16.8, 8.8))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.08], hspace=0.38, wspace=0.24)
    axes = [[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(2)]
    for col_idx, (metric, title, cmap, invert_text) in enumerate(metrics):
        vmin = float(df[metric].min())
        vmax = float(df[metric].max())
        last_im = None
        for row_idx, rep in enumerate(REP_ORDER):
            ax = axes[row_idx][col_idx]
            sub = df[df["representation"] == rep]
            matrix = (
                sub.pivot(index="architecture", columns="condition", values=metric)
                .reindex(index=list(ARCH_ORDER), columns=list(CONDITION_ORDER))
            )
            data = matrix.to_numpy(dtype=float)
            last_im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(CONDITION_ORDER)))
            ax.set_xticklabels(list(CONDITION_ORDER))
            ax.set_yticks(range(len(ARCH_ORDER)))
            ax.set_yticklabels(list(ARCH_ORDER))
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(rep, fontsize=12, fontweight="bold")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=_cell_text_color(float(data[i, j]), vmin, vmax, invert=invert_text),
                    )
        cax = fig.add_subplot(gs[2, col_idx])
        fig.colorbar(last_im, cax=cax, orientation="horizontal")
    fig.suptitle("Ordinal Metric Heatmaps", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08)
    return _save_figure(fig, output_dir, "ordinal_metric_heatmaps")


def plot_2d_vs_25d_degradation_gain(arch: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    metric_cfg = (
        ("oa_mean", "Ordinal Accuracy (OA) ↑", True),
        ("qwk_mean", "Quadratic Weighted Kappa (QWK) ↑", True),
        ("omae_mean", "Ordinal Mean Absolute Error (O-MAE) ↓", False),
        ("ser_mean", "Severe Error Rate (SER) ↓", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    x = np.arange(len(ARCH_ORDER))
    width = 0.36
    leaders = _leaders(arch)
    for ax, (metric, title, higher_is_better) in zip(axes.flatten(), metric_cfg):
        values_2d = []
        values_25d = []
        for name in ARCH_ORDER:
            v2 = float(
                arch[(arch["representation"] == "2D") & (arch["architecture"] == name)][metric].iloc[0]
            )
            v25 = float(
                arch[(arch["representation"] == "2.5D-to-3D") & (arch["architecture"] == name)][metric].iloc[0]
            )
            values_2d.append(v2)
            values_25d.append(v25)
        ax.bar(x - width / 2, values_2d, width, label="2D", color=REP_COLORS["2D"])
        ax.bar(x + width / 2, values_25d, width, label="2.5D-to-3D", color=REP_COLORS["2.5D-to-3D"])
        for i, (v2, v25) in enumerate(zip(values_2d, values_25d)):
            delta = (v25 - v2) if higher_is_better else (v2 - v25)
            color = GAIN_COLOR if delta > 0 else DEGRADATION_COLOR
            ymax = max(v2, v25)
            ax.text(
                i,
                ymax * 1.08 if ymax > 0 else 0.01,
                f"Δ {delta:+.4f}",
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(ARCH_ORDER))
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        lo = min(min(values_2d), min(values_25d))
        hi = max(max(values_2d), max(values_25d))
        pad = max(abs(hi) * 0.22, 0.02)
        ax.set_ylim(min(0.0, lo - pad * 0.1), hi + pad)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.suptitle("2D vs 2.5D-to-3D Ordinal Degradation / Gain", fontsize=16, fontweight="bold")
    if leaders:
        note = "  ·  ".join(f"{rep}: {arch_name} leads all four metrics" for rep, arch_name in leaders.items())
        fig.text(0.5, 0.008, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    else:
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    return _save_figure(fig, output_dir, "ordinal_2d_vs_25d_gain_degradation")


def plot_severe_ordinal_error_burden(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    _style()
    vmin = 0.0
    vmax = max(0.10, float(df["ser_mean"].max()))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
    last_im = None
    for ax, rep in zip(axes, REP_ORDER):
        sub = df[df["representation"] == rep]
        pivot = (
            sub.pivot(index="architecture", columns="condition", values="ser_mean")
            .reindex(index=list(ARCH_ORDER), columns=list(CONDITION_ORDER))
        )
        matrix = pivot.to_numpy(dtype=float)
        means = matrix.mean(axis=1, keepdims=True)
        full = np.concatenate([matrix, means], axis=1)
        last_im = ax.imshow(full, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        labels = list(CONDITION_ORDER) + ["Mean"]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(ARCH_ORDER)))
        ax.set_yticklabels(list(ARCH_ORDER))
        ax.set_title(f"{rep} Models", fontsize=13, fontweight="bold")
        for i in range(full.shape[0]):
            for j in range(full.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{full[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_cell_text_color(float(full[i, j]), vmin, vmax),
                )
        max_i, max_j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        ax.add_patch(
            Rectangle((max_j - 0.5, max_i - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=1.6)
        )
    cbar_ax = fig.add_axes((0.93, 0.18, 0.015, 0.64))
    fig.colorbar(last_im, cax=cbar_ax, label="Severe Ordinal Error Rate (SER)")
    fig.suptitle("Severe Ordinal Error Burden", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.91, wspace=0.22, top=0.84, bottom=0.14)
    return _save_figure(fig, output_dir, "severe_ordinal_error_burden")


def generate_ordinal_severity_figures(
    *,
    means_path: Path = DEFAULT_MEANS,
    output_dir: Path = ORDINAL_FIGURES_DIR,
    split: str = PLOT_SPLIT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_data(Path(means_path), split=split)
    arch = compute_architecture_means(df)
    source_cond, source_arch = save_plot_source_csvs(df, arch, output_dir)

    tradeoff_png, tradeoff_pdf = plot_qwk_vs_omae_tradeoff(arch, output_dir)
    heat_png, heat_pdf = plot_ordinal_metric_heatmaps(df, output_dir)
    gain_png, gain_pdf = plot_2d_vs_25d_degradation_gain(arch, output_dir)
    ser_png, ser_pdf = plot_severe_ordinal_error_burden(df, output_dir)

    extra = {
        "n_condition_rows": int(len(df)),
        "n_architecture_rows": int(len(arch)),
        "n_runs_min": None if df["n_runs"].isna().all() else int(df["n_runs"].min()),
        "leaders": _leaders(arch),
    }
    meta_tradeoff = save_metadata(output_dir, stem="ordinal_qwk_vs_omae_tradeoff", source_file=Path(means_path), extra=extra)
    meta_heat = save_metadata(output_dir, stem="ordinal_metric_heatmaps", source_file=Path(means_path), extra=extra)
    meta_gain = save_metadata(output_dir, stem="ordinal_2d_vs_25d_gain_degradation", source_file=Path(means_path), extra=extra)
    meta_ser = save_metadata(output_dir, stem="severe_ordinal_error_burden", source_file=Path(means_path), extra=extra)

    print(
        "\nORDINAL SEVERITY FIGURE GENERATION COMPLETE\n\n"
        f"Source:\n{means_path}\n\n"
        "Expected observations:\n"
        "40 condition-level rows\n"
        "8 architecture-representation configurations\n\n"
        "Generated:\n"
        "- Ordinal Performance Trade-Off: QWK vs O-MAE\n"
        "- Ordinal Metric Heatmaps\n"
        "- 2D vs 2.5D-to-3D Ordinal Degradation / Gain Plot\n"
        "- Severe Ordinal Error Burden\n\n"
        "All values were loaded from experimental result files.\n"
        "No manuscript table values or AI-generated numerical values were used.\n"
    )
    return {
        "plot_source_condition": source_cond,
        "plot_source_architecture": source_arch,
        "tradeoff_png": tradeoff_png,
        "tradeoff_pdf": tradeoff_pdf,
        "heatmaps_png": heat_png,
        "heatmaps_pdf": heat_pdf,
        "gain_png": gain_png,
        "gain_pdf": gain_pdf,
        "ser_png": ser_png,
        "ser_pdf": ser_pdf,
        "meta_tradeoff": meta_tradeoff,
        "meta_heat": meta_heat,
        "meta_gain": meta_gain,
        "meta_ser": meta_ser,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot ordinal severity figures from harvested CSVs.")
    parser.add_argument("--means", type=Path, default=DEFAULT_MEANS)
    parser.add_argument("--output-dir", type=Path, default=ORDINAL_FIGURES_DIR)
    parser.add_argument("--split", default=PLOT_SPLIT)
    args = parser.parse_args(argv)
    generate_ordinal_severity_figures(means_path=args.means, output_dir=args.output_dir, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
