"""Beyond-Accuracy spinal-level evaluation on harvested pipeline_results.

Anatomical decomposition of the same 200 selected runs: metrics per lumbar
level, then five-run means, then macro-level mean / best / worst / range / std.
Not a new classifier. Incomplete repeat groups are flagged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.article_summary import load_all_pipeline_results
from rsna2024_lumbar.best_config_runs.condition_specific import (
    CONDITION_ORDER,
    EXPECTED_REPEATS,
    EXPECTED_RUNS,
    LEVEL_SEVERITY_TOKEN,
    SPLIT_SOURCE_TO_NAME,
    SPLITS,
    architecture_label,
    condition_abbrev,
    condition_display_name,
    representation_label,
    _best_worst,
    _format_pm,
    _load_metrics_best_row,
    _mean_std,
    _oa_equals_accuracy,
)
from rsna2024_lumbar.preprocessing.constants import LEVEL_DISPLAY, LUMBAR_LEVELS

LEVEL_ORDER: tuple[str, ...] = tuple(LEVEL_DISPLAY[level] for level in LUMBAR_LEVELS)
LEVEL_SLUG_TO_CANON: dict[str, str] = dict(LEVEL_DISPLAY)
LEVEL_CANON_TO_SLUG: dict[str, str] = {v: k for k, v in LEVEL_DISPLAY.items()}
LEVEL_VALUE_COLS: tuple[str, ...] = (
    "l1_l2_value",
    "l2_l3_value",
    "l3_l4_value",
    "l4_l5_value",
    "l5_s1_value",
)

SPINAL_METRICS: tuple[tuple[str, bool], ...] = (
    ("accuracy", True),
    ("precision_macro", True),
    ("recall_macro", True),
    ("f1_macro", True),
    ("oa", True),
    ("omae", False),
    ("qwk", True),
    ("ser", False),
)
HIGHER_IS_BETTER: dict[str, bool] = {name: hib for name, hib in SPINAL_METRICS}
PAPER_METRICS: tuple[str, ...] = ("f1_macro", "qwk", "omae", "ser", "accuracy", "oa")
PAPER_SPLITS: tuple[str, ...] = ("validation", "test")
RADAR_METRICS: tuple[str, ...] = ("f1_macro", "qwk", "omae", "ser")
HEATMAP_METRICS: tuple[str, ...] = ("f1_macro", "qwk")
PIPELINE_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("ViT", "2D"),
    ("MaxViT", "2D"),
    ("ConvNeXt", "2D"),
    ("EfficientNet", "2D"),
    ("ViT", "2.5D-to-3D"),
    ("MaxViT", "2.5D-to-3D"),
    ("ConvNeXt", "2.5D-to-3D"),
    ("EfficientNet", "2.5D-to-3D"),
)

OUTPUT_RUNS = "spinal_level_runs.csv"
OUTPUT_MEANS = "spinal_level_means.csv"
OUTPUT_CONDITION_SUMMARY = "spinal_level_condition_summary.csv"
OUTPUT_PIPELINE_SUMMARY = "spinal_level_pipeline_summary.csv"
OUTPUT_PAPER_TABLE = "spinal_level_paper_table.csv"
OUTPUT_JSON = "spinal_level_summary.json"

SPLIT_SRC_TO_CM: dict[str, str] = {"train": "train", "val": "val", "test": "test"}


def canonical_spinal_level(raw: str) -> str:
    text = str(raw or "").strip()
    if text in LEVEL_ORDER:
        return text
    key = text.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    if key in LEVEL_SLUG_TO_CANON:
        return LEVEL_SLUG_TO_CANON[key]
    upper = text.upper().replace("_", "/")
    if upper in LEVEL_ORDER:
        return upper
    raise ValueError(f"Unknown spinal level alias: {raw!r}")


def _metrics_from_row(row: pd.Series, split_src: str, level_slug: str) -> dict[str, Any]:
    sev = LEVEL_SEVERITY_TOKEN[level_slug]
    mapping = {
        "accuracy": f"{split_src}_accuracy_{level_slug}",
        "precision_macro": f"{split_src}_precision_macro_{level_slug}",
        "recall_macro": f"{split_src}_recall_macro_{level_slug}",
        "f1_macro": f"{split_src}_f1_macro_{level_slug}",
        "oa": f"{split_src}_oa_{sev}",
        "omae": f"{split_src}_omae_{sev}",
        "qwk": f"{split_src}_qwk_{sev}",
        "ser": f"{split_src}_ser_{sev}",
    }
    out: dict[str, Any] = {}
    for dest, src in mapping.items():
        out[dest] = row[src] if src in row.index else ""
    return out


def _class_support_from_cm(
    repeat_dir: Path, split_src: str, level_slug: str
) -> dict[str, Any]:
    cm_split = SPLIT_SRC_TO_CM.get(split_src, split_src)
    path = repeat_dir / f"confusion_matrix_{cm_split}_{level_slug}.csv"
    empty = {"n_samples": "", "n_class_0": "", "n_class_1": "", "n_class_2": ""}
    if not path.is_file():
        return empty
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return empty
    counts = pd.to_numeric(df.sum(axis=1), errors="coerce").fillna(0.0)
    n0 = float(counts.iloc[0]) if len(counts) > 0 else 0.0
    n1 = float(counts.iloc[1]) if len(counts) > 1 else 0.0
    n2 = float(counts.iloc[2]) if len(counts) > 2 else 0.0
    return {
        "n_samples": int(n0 + n1 + n2),
        "n_class_0": int(n0),
        "n_class_1": int(n1),
        "n_class_2": int(n2),
    }


def _repeat_dir(output_base: Path | None, run_dir: str) -> Path | None:
    if not run_dir:
        return None
    path = Path(run_dir)
    if path.is_dir():
        return path
    if output_base is not None:
        nested = output_base / run_dir
        if nested.is_dir():
            return nested
    return None


def build_runs_dataframe(
    pipeline_df: pd.DataFrame,
    *,
    output_base: Path | None = None,
) -> pd.DataFrame:
    if pipeline_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    missing_class_warnings: list[str] = []
    for _, src in pipeline_df.iterrows():
        architecture = architecture_label(src.get("family", ""))
        representation = representation_label(src.get("archive_layout", ""))
        condition_slug = str(src.get("condition") or "")
        abbrev = condition_abbrev(condition_slug)
        seed = src.get("split_seed", "")
        repeat_dir = _repeat_dir(output_base, str(src.get("run_dir") or ""))
        metrics_row = src
        if repeat_dir is not None:
            loaded = _load_metrics_best_row(
                repeat_dir / "training_metrics.csv", src.get("best_epoch")
            )
            if loaded is not None:
                metrics_row = loaded
        for split_src, split_name in SPLIT_SOURCE_TO_NAME.items():
            for level_slug in LUMBAR_LEVELS:
                spinal_level = LEVEL_SLUG_TO_CANON[level_slug]
                rec: dict[str, Any] = {
                    "architecture": architecture,
                    "representation": representation,
                    "condition": abbrev,
                    "condition_name": condition_display_name(condition_slug),
                    "condition_slug": condition_slug,
                    "repeat": src.get("repeat_index", ""),
                    "seed": seed,
                    "split": split_name,
                    "spinal_level": spinal_level,
                    "model_config": src.get("model_config", ""),
                    "model_label": src.get("model_label", ""),
                    "trial_id": src.get("nas_trial_id", ""),
                    "best_epoch": src.get("best_epoch", ""),
                    "checkpoint": "",
                }
                rec.update(_metrics_from_row(metrics_row, split_src, level_slug))
                rec["oa_equals_accuracy"] = _oa_equals_accuracy(rec.get("oa"), rec.get("accuracy"))
                support = (
                    _class_support_from_cm(repeat_dir, split_src, level_slug)
                    if repeat_dir is not None
                    else {"n_samples": "", "n_class_0": "", "n_class_1": "", "n_class_2": ""}
                )
                rec.update(support)
                for cls_i, col in enumerate(("n_class_0", "n_class_1", "n_class_2")):
                    val = support.get(col, "")
                    if val == 0:
                        missing_class_warnings.append(
                            f"{architecture} {representation} {abbrev} {split_name} "
                            f"{spinal_level}: class {cls_i} has zero support"
                        )
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["missing_class_warnings"] = missing_class_warnings
    return out


def aggregate_level_means(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    keys = ("architecture", "representation", "condition", "split", "spinal_level")
    rows: list[dict[str, Any]] = []
    for key_tuple, group in runs.groupby(list(keys), dropna=False, sort=False):
        key_vals = dict(zip(keys, key_tuple))
        n_repeats = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        status = "complete" if n_repeats >= EXPECTED_REPEATS else "incomplete"
        rec: dict[str, Any] = {**key_vals, "n_repeats": n_repeats, "status": status}
        for metric, _hib in SPINAL_METRICS:
            if metric not in group.columns:
                rec[f"{metric}_mean"] = float("nan")
                rec[f"{metric}_std"] = float("nan")
                continue
            mean, std = _mean_std(pd.to_numeric(group[metric], errors="coerce").tolist())
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    level_rank = {lv: i for i, lv in enumerate(LEVEL_ORDER)}
    out["_c"] = out["condition"].map(cond_rank)
    out["_s"] = out["split"].map(split_rank)
    out["_l"] = out["spinal_level"].map(level_rank)
    return (
        out.sort_values(["architecture", "representation", "_c", "_s", "_l"], kind="mergesort")
        .drop(columns=["_c", "_s", "_l"])
        .reset_index(drop=True)
    )


def summarize_by_condition(means: pd.DataFrame) -> pd.DataFrame:
    if means.empty:
        return pd.DataFrame()
    keys = ("architecture", "representation", "condition", "split")
    rows: list[dict[str, Any]] = []
    for key_tuple, group in means.groupby(list(keys), dropna=False, sort=False):
        key_vals = dict(zip(keys, key_tuple))
        incomplete = bool((group["status"] == "incomplete").any()) if "status" in group.columns else False
        for metric, hib in SPINAL_METRICS:
            col = f"{metric}_mean"
            level_vals = {
                str(r["spinal_level"]): float(r[col])
                for _, r in group.iterrows()
                if col in r.index and np.isfinite(pd.to_numeric(r[col], errors="coerce"))
            }
            ordered = [level_vals[lv] for lv in LEVEL_ORDER if lv in level_vals]
            macro, level_std = _mean_std(ordered)
            best, worst = _best_worst(level_vals, higher_is_better=hib)
            rec: dict[str, Any] = {
                **key_vals,
                "metric": metric,
                "macro_level_mean": macro,
                "level_std": level_std,
                "best_level": best,
                "best_level_value": level_vals.get(best, float("nan")),
                "worst_level": worst,
                "worst_level_value": level_vals.get(worst, float("nan")),
                "level_range": float(max(ordered) - min(ordered)) if ordered else float("nan"),
                "n_levels": len(level_vals),
                "status": "incomplete" if incomplete or len(level_vals) < 5 else "complete",
            }
            for slug, col_name in zip(LUMBAR_LEVELS, LEVEL_VALUE_COLS):
                rec[col_name] = level_vals.get(LEVEL_SLUG_TO_CANON[slug], float("nan"))
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    metric_rank = {m: i for i, (m, _) in enumerate(SPINAL_METRICS)}
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    out["_c"] = out["condition"].map(cond_rank)
    out["_s"] = out["split"].map(split_rank)
    out["_m"] = out["metric"].map(metric_rank)
    return (
        out.sort_values(["architecture", "representation", "_c", "_s", "_m"], kind="mergesort")
        .drop(columns=["_c", "_s", "_m"])
        .reset_index(drop=True)
    )


def summarize_pipeline(means: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean across the five conditions at each spinal level."""
    if means.empty:
        return pd.DataFrame()
    keys = ("architecture", "representation", "split")
    rows: list[dict[str, Any]] = []
    for key_tuple, group in means.groupby(list(keys), dropna=False, sort=False):
        key_vals = dict(zip(keys, key_tuple))
        incomplete = bool((group["status"] == "incomplete").any()) if "status" in group.columns else False
        for metric, hib in SPINAL_METRICS:
            col = f"{metric}_mean"
            per_level: dict[str, float] = {}
            for level in LEVEL_ORDER:
                slice_ = group[group["spinal_level"] == level]
                vals = [
                    float(r[col])
                    for _, r in slice_.iterrows()
                    if col in r.index and np.isfinite(pd.to_numeric(r[col], errors="coerce"))
                ]
                if vals:
                    per_level[level] = float(np.mean(vals))
            ordered = [per_level[lv] for lv in LEVEL_ORDER if lv in per_level]
            macro, level_std = _mean_std(ordered)
            best, worst = _best_worst(per_level, higher_is_better=hib)
            rec: dict[str, Any] = {
                **key_vals,
                "metric": metric,
                "l1_l2_mean": per_level.get("L1/L2", float("nan")),
                "l2_l3_mean": per_level.get("L2/L3", float("nan")),
                "l3_l4_mean": per_level.get("L3/L4", float("nan")),
                "l4_l5_mean": per_level.get("L4/L5", float("nan")),
                "l5_s1_mean": per_level.get("L5/S1", float("nan")),
                "macro_level_mean": macro,
                "best_level": best,
                "best_level_value": per_level.get(best, float("nan")),
                "worst_level": worst,
                "worst_level_value": per_level.get(worst, float("nan")),
                "level_range": float(max(ordered) - min(ordered)) if ordered else float("nan"),
                "level_std": level_std,
                "n_conditions": int(group["condition"].nunique()) if "condition" in group.columns else 0,
                "status": "incomplete" if incomplete or len(per_level) < 5 else "complete",
            }
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    metric_rank = {m: i for i, (m, _) in enumerate(SPINAL_METRICS)}
    out["_s"] = out["split"].map(split_rank)
    out["_m"] = out["metric"].map(metric_rank)
    return (
        out.sort_values(["architecture", "representation", "_s", "_m"], kind="mergesort")
        .drop(columns=["_s", "_m"])
        .reset_index(drop=True)
    )


def build_paper_table(pipeline_summary: pd.DataFrame) -> pd.DataFrame:
    if pipeline_summary.empty:
        return pd.DataFrame()
    keep = pipeline_summary[
        pipeline_summary["metric"].isin(PAPER_METRICS)
        & pipeline_summary["split"].isin(PAPER_SPLITS)
    ]
    rows: list[dict[str, Any]] = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "split": row["split"],
                "representation": row["representation"],
                "architecture": row["architecture"],
                "metric": row["metric"],
                "l1_l2": row["l1_l2_mean"],
                "l2_l3": row["l2_l3_mean"],
                "l3_l4": row["l3_l4_mean"],
                "l4_l5": row["l4_l5_mean"],
                "l5_s1": row["l5_s1_mean"],
                "macro_level_mean": row["macro_level_mean"],
                "best_level": row["best_level"],
                "worst_level": row["worst_level"],
                "level_range": row["level_range"],
                "level_std": row["level_std"],
                "status": row["status"],
            }
        )
    return pd.DataFrame(rows)


def build_radar_frame(pipeline_summary: pd.DataFrame, *, split: str) -> pd.DataFrame:
    subset = pipeline_summary[
        (pipeline_summary["split"] == split) & pipeline_summary["metric"].isin(RADAR_METRICS)
    ]
    if subset.empty:
        return pd.DataFrame(columns=["architecture", "representation", "metric", *LEVEL_ORDER])
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "architecture": row["architecture"],
                "representation": row["representation"],
                "metric": row["metric"],
                "L1/L2": row["l1_l2_mean"],
                "L2/L3": row["l2_l3_mean"],
                "L3/L4": row["l3_l4_mean"],
                "L4/L5": row["l4_l5_mean"],
                "L5/S1": row["l5_s1_mean"],
            }
        )
    return pd.DataFrame(rows)


def build_heatmap_frame(pipeline_summary: pd.DataFrame, *, metric: str, split: str) -> pd.DataFrame:
    subset = pipeline_summary[
        (pipeline_summary["metric"] == metric) & (pipeline_summary["split"] == split)
    ]
    rows: list[dict[str, Any]] = []
    lookup = {
        (str(r["architecture"]), str(r["representation"])): r for _, r in subset.iterrows()
    }
    for arch, rep in PIPELINE_ROW_ORDER:
        rec: dict[str, Any] = {"pipeline": f"{arch} - {rep}"}
        hit = lookup.get((arch, rep))
        if hit is None:
            for level in LEVEL_ORDER:
                rec[level] = float("nan")
        else:
            rec["L1/L2"] = hit["l1_l2_mean"]
            rec["L2/L3"] = hit["l2_l3_mean"]
            rec["L3/L4"] = hit["l3_l4_mean"]
            rec["L4/L5"] = hit["l4_l5_mean"]
            rec["L5/S1"] = hit["l5_s1_mean"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _ranking_notes(pipeline_summary: pd.DataFrame) -> str:
    if pipeline_summary.empty:
        return "No complete spinal-level summaries to rank."
    bits: list[str] = []
    sub = pipeline_summary[
        (pipeline_summary["metric"] == "f1_macro")
        & (pipeline_summary["split"] == "validation")
        & (pipeline_summary["status"] == "complete")
    ]
    for _, row in sub.iterrows():
        bits.append(
            f"{row['architecture']} {row['representation']} validation Macro-F1: "
            f"macro-level {float(row['macro_level_mean']):.4f}, "
            f"best {row['best_level']}, worst {row['worst_level']}, "
            f"range {float(row['level_range']):.4f}."
        )
    return " ".join(bits) if bits else "Incomplete groups only; no ranking."


def build_summary_json(
    runs: pd.DataFrame,
    means: pd.DataFrame,
    pipeline_summary: pd.DataFrame,
) -> dict[str, Any]:
    incomplete: list[dict[str, Any]] = []
    if not means.empty:
        seen: set[tuple[str, str, str]] = set()
        for _, row in means.iterrows():
            if row.get("status") != "incomplete":
                continue
            key = (str(row["architecture"]), str(row["representation"]), str(row["condition"]))
            if key in seen:
                continue
            seen.add(key)
            incomplete.append(
                {
                    "architecture": key[0],
                    "representation": key[1],
                    "condition": key[2],
                    "n_repeats": int(row["n_repeats"]),
                    "expected_repeats": EXPECTED_REPEATS,
                }
            )
    n_runs = 0
    if not runs.empty:
        n_runs = int(
            runs.drop_duplicates(["architecture", "representation", "condition", "seed"]).shape[0]
        )
    oa_flags = runs["oa_equals_accuracy"] if not runs.empty and "oa_equals_accuracy" in runs.columns else pd.Series(dtype=object)
    comparable = oa_flags.dropna()
    return {
        "n_runs_expected": EXPECTED_RUNS,
        "n_runs_found": n_runs,
        "architectures": sorted(runs["architecture"].unique().tolist()) if not runs.empty else [],
        "representations": sorted(runs["representation"].unique().tolist()) if not runs.empty else [],
        "conditions": list(CONDITION_ORDER),
        "spinal_levels": list(LEVEL_ORDER),
        "incomplete_groups": incomplete,
        "oa_equals_accuracy": bool(comparable.all()) if len(comparable) else None,
        "oa_equals_accuracy_checked_rows": int(len(comparable)),
        "missing_class_warnings": list(dict.fromkeys(runs.attrs.get("missing_class_warnings") or [])),
        "ranking_notes": _ranking_notes(pipeline_summary),
    }


def print_terminal_summary(pipeline_summary: pd.DataFrame, *, split: str = "validation", metric: str = "f1_macro") -> None:
    sub = pipeline_summary[
        (pipeline_summary["split"] == split) & (pipeline_summary["metric"] == metric)
    ]
    for _, row in sub.iterrows():
        print(
            f"\nArchitecture: {row['architecture']}\n"
            f"Representation: {row['representation']}\n"
            f"Split: {split}\n"
            f"Metric: Macro-F1\n\n"
            f"L1/L2: {row['l1_l2_mean']}\n"
            f"L2/L3: {row['l2_l3_mean']}\n"
            f"L3/L4: {row['l3_l4_mean']}\n"
            f"L4/L5: {row['l4_l5_mean']}\n"
            f"L5/S1: {row['l5_s1_mean']}\n\n"
            f"Macro-level mean: {row['macro_level_mean']}\n"
            f"Best level: {row['best_level']}\n"
            f"Worst level: {row['worst_level']}\n"
            f"Level range: {row['level_range']}\n"
            f"Level SD: {row['level_std']}"
        )


def write_spinal_level_outputs(
    output_base: Path,
    *,
    pipeline_df: pd.DataFrame | None = None,
    print_summary: bool = True,
) -> dict[str, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    df = pipeline_df if pipeline_df is not None else load_all_pipeline_results(output_base)
    if df.empty:
        raise ValueError(f"No pipeline_results.csv under {output_base}")

    runs = build_runs_dataframe(df, output_base=output_base)
    means = aggregate_level_means(runs)
    by_condition = summarize_by_condition(means)
    pipeline_summary = summarize_pipeline(means)
    paper = build_paper_table(pipeline_summary)
    summary = build_summary_json(runs, means, pipeline_summary)

    paths: dict[str, Path] = {}
    paths["runs"] = output_base / OUTPUT_RUNS
    runs.to_csv(paths["runs"], index=False)
    paths["means"] = output_base / OUTPUT_MEANS
    means.to_csv(paths["means"], index=False)
    paths["condition_summary"] = output_base / OUTPUT_CONDITION_SUMMARY
    by_condition.to_csv(paths["condition_summary"], index=False)
    paths["pipeline_summary"] = output_base / OUTPUT_PIPELINE_SUMMARY
    pipeline_summary.to_csv(paths["pipeline_summary"], index=False)
    paths["paper_table"] = output_base / OUTPUT_PAPER_TABLE
    paper.to_csv(paths["paper_table"], index=False)

    for split in PAPER_SPLITS:
        radar = build_radar_frame(pipeline_summary, split=split)
        path = output_base / f"spinal_level_radar_{split}.csv"
        radar.to_csv(path, index=False)
        paths[f"radar_{split}"] = path

    for metric in HEATMAP_METRICS:
        short = "macro_f1" if metric == "f1_macro" else metric
        heat = build_heatmap_frame(pipeline_summary, metric=metric, split="validation")
        path = output_base / f"spinal_level_heatmap_{short}_validation.csv"
        heat.to_csv(path, index=False)
        paths[f"heatmap_{short}_validation"] = path

    paths["summary_json"] = output_base / OUTPUT_JSON
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    unique_warns = list(dict.fromkeys(summary.get("missing_class_warnings") or []))
    for msg in unique_warns[:20]:
        print(f"WARNING: {msg}")
    if len(unique_warns) > 20:
        print(f"WARNING: {len(unique_warns) - 20} additional class-support warnings (see {OUTPUT_JSON})")
    if print_summary:
        print_terminal_summary(pipeline_summary)
    return paths
