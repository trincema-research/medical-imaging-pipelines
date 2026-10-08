"""Beyond-Accuracy condition-specific evaluation on harvested pipeline_results.

Post-hoc layer (same pattern as article-summary): five-run means per condition,
then condition-macro mean / best / worst / range / std. Does not train a new
classifier. Incomplete repeat groups are flagged and are not treated as complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.article_summary import load_all_pipeline_results
from rsna2024_lumbar.best_config_runs.paths import CONDITION_SPECIFIC_SPLIT_SEEDS
from rsna2024_lumbar.preprocessing.constants import CONDITIONS, LUMBAR_LEVELS

EXPECTED_RUNS = 200
EXPECTED_REPEATS = 5
EXPECTED_CONDITIONS = 5
OA_EQ_ATOL = 1e-6

CONDITION_ABBREV: dict[str, str] = {
    "spinal_canal_stenosis": "SCS",
    "left_neural_foraminal_narrowing": "LFN",
    "right_neural_foraminal_narrowing": "RFN",
    "left_subarticular_stenosis": "LSS",
    "right_subarticular_stenosis": "RSS",
}
CONDITION_ORDER: tuple[str, ...] = ("SCS", "LFN", "RFN", "LSS", "RSS")

ARCHITECTURE_LABELS: dict[str, str] = {
    "vit": "ViT",
    "vit3d": "ViT",
    "maxvit": "MaxViT",
    "maxvit3d": "MaxViT",
    "convnext": "ConvNeXt",
    "convnext3d": "ConvNeXt",
    "efficientnet": "EfficientNet",
    "efficientnet3d": "EfficientNet",
}

SPLIT_SOURCE_TO_NAME: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}
SPLITS: tuple[str, ...] = ("train", "validation", "test")

# Dest metric name, source suffix after "{split}_", higher_is_better.
OVERALL_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy_overall", "accuracy_overall", True),
    ("precision_macro_overall", "precision_macro_overall", True),
    ("recall_macro_overall", "recall_macro_overall", True),
    ("f1_macro_overall", "f1_macro_overall", True),
    ("oa_overall", "oa_overall", True),
    ("omae_overall", "omae_overall", False),
    ("qwk_overall", "qwk_overall", True),
    ("ser_overall", "ser_overall", False),
)
HIGHER_IS_BETTER: dict[str, bool] = {name: hib for name, _, hib in OVERALL_METRICS}
LOWER_IS_BETTER: frozenset[str] = frozenset(
    name for name, hib in HIGHER_IS_BETTER.items() if not hib
)

RADAR_METRICS: tuple[str, ...] = tuple(name for name, _, _ in OVERALL_METRICS)
PAPER_METRICS: tuple[str, ...] = (
    "accuracy_overall",
    "f1_macro_overall",
    "oa_overall",
    "qwk_overall",
    "omae_overall",
    "ser_overall",
)
PAPER_SPLITS: tuple[str, ...] = ("validation", "test")

LEVEL_SEVERITY_TOKEN: dict[str, str] = {
    "l1_l2": "L1_L2",
    "l2_l3": "L2_L3",
    "l3_l4": "L3_L4",
    "l4_l5": "L4_L5",
    "l5_s1": "L5_S1",
}

OUTPUT_RUNS = "condition_specific_runs.csv"
OUTPUT_BY_CONDITION = "condition_specific_by_condition.csv"
OUTPUT_PIPELINE_SUMMARY = "condition_specific_pipeline_summary.csv"
OUTPUT_PAPER_TABLE = "condition_specific_paper_table.csv"
OUTPUT_JSON = "condition_specific_summary.json"


def architecture_label(family: str) -> str:
    key = str(family or "").strip().lower()
    return ARCHITECTURE_LABELS.get(key, str(family or "").strip() or "unknown")


def representation_label(archive_layout: str) -> str:
    layout = str(archive_layout or "").strip().lower()
    if layout in {"2d"}:
        return "2D"
    if layout in {"3d", "3d_level_stack"}:
        return "2.5D-to-3D"
    return str(archive_layout or "").strip() or "unknown"


def condition_abbrev(condition: str) -> str:
    slug = str(condition or "").strip()
    return CONDITION_ABBREV.get(slug, slug)


def condition_display_name(condition: str) -> str:
    slug = str(condition or "").strip()
    meta = CONDITIONS.get(slug) or {}
    return str(meta.get("coord_name") or slug)


def _finite(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals[np.isfinite(vals)]


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _format_pm(mean: float, std: float, *, decimals: int = 4) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return ""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _oa_equals_accuracy(oa: Any, accuracy: Any, *, atol: float = OA_EQ_ATOL) -> bool | None:
    oa_v = pd.to_numeric(oa, errors="coerce")
    acc_v = pd.to_numeric(accuracy, errors="coerce")
    if not np.isfinite(oa_v) or not np.isfinite(acc_v):
        return None
    return bool(abs(float(oa_v) - float(acc_v)) <= atol)


def _best_worst(condition_means: dict[str, float], *, higher_is_better: bool) -> tuple[str, str]:
    items = [(k, v) for k, v in condition_means.items() if np.isfinite(v)]
    if not items:
        return "", ""
    if higher_is_better:
        best = max(items, key=lambda kv: kv[1])[0]
        worst = min(items, key=lambda kv: kv[1])[0]
    else:
        best = min(items, key=lambda kv: kv[1])[0]
        worst = max(items, key=lambda kv: kv[1])[0]
    return best, worst


def _per_level_from_metrics_row(row: pd.Series, split_src: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in LUMBAR_LEVELS:
        for dest, src_metric in (
            (f"accuracy_{level}", f"{split_src}_accuracy_{level}"),
            (f"f1_macro_{level}", f"{split_src}_f1_macro_{level}"),
            (f"precision_macro_{level}", f"{split_src}_precision_macro_{level}"),
            (f"recall_macro_{level}", f"{split_src}_recall_macro_{level}"),
        ):
            if src_metric in row.index:
                out[dest] = row[src_metric]
        sev = LEVEL_SEVERITY_TOKEN[level]
        for dest_metric, src_metric in (
            (f"oa_{level}", f"{split_src}_oa_{sev}"),
            (f"omae_{level}", f"{split_src}_omae_{sev}"),
            (f"qwk_{level}", f"{split_src}_qwk_{sev}"),
            (f"ser_{level}", f"{split_src}_ser_{sev}"),
        ):
            if src_metric in row.index:
                out[dest_metric] = row[src_metric]
    return out


def _load_metrics_best_row(metrics_path: Path, best_epoch: Any) -> pd.Series | None:
    if not metrics_path.is_file():
        return None
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    if df.empty:
        return None
    epoch = pd.to_numeric(best_epoch, errors="coerce")
    if np.isfinite(epoch) and "epoch" in df.columns:
        matched = df[pd.to_numeric(df["epoch"], errors="coerce") == int(epoch)]
        if not matched.empty:
            return matched.iloc[-1]
    if "val_acc" in df.columns:
        return df.loc[df["val_acc"].idxmax()]
    return df.iloc[-1]


def _enrich_per_level(
    pipeline_row: pd.Series,
    split_src: str,
    *,
    output_base: Path | None,
) -> dict[str, Any]:
    extras = _per_level_from_metrics_row(pipeline_row, split_src)
    if extras:
        return extras
    run_dir = str(pipeline_row.get("run_dir") or "")
    if not run_dir or output_base is None:
        return {}
    metrics_path = Path(run_dir) / "training_metrics.csv"
    if not metrics_path.is_file():
        metrics_path = output_base / run_dir / "training_metrics.csv"
    row = _load_metrics_best_row(metrics_path, pipeline_row.get("best_epoch"))
    if row is None:
        return {}
    return _per_level_from_metrics_row(row, split_src)


def build_runs_dataframe(
    pipeline_df: pd.DataFrame,
    *,
    output_base: Path | None = None,
) -> pd.DataFrame:
    """Long table: one row per harvested run × split."""
    if pipeline_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, src in pipeline_df.iterrows():
        architecture = architecture_label(src.get("family", ""))
        representation = representation_label(src.get("archive_layout", ""))
        condition_slug = str(src.get("condition") or "")
        abbrev = condition_abbrev(condition_slug)
        seed = src.get("split_seed", "")
        run_id = f"{architecture}|{representation}|{abbrev}|{seed}"
        for split_src, split_name in SPLIT_SOURCE_TO_NAME.items():
            rec: dict[str, Any] = {
                "architecture": architecture,
                "representation": representation,
                "condition": abbrev,
                "condition_name": condition_display_name(condition_slug),
                "condition_slug": condition_slug,
                "seed": seed,
                "repeat_index": src.get("repeat_index", ""),
                "split": split_name,
                "model_config": src.get("model_config", ""),
                "model_label": src.get("model_label", ""),
                "run_id": run_id,
                "n_trainable_params": src.get("n_trainable_params", ""),
            }
            for dest, src_suffix, _hib in OVERALL_METRICS:
                col = f"{split_src}_{src_suffix}"
                rec[dest] = src[col] if col in src.index else ""
            rec["oa_equals_accuracy"] = _oa_equals_accuracy(
                rec.get("oa_overall"), rec.get("accuracy_overall")
            )
            rec.update(_enrich_per_level(src, split_src, output_base=output_base))
            rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_by_condition(runs: pd.DataFrame) -> pd.DataFrame:
    """Five-run mean ± std per architecture × representation × condition × metric × split."""
    if runs.empty:
        return pd.DataFrame()
    group_keys = ("architecture", "representation", "condition", "condition_name", "split")
    rows: list[dict[str, Any]] = []
    grouped = runs.groupby(list(group_keys), dropna=False, sort=False)
    for key_tuple, group in grouped:
        key_vals = dict(zip(group_keys, key_tuple))
        n_runs = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        status = "complete" if n_runs >= EXPECTED_REPEATS else "incomplete"
        for metric, _src, _hib in OVERALL_METRICS:
            if metric not in group.columns:
                continue
            mean, std = _mean_std(_finite(group[metric]).tolist())
            rows.append(
                {
                    **key_vals,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "mean_pm_std": _format_pm(mean, std),
                    "n_runs": n_runs,
                    "expected_runs": EXPECTED_REPEATS,
                    "status": status,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    metric_rank = {m: i for i, (m, _, _) in enumerate(OVERALL_METRICS)}
    out["_c"] = out["condition"].map(cond_rank)
    out["_s"] = out["split"].map(split_rank)
    out["_m"] = out["metric"].map(metric_rank)
    out = out.sort_values(
        ["architecture", "representation", "_c", "_s", "_m"],
        kind="mergesort",
    ).drop(columns=["_c", "_s", "_m"])
    return out.reset_index(drop=True)


def summarize_pipeline(by_condition: pd.DataFrame) -> pd.DataFrame:
    """Condition-macro mean, best/worst, range, std per architecture × representation."""
    if by_condition.empty:
        return pd.DataFrame()
    group_keys = ("architecture", "representation", "metric", "split")
    rows: list[dict[str, Any]] = []
    for key_tuple, group in by_condition.groupby(list(group_keys), dropna=False, sort=False):
        key_vals = dict(zip(group_keys, key_tuple))
        cond_means = {
            str(r["condition"]): float(r["mean"])
            for _, r in group.iterrows()
            if np.isfinite(pd.to_numeric(r["mean"], errors="coerce"))
        }
        ordered = [cond_means[c] for c in CONDITION_ORDER if c in cond_means]
        macro, cond_std = _mean_std(ordered)
        metric = str(key_vals["metric"])
        hib = HIGHER_IS_BETTER.get(metric, True)
        best, worst = _best_worst(cond_means, higher_is_better=hib)
        n_conditions = len(cond_means)
        n_runs = int(pd.to_numeric(group["n_runs"], errors="coerce").sum()) if "n_runs" in group else 0
        incomplete = (
            n_conditions < EXPECTED_CONDITIONS
            or bool((group["status"] == "incomplete").any())
            if "status" in group.columns
            else n_conditions < EXPECTED_CONDITIONS
        )
        rows.append(
            {
                **key_vals,
                "condition_macro_mean": macro,
                "best_condition": best,
                "worst_condition": worst,
                "condition_range": (
                    float(max(ordered) - min(ordered)) if ordered else float("nan")
                ),
                "condition_std": cond_std,
                "n_conditions": n_conditions,
                "n_runs": n_runs,
                "status": "incomplete" if incomplete else "complete",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    metric_rank = {m: i for i, (m, _, _) in enumerate(OVERALL_METRICS)}
    out["_s"] = out["split"].map(split_rank)
    out["_m"] = out["metric"].map(metric_rank)
    out = out.sort_values(
        ["architecture", "representation", "_s", "_m"],
        kind="mergesort",
    ).drop(columns=["_s", "_m"])
    return out.reset_index(drop=True)


def build_paper_table(pipeline_summary: pd.DataFrame, by_condition: pd.DataFrame) -> pd.DataFrame:
    del by_condition
    if pipeline_summary.empty:
        return pd.DataFrame()
    keep = pipeline_summary[
        pipeline_summary["metric"].isin(PAPER_METRICS)
        & pipeline_summary["split"].isin(PAPER_SPLITS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in keep.iterrows():
        macro = float(row["condition_macro_mean"]) if np.isfinite(
            pd.to_numeric(row["condition_macro_mean"], errors="coerce")
        ) else float("nan")
        cond_std = float(row["condition_std"]) if np.isfinite(
            pd.to_numeric(row["condition_std"], errors="coerce")
        ) else float("nan")
        rows.append(
            {
                "architecture": row["architecture"],
                "representation": row["representation"],
                "metric": row["metric"],
                "split": row["split"],
                "condition_macro_mean": macro,
                "condition_macro_std": cond_std,
                "mean_pm_std": _format_pm(macro, cond_std),
                "best_condition": row["best_condition"],
                "worst_condition": row["worst_condition"],
                "condition_range": row["condition_range"],
                "n_conditions": row["n_conditions"],
                "n_runs": row["n_runs"],
                "status": row["status"],
            }
        )
    return pd.DataFrame(rows)


def build_radar_frame(by_condition: pd.DataFrame, *, metric: str, split: str) -> pd.DataFrame:
    subset = by_condition[
        (by_condition["metric"] == metric) & (by_condition["split"] == split)
    ]
    if subset.empty:
        return pd.DataFrame(
            columns=["architecture", "representation", *CONDITION_ORDER, "condition_macro_mean"]
        )
    rows: list[dict[str, Any]] = []
    for (arch, rep), group in subset.groupby(["architecture", "representation"], dropna=False, sort=False):
        rec: dict[str, Any] = {"architecture": arch, "representation": rep}
        means: list[float] = []
        for cond in CONDITION_ORDER:
            hit = group[group["condition"] == cond]
            val = float(hit.iloc[0]["mean"]) if not hit.empty else float("nan")
            rec[cond] = val
            if np.isfinite(val):
                means.append(val)
        rec["condition_macro_mean"] = float(np.mean(means)) if means else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _ranking_notes(pipeline_summary: pd.DataFrame) -> str:
    if pipeline_summary.empty:
        return "No complete condition-specific summaries to rank."
    bits: list[str] = []
    for metric in ("oa_overall", "f1_macro_overall", "qwk_overall"):
        sub = pipeline_summary[
            (pipeline_summary["metric"] == metric)
            & (pipeline_summary["split"] == "validation")
            & (pipeline_summary["status"] == "complete")
        ]
        if sub.empty:
            continue
        hib = HIGHER_IS_BETTER.get(metric, True)
        idx = sub["condition_macro_mean"].idxmax() if hib else sub["condition_macro_mean"].idxmin()
        top = sub.loc[idx]
        bits.append(
            f"{metric} validation: best condition-macro is {top['architecture']} "
            f"{top['representation']} ({float(top['condition_macro_mean']):.4f}); "
            f"best condition {top['best_condition']}, worst {top['worst_condition']}."
        )
    return " ".join(bits) if bits else "Incomplete groups only; no ranking."


def build_summary_json(
    runs: pd.DataFrame,
    by_condition: pd.DataFrame,
    pipeline_summary: pd.DataFrame,
) -> dict[str, Any]:
    incomplete: list[dict[str, Any]] = []
    if not by_condition.empty:
        seen: set[tuple[str, str, str]] = set()
        for _, row in by_condition.iterrows():
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
                    "n_runs": int(row["n_runs"]),
                    "expected_runs": EXPECTED_REPEATS,
                }
            )
    oa_flags = runs["oa_equals_accuracy"] if not runs.empty and "oa_equals_accuracy" in runs.columns else pd.Series(dtype=object)
    comparable = oa_flags.dropna()
    n_runs_found = int(runs["run_id"].nunique()) if not runs.empty and "run_id" in runs.columns else 0
    seeds = sorted({str(s) for s in runs["seed"].tolist()}) if not runs.empty else []
    return {
        "n_runs_expected": EXPECTED_RUNS,
        "n_runs_found": n_runs_found,
        "architectures": sorted(runs["architecture"].unique().tolist()) if not runs.empty else [],
        "representations": sorted(runs["representation"].unique().tolist()) if not runs.empty else [],
        "conditions": list(CONDITION_ORDER),
        "seeds": seeds,
        "protocol_seeds": list(CONDITION_SPECIFIC_SPLIT_SEEDS),
        "incomplete_groups": incomplete,
        "oa_equals_accuracy": bool(comparable.all()) if len(comparable) else None,
        "oa_equals_accuracy_checked_rows": int(len(comparable)),
        "ranking_notes": _ranking_notes(pipeline_summary),
    }


def write_condition_specific_outputs(
    output_base: Path,
    *,
    pipeline_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    df = pipeline_df if pipeline_df is not None else load_all_pipeline_results(output_base)
    if df.empty:
        raise ValueError(f"No pipeline_results.csv under {output_base}")

    runs = build_runs_dataframe(df, output_base=output_base)
    by_condition = aggregate_by_condition(runs)
    pipeline_summary = summarize_pipeline(by_condition)
    paper = build_paper_table(pipeline_summary, by_condition)
    summary = build_summary_json(runs, by_condition, pipeline_summary)

    paths: dict[str, Path] = {}
    paths["runs"] = output_base / OUTPUT_RUNS
    runs.to_csv(paths["runs"], index=False)
    paths["by_condition"] = output_base / OUTPUT_BY_CONDITION
    by_condition.to_csv(paths["by_condition"], index=False)
    paths["pipeline_summary"] = output_base / OUTPUT_PIPELINE_SUMMARY
    pipeline_summary.to_csv(paths["pipeline_summary"], index=False)
    paths["paper_table"] = output_base / OUTPUT_PAPER_TABLE
    paper.to_csv(paths["paper_table"], index=False)

    for metric in RADAR_METRICS:
        short = metric.replace("_overall", "").replace("_macro", "")
        for split in PAPER_SPLITS:
            radar = build_radar_frame(by_condition, metric=metric, split=split)
            name = f"condition_specific_radar_{short}_{split}.csv"
            path = output_base / name
            radar.to_csv(path, index=False)
            paths[f"radar_{short}_{split}"] = path

    paths["summary_json"] = output_base / OUTPUT_JSON
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths
