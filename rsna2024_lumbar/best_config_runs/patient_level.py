"""Beyond-Accuracy patient-level evaluation on per-prediction dumps.

Joins the same 200 selected runs by patient_id (study_id) across conditions
and lumbar levels. Equal-weight patient macros; do not substitute pooled
accuracy. Incomplete 5-run groups are flagged. PLOA is not computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.article_summary import load_all_pipeline_results
from rsna2024_lumbar.best_config_runs.condition_specific import (
    CONDITION_ORDER,
    EXPECTED_REPEATS,
    EXPECTED_RUNS,
    SPLIT_SOURCE_TO_NAME,
    SPLITS,
    architecture_label,
    condition_abbrev,
    representation_label,
    _format_pm,
    _mean_std,
)
from rsna2024_lumbar.best_config_runs.spinal_level import (
    PIPELINE_ROW_ORDER,
    canonical_spinal_level,
)

PROFILE_ORDER: tuple[str, ...] = ("exact", "adjacent_only", "severe")
PAPER_SPLITS: tuple[str, ...] = ("validation", "test")
PREDICTION_FILES: tuple[tuple[str, str], ...] = tuple(SPLIT_SOURCE_TO_NAME.items())
UNIQUENESS_KEYS: tuple[str, ...] = (
    "patient_id",
    "condition",
    "spinal_level",
    "architecture",
    "representation",
    "repeat",
    "split",
)
REPEAT_AGG_METRICS: tuple[str, ...] = (
    "patient_macro_exact_rate",
    "patient_macro_omae",
    "patient_exact_profile_rate",
    "patient_adjacent_only_rate",
    "patient_severe_error_rate",
    "mean_n_errors_per_patient",
    "mean_n_severe_errors_per_patient",
    "mean_n_predictions_per_patient",
)

OUTPUT_RUNS = "patient_level_runs.csv"
OUTPUT_REPEAT = "patient_level_repeat_summary.csv"
OUTPUT_SUMMARY = "patient_level_summary.csv"
OUTPUT_PAPER = "patient_level_paper_table.csv"
OUTPUT_BY_CONDITION = "patient_level_by_condition.csv"
OUTPUT_BY_LEVEL = "patient_level_by_spinal_level.csv"
OUTPUT_JSON = "patient_level_summary.json"


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


def error_profile_from_max_error(patient_max_error: int) -> str:
    if patient_max_error == 0:
        return "exact"
    if patient_max_error == 1:
        return "adjacent_only"
    if patient_max_error == 2:
        return "severe"
    raise ValueError(f"patient_max_error must be 0, 1, or 2; got {patient_max_error}")


def compute_patient_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if yt.size == 0 or yt.size != yp.size:
        raise ValueError("patient metrics need a non-empty y_true/y_pred pair of equal length")
    distance = np.abs(yt - yp)
    n_predictions = int(distance.size)
    n_exact = int(np.sum(distance == 0))
    n_errors = int(np.sum(distance > 0))
    n_adjacent_errors = int(np.sum(distance == 1))
    n_severe_errors = int(np.sum(distance == 2))
    patient_max_error = int(distance.max())
    exact_rate = float(n_exact / n_predictions)
    return {
        "n_predictions": n_predictions,
        "n_exact": n_exact,
        "n_errors": n_errors,
        "n_adjacent_errors": n_adjacent_errors,
        "n_severe_errors": n_severe_errors,
        "patient_exact_rate": exact_rate,
        "patient_error_rate": float(1.0 - exact_rate),
        "patient_omae": float(np.mean(distance)),
        "patient_max_error": patient_max_error,
        "error_profile": error_profile_from_max_error(patient_max_error),
        "has_severe_error": int(patient_max_error == 2),
    }


def load_prediction_records(
    pipeline_df: pd.DataFrame,
    *,
    output_base: Path | None = None,
) -> pd.DataFrame:
    if pipeline_df.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, src in pipeline_df.iterrows():
        repeat_dir = _repeat_dir(output_base, str(src.get("run_dir") or ""))
        if repeat_dir is None:
            continue
        architecture = architecture_label(src.get("family", ""))
        representation = representation_label(src.get("archive_layout", ""))
        condition_slug = str(src.get("condition") or "")
        for split_src, split_name in PREDICTION_FILES:
            path = repeat_dir / f"predictions_{split_src}.csv"
            if not path.is_file():
                continue
            part = pd.read_csv(path)
            if part.empty:
                continue
            part = part.copy()
            part["architecture"] = architecture
            part["representation"] = representation
            part["condition_slug"] = condition_slug
            part["condition"] = condition_abbrev(condition_slug)
            part["repeat"] = src.get("repeat_index", "")
            part["seed"] = src.get("split_seed", "")
            part["trial_id"] = src.get("nas_trial_id", "")
            part["best_epoch"] = src.get("best_epoch", "")
            part["model_config"] = src.get("model_config", "")
            part["split"] = split_name
            part["spinal_level"] = part["spinal_level"].map(
                lambda v: canonical_spinal_level(str(v))
            )
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    return out


def _drop_invalid_and_duplicates(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if predictions.empty:
        return predictions, warnings
    missing = predictions["patient_id"].isna() | predictions["y_true"].isna() | predictions["y_pred"].isna()
    if bool(missing.any()):
        n_drop = int(missing.sum())
        warnings.append(f"Dropped {n_drop} prediction rows with missing patient_id/y_true/y_pred")
        predictions = predictions.loc[~missing].copy()
    dup_mask = predictions.duplicated(list(UNIQUENESS_KEYS), keep=False)
    if bool(dup_mask.any()):
        n_dup = int(predictions.duplicated(list(UNIQUENESS_KEYS)).sum())
        warnings.append(
            f"Dropped {n_dup} duplicate prediction keys (kept first): "
            + ", ".join(UNIQUENESS_KEYS)
        )
        predictions = predictions.drop_duplicates(list(UNIQUENESS_KEYS), keep="first")
    return predictions.reset_index(drop=True), warnings


def leakage_warnings(predictions: pd.DataFrame) -> list[str]:
    warns: list[str] = []
    if predictions.empty:
        return warns
    keys = ["architecture", "representation", "repeat", "seed"]
    for key_tuple, group in predictions.groupby(keys, dropna=False, sort=False):
        by_split: dict[str, set[Any]] = {}
        for split, sub in group.groupby("split", dropna=False):
            by_split[str(split)] = set(pd.to_numeric(sub["patient_id"], errors="coerce").dropna().tolist())
        pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
        for a, b in pairs:
            overlap = by_split.get(a, set()) & by_split.get(b, set())
            if overlap:
                arch, rep, repeat, seed = key_tuple
                warns.append(
                    f"{arch} {rep} repeat {repeat} seed {seed}: "
                    f"{len(overlap)} patient_id(s) in both {a} and {b}"
                )
    return warns


def build_patient_runs(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    keys = ["architecture", "representation", "repeat", "seed", "split", "patient_id"]
    rows: list[dict[str, Any]] = []
    for key_tuple, group in predictions.groupby(keys, dropna=False, sort=False):
        metrics = compute_patient_metrics(group["y_true"].tolist(), group["y_pred"].tolist())
        rec = dict(zip(keys, key_tuple))
        rec.update(metrics)
        rec["trial_id"] = group["trial_id"].iloc[0] if "trial_id" in group.columns else ""
        rec["best_epoch"] = group["best_epoch"].iloc[0] if "best_epoch" in group.columns else ""
        rec["n_conditions"] = int(group["condition"].nunique()) if "condition" in group.columns else 0
        rec["n_levels"] = int(group["spinal_level"].nunique()) if "spinal_level" in group.columns else 0
        rows.append(rec)
    out = pd.DataFrame(rows)
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    out["_s"] = out["split"].map(split_rank)
    return (
        out.sort_values(["architecture", "representation", "repeat", "_s", "patient_id"], kind="mergesort")
        .drop(columns=["_s"])
        .reset_index(drop=True)
    )


def summarize_repeats(patient_runs: pd.DataFrame) -> pd.DataFrame:
    if patient_runs.empty:
        return pd.DataFrame()
    keys = ["architecture", "representation", "repeat", "seed", "split"]
    rows: list[dict[str, Any]] = []
    for key_tuple, group in patient_runs.groupby(keys, dropna=False, sort=False):
        profiles = group["error_profile"].astype(str)
        n_patients = int(len(group))
        exact_mean, exact_std = _mean_std(group["patient_exact_rate"].tolist())
        omae_mean, omae_std = _mean_std(group["patient_omae"].tolist())
        rec = dict(zip(keys, key_tuple))
        rec.update(
            {
                "n_patients": n_patients,
                "patient_macro_exact_rate": exact_mean,
                "within_repeat_patient_exact_std": exact_std,
                "patient_macro_omae": omae_mean,
                "within_repeat_patient_omae_std": omae_std,
                "patient_exact_profile_rate": float((profiles == "exact").mean()) if n_patients else float("nan"),
                "patient_adjacent_only_rate": float((profiles == "adjacent_only").mean()) if n_patients else float("nan"),
                "patient_severe_error_rate": float((profiles == "severe").mean()) if n_patients else float("nan"),
                "mean_n_errors_per_patient": float(group["n_errors"].mean()),
                "median_n_errors_per_patient": float(group["n_errors"].median()),
                "mean_n_severe_errors_per_patient": float(group["n_severe_errors"].mean()),
                "max_n_severe_errors_per_patient": int(group["n_severe_errors"].max()),
                "mean_n_predictions_per_patient": float(group["n_predictions"].mean()),
                "median_n_predictions_per_patient": float(group["n_predictions"].median()),
                "min_n_predictions_per_patient": int(group["n_predictions"].min()),
                "max_n_predictions_per_patient": int(group["n_predictions"].max()),
                "pooled_exact_rate": float(group["n_exact"].sum() / group["n_predictions"].sum())
                if group["n_predictions"].sum()
                else float("nan"),
                "pooled_omae": float(
                    (group["patient_omae"] * group["n_predictions"]).sum() / group["n_predictions"].sum()
                )
                if group["n_predictions"].sum()
                else float("nan"),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    out["_s"] = out["split"].map(split_rank)
    return (
        out.sort_values(["architecture", "representation", "repeat", "_s"], kind="mergesort")
        .drop(columns=["_s"])
        .reset_index(drop=True)
    )


def _incomplete_status(n_repeats: int) -> str:
    return "complete" if n_repeats >= EXPECTED_REPEATS else "incomplete"


def summarize_across_repeats(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    if repeat_summary.empty:
        return pd.DataFrame()
    keys = ["architecture", "representation", "split"]
    rows: list[dict[str, Any]] = []
    for key_tuple, group in repeat_summary.groupby(keys, dropna=False, sort=False):
        n_repeats = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        rec: dict[str, Any] = {
            **dict(zip(keys, key_tuple)),
            "n_repeats": n_repeats,
            "status": _incomplete_status(n_repeats),
        }
        for metric in REPEAT_AGG_METRICS:
            if metric not in group.columns:
                rec[f"{metric}_mean"] = float("nan")
                rec[f"{metric}_std"] = float("nan")
                continue
            mean, std = _mean_std(group[metric].tolist())
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
        rows.append(rec)
    out = pd.DataFrame(rows)
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    out["_s"] = out["split"].map(split_rank)
    return (
        out.sort_values(["architecture", "representation", "_s"], kind="mergesort")
        .drop(columns=["_s"])
        .reset_index(drop=True)
    )


def summarize_by_condition(predictions: pd.DataFrame, patient_runs: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    keys = ["architecture", "representation", "condition", "repeat", "seed", "split", "patient_id"]
    per_patient: list[dict[str, Any]] = []
    for key_tuple, group in predictions.groupby(keys, dropna=False, sort=False):
        metrics = compute_patient_metrics(group["y_true"].tolist(), group["y_pred"].tolist())
        rec = dict(zip(keys, key_tuple))
        rec.update(metrics)
        per_patient.append(rec)
    by_patient = pd.DataFrame(per_patient)
    out_keys = ["architecture", "representation", "condition", "split"]
    rows: list[dict[str, Any]] = []
    for key_tuple, group in by_patient.groupby(out_keys, dropna=False, sort=False):
        n_repeats = int(group["seed"].nunique()) if "seed" in group.columns else 0
        exact_mean, exact_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["patient_exact_rate"].mean().tolist()
        )
        omae_mean, omae_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["patient_omae"].mean().tolist()
        )
        severe_mean, severe_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["has_severe_error"].mean().tolist()
        )
        rec = dict(zip(out_keys, key_tuple))
        rec.update(
            {
                "n_repeats": n_repeats,
                "status": _incomplete_status(n_repeats),
                "patient_macro_exact_rate_mean": exact_mean,
                "patient_macro_exact_rate_std": exact_std,
                "patient_macro_omae_mean": omae_mean,
                "patient_macro_omae_std": omae_std,
                "patient_severe_error_rate_mean": severe_mean,
                "patient_severe_error_rate_std": severe_std,
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    out["_c"] = out["condition"].map(cond_rank)
    out["_s"] = out["split"].map(split_rank)
    return (
        out.sort_values(["architecture", "representation", "_c", "_s"], kind="mergesort")
        .drop(columns=["_c", "_s"])
        .reset_index(drop=True)
    )


def summarize_by_spinal_level(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    keys = ["architecture", "representation", "spinal_level", "repeat", "seed", "split", "patient_id"]
    per_patient: list[dict[str, Any]] = []
    for key_tuple, group in predictions.groupby(keys, dropna=False, sort=False):
        metrics = compute_patient_metrics(group["y_true"].tolist(), group["y_pred"].tolist())
        rec = dict(zip(keys, key_tuple))
        rec.update(metrics)
        per_patient.append(rec)
    by_patient = pd.DataFrame(per_patient)
    out_keys = ["architecture", "representation", "spinal_level", "split"]
    rows: list[dict[str, Any]] = []
    for key_tuple, group in by_patient.groupby(out_keys, dropna=False, sort=False):
        n_repeats = int(group["seed"].nunique()) if "seed" in group.columns else 0
        exact_mean, exact_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["patient_exact_rate"].mean().tolist()
        )
        omae_mean, omae_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["patient_omae"].mean().tolist()
        )
        severe_mean, severe_std = _mean_std(
            group.groupby(["repeat", "seed"], dropna=False)["has_severe_error"].mean().tolist()
        )
        rec = dict(zip(out_keys, key_tuple))
        rec.update(
            {
                "n_repeats": n_repeats,
                "status": _incomplete_status(n_repeats),
                "patient_macro_exact_rate_mean": exact_mean,
                "patient_macro_exact_rate_std": exact_std,
                "patient_macro_omae_mean": omae_mean,
                "patient_macro_omae_std": omae_std,
                "patient_severe_error_rate_mean": severe_mean,
                "patient_severe_error_rate_std": severe_std,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def build_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keep = summary[summary["split"].isin(PAPER_SPLITS)].copy()
    cols = [
        "split",
        "representation",
        "architecture",
        "patient_macro_exact_rate_mean",
        "patient_macro_exact_rate_std",
        "patient_macro_omae_mean",
        "patient_macro_omae_std",
        "patient_exact_profile_rate_mean",
        "patient_exact_profile_rate_std",
        "patient_adjacent_only_rate_mean",
        "patient_adjacent_only_rate_std",
        "patient_severe_error_rate_mean",
        "patient_severe_error_rate_std",
        "status",
    ]
    return keep[[c for c in cols if c in keep.columns]].reset_index(drop=True)


def build_error_profile_frame(summary: pd.DataFrame, *, split: str) -> pd.DataFrame:
    lookup = {
        (str(r["architecture"]), str(r["representation"])): r
        for _, r in summary[summary["split"] == split].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for arch, rep in PIPELINE_ROW_ORDER:
        hit = lookup.get((arch, rep))
        rec: dict[str, Any] = {"architecture": arch, "representation": rep}
        if hit is None:
            rec.update({"exact_rate": float("nan"), "adjacent_only_rate": float("nan"), "severe_rate": float("nan")})
        else:
            rec["exact_rate"] = hit.get("patient_exact_profile_rate_mean", float("nan"))
            rec["adjacent_only_rate"] = hit.get("patient_adjacent_only_rate_mean", float("nan"))
            rec["severe_rate"] = hit.get("patient_severe_error_rate_mean", float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_distribution_frame(patient_runs: pd.DataFrame, *, split: str, metric: str) -> pd.DataFrame:
    keep = patient_runs[patient_runs["split"] == split]
    cols = ["architecture", "representation", "repeat", "seed", "patient_id", metric]
    return keep[[c for c in cols if c in keep.columns]].reset_index(drop=True)


def _incomplete_from_pipeline(pipeline_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if pipeline_df is None or pipeline_df.empty:
        return []
    incomplete: list[dict[str, Any]] = []
    work = pipeline_df.copy()
    work["architecture"] = work["family"].map(architecture_label)
    work["representation"] = work["archive_layout"].map(representation_label)
    work["condition_abbrev"] = work["condition"].map(condition_abbrev)
    for (arch, rep, cond), group in work.groupby(["architecture", "representation", "condition_abbrev"], dropna=False):
        n = int(group["split_seed"].nunique()) if "split_seed" in group.columns else 0
        if n >= EXPECTED_REPEATS:
            continue
        incomplete.append(
            {
                "architecture": arch,
                "representation": rep,
                "affected_condition": cond,
                "n_repeats": n,
                "expected_repeats": EXPECTED_REPEATS,
                "missing_repeat_count": EXPECTED_REPEATS - n,
            }
        )
    return incomplete


def build_summary_json(
    patient_runs: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    prediction_warnings: list[str],
    leakage: list[str],
    incomplete: list[dict[str, Any]],
) -> dict[str, Any]:
    n_runs = 0
    if not patient_runs.empty:
        n_runs = int(
            patient_runs.drop_duplicates(["architecture", "representation", "seed"]).shape[0]
        )
    pipelines: list[dict[str, Any]] = []
    if not summary.empty:
        val = summary[summary["split"] == "validation"]
        for _, row in val.iterrows():
            pipelines.append(
                {
                    "architecture": row["architecture"],
                    "representation": row["representation"],
                    "split": "validation",
                    "n_repeats": int(row["n_repeats"]),
                    "status": row.get("status", ""),
                    "patient_macro_exact_rate": {
                        "mean": row.get("patient_macro_exact_rate_mean"),
                        "std": row.get("patient_macro_exact_rate_std"),
                    },
                    "patient_macro_omae": {
                        "mean": row.get("patient_macro_omae_mean"),
                        "std": row.get("patient_macro_omae_std"),
                    },
                    "patient_error_profile": {
                        "exact": {
                            "mean": row.get("patient_exact_profile_rate_mean"),
                            "std": row.get("patient_exact_profile_rate_std"),
                        },
                        "adjacent_only": {
                            "mean": row.get("patient_adjacent_only_rate_mean"),
                            "std": row.get("patient_adjacent_only_rate_std"),
                        },
                        "severe": {
                            "mean": row.get("patient_severe_error_rate_mean"),
                            "std": row.get("patient_severe_error_rate_std"),
                        },
                    },
                }
            )
    return {
        "n_runs_expected": EXPECTED_RUNS,
        "n_architecture_representation_seeds": n_runs,
        "ploa_implemented": False,
        "incomplete_groups": incomplete,
        "prediction_warnings": prediction_warnings,
        "leakage_warnings": leakage,
        "pipelines": pipelines,
    }


def print_terminal_summary(summary: pd.DataFrame, *, split: str = "validation") -> None:
    sub = summary[summary["split"] == split]
    for _, row in sub.iterrows():
        print(
            f"\nArchitecture: {row['architecture']}\n"
            f"Representation: {row['representation']}\n"
            f"Split: {split}\n\n"
            f"Completed repeats: {int(row['n_repeats'])}/{EXPECTED_REPEATS}\n\n"
            f"Patient macro exact-match rate: {_format_pm(row['patient_macro_exact_rate_mean'], row['patient_macro_exact_rate_std'])}\n"
            f"Patient macro O-MAE: {_format_pm(row['patient_macro_omae_mean'], row['patient_macro_omae_std'])}\n\n"
            f"All-exact patients: {_format_pm(row['patient_exact_profile_rate_mean'], row['patient_exact_profile_rate_std'])}\n"
            f"Adjacent-only patients: {_format_pm(row['patient_adjacent_only_rate_mean'], row['patient_adjacent_only_rate_std'])}\n"
            f"Patients with >=1 severe error: {_format_pm(row['patient_severe_error_rate_mean'], row['patient_severe_error_rate_std'])}\n\n"
            f"Mean errors per patient: {row.get('mean_n_errors_per_patient_mean', '')}\n"
            f"Mean severe errors per patient: {row.get('mean_n_severe_errors_per_patient_mean', '')}"
        )


def write_patient_level_outputs(
    output_base: Path,
    *,
    pipeline_df: pd.DataFrame | None = None,
    prediction_df: pd.DataFrame | None = None,
    print_summary: bool = True,
) -> dict[str, Path]:
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    df = pipeline_df if pipeline_df is not None else load_all_pipeline_results(output_base)
    if prediction_df is not None:
        predictions = prediction_df.copy()
    else:
        predictions = load_prediction_records(df, output_base=output_base)
    if predictions.empty:
        raise ValueError(
            f"No predictions_{{train,val,test}}.csv under repeat dirs in {output_base}"
        )

    predictions, pred_warnings = _drop_invalid_and_duplicates(predictions)
    leak = leakage_warnings(predictions)
    patient_runs = build_patient_runs(predictions)
    repeat_summary = summarize_repeats(patient_runs)
    summary = summarize_across_repeats(repeat_summary)
    by_condition = summarize_by_condition(predictions, patient_runs)
    by_level = summarize_by_spinal_level(predictions)
    paper = build_paper_table(summary)
    incomplete = _incomplete_from_pipeline(df)
    payload = build_summary_json(
        patient_runs,
        summary,
        prediction_warnings=pred_warnings,
        leakage=leak,
        incomplete=incomplete,
    )

    paths: dict[str, Path] = {}
    paths["runs"] = output_base / OUTPUT_RUNS
    patient_runs.to_csv(paths["runs"], index=False)
    paths["repeat_summary"] = output_base / OUTPUT_REPEAT
    repeat_summary.to_csv(paths["repeat_summary"], index=False)
    paths["summary"] = output_base / OUTPUT_SUMMARY
    summary.to_csv(paths["summary"], index=False)
    paths["paper_table"] = output_base / OUTPUT_PAPER
    paper.to_csv(paths["paper_table"], index=False)
    paths["by_condition"] = output_base / OUTPUT_BY_CONDITION
    by_condition.to_csv(paths["by_condition"], index=False)
    paths["by_spinal_level"] = output_base / OUTPUT_BY_LEVEL
    by_level.to_csv(paths["by_spinal_level"], index=False)

    for split in PAPER_SPLITS:
        profile = build_error_profile_frame(summary, split=split)
        path = output_base / f"patient_error_profile_{split}.csv"
        profile.to_csv(path, index=False)
        paths[f"error_profile_{split}"] = path
        exact_dist = build_distribution_frame(patient_runs, split=split, metric="patient_exact_rate")
        path = output_base / f"patient_exact_rate_distribution_{split}.csv"
        exact_dist.to_csv(path, index=False)
        paths[f"exact_dist_{split}"] = path
        omae_dist = build_distribution_frame(patient_runs, split=split, metric="patient_omae")
        path = output_base / f"patient_omae_distribution_{split}.csv"
        omae_dist.to_csv(path, index=False)
        paths[f"omae_dist_{split}"] = path
        severe = profile[["architecture", "representation", "severe_rate"]].rename(
            columns={"severe_rate": "patient_severe_error_rate"}
        )
        path = output_base / f"patient_severe_error_rate_{split}.csv"
        severe.to_csv(path, index=False)
        paths[f"severe_rate_{split}"] = path

    paths["summary_json"] = output_base / OUTPUT_JSON
    paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for msg in pred_warnings + leak:
        print(f"WARNING: {msg}")
    if print_summary:
        print_terminal_summary(summary)
    return paths
