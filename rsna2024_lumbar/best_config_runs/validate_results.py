"""Validators for harvested pipeline CSVs, article tables, and repeat audit JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from rsna2024_lumbar.best_config_runs.article_summary import (
    ARTICLE_METRIC_COLUMNS,
    GROUP_KEYS,
    aggregate_pipeline_results,
)
from rsna2024_lumbar.best_config_runs.paths import DEFAULT_REPEATS, DEFAULT_SPLIT_SEEDS, RESULTS_DIR
from rsna2024_lumbar.best_config_runs.results import PIPELINE_RESULTS_COLUMNS
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

LUMBAR_CONDITIONS: tuple[str, ...] = tuple(sorted(CONDITIONS.keys()))

# Models expected to have full 5×5 retrain grids after the clean cloud run.
FULL_RETRAIN_MODELS: frozenset[str] = frozenset(
    {
        "nas_best_convnext_2d",
        "nas_best_convnext3d_3d",
        "nas_best_efficientnet_2d",
        "nas_best_efficientnet3d_3d",
        "nas_best_vit_2d",
        "nas_best_vit3d_3d",
    }
)

SEVERITY_OVERALL_COLUMNS: tuple[str, ...] = tuple(
    f"{split}_{m}_overall"
    for split in ("train", "val", "test")
    for m in ("oa", "omae", "qwk", "ser")
)

PROBABILITY_METRICS: tuple[str, ...] = (
    "train_acc",
    "val_acc",
    "test_acc",
    "train_oa_overall",
    "val_oa_overall",
    "test_oa_overall",
)


@dataclass
class ValidationIssue:
    level: Literal["error", "warning"]
    code: str
    message: str
    path: str = ""


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def error(self, code: str, message: str, *, path: str = "") -> None:
        self.issues.append(ValidationIssue("error", code, message, path))

    def warning(self, code: str, message: str, *, path: str = "") -> None:
        self.issues.append(ValidationIssue("warning", code, message, path))


def _is_finite_number(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def validate_pipeline_results_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "pipeline_results",
    require_severity: bool = True,
    expected_repeats: int = DEFAULT_REPEATS,
    strict_full_models: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    if df.empty:
        report.error("empty", f"{source}: no rows")
        return report

    missing_cols = [c for c in PIPELINE_RESULTS_COLUMNS if c not in df.columns]
    if missing_cols:
        report.error("columns", f"{source}: missing columns: {', '.join(missing_cols)}")
        return report

    extra = [c for c in df.columns if c not in PIPELINE_RESULTS_COLUMNS]
    if extra:
        report.warning("extra_columns", f"{source}: unexpected columns: {', '.join(extra)}")

    if df.duplicated(subset=["model_config", "condition", "repeat_index"]).any():
        dupes = df[df.duplicated(subset=["model_config", "condition", "repeat_index"], keep=False)]
        report.error(
            "duplicate_rows",
            f"{source}: duplicate model/condition/repeat keys ({len(dupes)} rows)",
        )

    for idx, row in df.iterrows():
        prefix = f"{source} row {idx}"
        for col in ("best_epoch", "repeat_index"):
            if not _is_finite_number(row[col]):
                report.error("missing_field", f"{prefix}: invalid {col}", path=str(row.get("run_dir", "")))
        for col in PROBABILITY_METRICS:
            if col in df.columns and not _is_finite_number(row.get(col)):
                report.error("missing_metric", f"{prefix}: missing or non-finite {col}", path=str(row.get("run_dir", "")))
        if require_severity:
            for col in SEVERITY_OVERALL_COLUMNS:
                if not _is_finite_number(row.get(col)):
                    report.error(
                        "missing_severity",
                        f"{prefix}: missing or non-finite {col} (train with --log-pipeline-metrics)",
                        path=str(row.get("run_dir", "")),
                    )
                    break
        for split in ("train", "val", "test"):
            acc_key = f"{split}_accuracy_overall"
            oa_key = f"{split}_oa_overall"
            if acc_key in df.columns and oa_key in df.columns:
                acc_v = row.get(acc_key)
                oa_v = row.get(oa_key)
                if _is_finite_number(acc_v) and _is_finite_number(oa_v):
                    if abs(float(acc_v) - float(oa_v)) > 1e-9:
                        report.error(
                            "oa_accuracy_mismatch",
                            f"{prefix}: {oa_key}={oa_v} != {acc_key}={acc_v}",
                            path=str(row.get("run_dir", "")),
                        )

        repeat_index = int(row["repeat_index"])
        if 1 <= repeat_index <= len(DEFAULT_SPLIT_SEEDS):
            expected_seed = DEFAULT_SPLIT_SEEDS[repeat_index - 1]
            seed_val = row.get("split_seed", "")
            if str(seed_val) != str(expected_seed) and _is_finite_number(seed_val):
                if int(seed_val) != expected_seed:
                    report.warning(
                        "split_seed",
                        f"{prefix}: split_seed {seed_val} != protocol {expected_seed}",
                        path=str(row.get("run_dir", "")),
                    )

    unknown = df[~df["condition"].isin(LUMBAR_CONDITIONS)]
    if not unknown.empty:
        report.error(
            "unknown_condition",
            f"{source}: unknown condition values: {sorted(unknown['condition'].unique())}",
        )

    for model_config, group in df.groupby("model_config", sort=True):
        for condition in LUMBAR_CONDITIONS:
            sub = group[group["condition"] == condition]
            n = len(sub)
            if n == 0:
                report.warning(
                    "missing_condition",
                    f"{model_config}/{condition}: no harvested rows",
                )
                continue
            if n != expected_repeats:
                level = "error" if strict_full_models and model_config in FULL_RETRAIN_MODELS else "warning"
                msg = f"{model_config}/{condition}: expected {expected_repeats} repeats, got {n}"
                if level == "error":
                    report.error("incomplete_repeats", msg)
                else:
                    report.warning("incomplete_repeats", msg)

    return report


def validate_pipeline_results_file(path: Path, **kwargs: Any) -> ValidationReport:
    path = path.resolve()
    if not path.is_file():
        report = ValidationReport()
        report.error("missing_file", f"pipeline_results not found: {path}", path=str(path))
        return report
    df = pd.read_csv(path)
    return validate_pipeline_results_dataframe(df, source=str(path), **kwargs)


def validate_results_tree(
    output_base: Path,
    *,
    require_severity: bool = True,
    strict_full_models: bool = False,
    check_audit_files: bool = True,
    atol: float = 1e-5,
) -> ValidationReport:
    """Validate per-model CSVs, combined CSV, article tables, and optional repeat JSON."""
    output_base = output_base.resolve()
    report = ValidationReport()

    per_model_paths = sorted(output_base.glob("nas_best_*/pipeline_results.csv"))
    if not per_model_paths:
        report.error("no_models", f"No nas_best_*/pipeline_results.csv under {output_base}")
        return report

    frames: list[pd.DataFrame] = []
    for path in per_model_paths:
        part = pd.read_csv(path)
        frames.append(part)
        sub = validate_pipeline_results_file(
            path,
            require_severity=require_severity,
            strict_full_models=strict_full_models,
        )
        report.issues.extend(sub.issues)

    all_rows = pd.concat(frames, ignore_index=True)
    combined_path = output_base / "pipeline_results_all_models.csv"
    if combined_path.is_file():
        combined = pd.read_csv(combined_path)
        if len(combined) != len(all_rows):
            report.error(
                "combined_count",
                f"{combined_path.name}: {len(combined)} rows != sum of per-model ({len(all_rows)})",
                path=str(combined_path),
            )
        else:
            key_cols = ["model_config", "condition", "repeat_index"]
            left = combined.sort_values(key_cols).reset_index(drop=True)
            right = all_rows.sort_values(key_cols).reset_index(drop=True)
            for col in ("val_acc", "test_oa_overall", "best_epoch"):
                if col not in left.columns:
                    continue
                a = pd.to_numeric(left[col], errors="coerce")
                b = pd.to_numeric(right[col], errors="coerce")
                if not np.allclose(a, b, rtol=0, atol=atol, equal_nan=True):
                    report.error(
                        "combined_mismatch",
                        f"combined CSV disagrees with per-model on {col}",
                        path=str(combined_path),
                    )
                    break
    else:
        report.warning("missing_combined", f"Missing {combined_path.name}")

    wide_path = output_base / "article_summary_by_condition.csv"
    if wide_path.is_file():
        report.issues.extend(
            _validate_article_summary(all_rows, wide_path, atol=atol).issues
        )
    else:
        report.warning("missing_article", f"Missing {wide_path.name}")

    if check_audit_files:
        report.issues.extend(_validate_repeat_audit_files(output_base).issues)

    return report


def _validate_article_summary(
    pipeline_df: pd.DataFrame,
    wide_path: Path,
    *,
    atol: float,
) -> ValidationReport:
    report = ValidationReport()
    wide = pd.read_csv(wide_path)
    if wide.empty:
        report.error("article_empty", f"Empty article summary: {wide_path}", path=str(wide_path))
        return report

    expected = aggregate_pipeline_results(pipeline_df)
    if len(wide) != len(expected):
        report.error(
            "article_row_count",
            f"article rows {len(wide)} != recomputed {len(expected)}",
            path=str(wide_path),
        )

    key_cols = list(GROUP_KEYS)
    wide_idx = wide.set_index(key_cols)
    exp_idx = expected.set_index(key_cols)
    for key in exp_idx.index:
        if key not in wide_idx.index:
            report.error("article_missing_group", f"article summary missing {key}", path=str(wide_path))
            continue
        w = wide_idx.loc[key]
        e = exp_idx.loc[key]
        if int(w["n_repeats"]) != int(e["n_repeats"]):
            report.error(
                "article_n_repeats",
                f"{key}: n_repeats {w['n_repeats']} != {e['n_repeats']}",
                path=str(wide_path),
            )
        for col in ("val_acc", "test_oa_overall", "val_omae_overall"):
            mean_col = f"{col}_mean"
            if mean_col not in w.index:
                continue
            if not _is_finite_number(w[mean_col]) or not _is_finite_number(e[mean_col]):
                continue
            if abs(float(w[mean_col]) - float(e[mean_col])) > atol:
                report.error(
                    "article_mean_drift",
                    f"{key}: {mean_col} file={w[mean_col]} recomputed={e[mean_col]}",
                    path=str(wide_path),
                )

    long_path = wide_path.parent / "article_summary_metrics_long.csv"
    if not long_path.is_file():
        report.warning("missing_article_long", f"Missing {long_path.name}")
    return report


def _harvested_repeat_keys(output_base: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for path in output_base.glob("nas_best_*/pipeline_results.csv"):
        model_config = path.parent.name
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            keys.add((model_config, str(row["condition"]), int(row["repeat_index"])))
    return keys


def _validate_repeat_audit_files(output_base: Path) -> ValidationReport:
    """Validate run_config + training_history for repeats present in pipeline_results."""
    report = ValidationReport()
    for model_config, condition, repeat_index in sorted(_harvested_repeat_keys(output_base)):
        repeat_dir = (
            output_base / model_config / condition / f"repeat_{repeat_index:02d}"
        )
        rel = repeat_dir.relative_to(output_base).as_posix()
        run_config_path = repeat_dir / "run_config.json"
        if not run_config_path.is_file():
            report.error("missing_run_config", f"{rel}: no run_config.json", path=rel)
            continue

        try:
            cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error("run_config_json", f"{rel}: invalid JSON: {exc}", path=rel)
            continue

        if cfg.get("condition") != condition:
            report.error(
                "run_config_condition",
                f"{rel}: condition {cfg.get('condition')!r} != folder {condition!r}",
                path=rel,
            )

        if 1 <= repeat_index <= len(DEFAULT_SPLIT_SEEDS):
            expected_seed = DEFAULT_SPLIT_SEEDS[repeat_index - 1]
            seed = cfg.get("seed")
            if seed is not None and int(seed) != expected_seed:
                report.warning(
                    "run_config_seed",
                    f"{rel}: seed {seed} != protocol {expected_seed}",
                    path=rel,
                )

        history_path = repeat_dir / "training_history.json"
        if not history_path.is_file():
            report.error("missing_history", f"{rel}: no training_history.json", path=rel)
            continue

        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error("history_json", f"{rel}: invalid training_history.json: {exc}", path=rel)
            continue

        if not isinstance(history, list) or not history:
            report.error("history_empty", f"{rel}: training_history must be a non-empty list", path=rel)
            continue

        pipeline_path = output_base / model_config / "pipeline_results.csv"
        pdf = pd.read_csv(pipeline_path)
        row = pdf[(pdf["condition"] == condition) & (pdf["repeat_index"] == repeat_index)]
        if len(row) == 1:
            best_epoch = int(row.iloc[0]["best_epoch"])
            by_val = max(
                (e for e in history if isinstance(e, dict) and "val_acc" in e),
                key=lambda e: float(e["val_acc"]),
                default=None,
            )
            if by_val is not None and int(by_val.get("epoch", -1)) != best_epoch:
                report.warning(
                    "history_best_epoch",
                    f"{rel}: history argmax val_acc epoch {by_val.get('epoch')} != "
                    f"pipeline best_epoch {best_epoch}",
                    path=rel,
                )

        last = history[-1]
        if isinstance(last, dict) and "val_oa_overall" not in last and "val_acc" in last:
            report.warning(
                "history_no_severity",
                f"{rel}: final epoch lacks val_oa_overall (older bundle?)",
                path=rel,
            )

    return report


def format_report(report: ValidationReport) -> str:
    if not report.issues:
        return "OK: no issues"
    lines: list[str] = []
    for issue in report.issues:
        loc = f" ({issue.path})" if issue.path else ""
        lines.append(f"[{issue.level.upper()}] {issue.code}{loc}: {issue.message}")
    errors = sum(1 for i in report.issues if i.level == "error")
    warnings = sum(1 for i in report.issues if i.level == "warning")
    lines.append(f"Summary: {errors} error(s), {warnings} warning(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate best_config_runs result CSVs and repeat audit JSON.")
    p.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    p.add_argument(
        "--strict-full-models",
        action="store_true",
        help=f"Treat incomplete grids on {sorted(FULL_RETRAIN_MODELS)} as errors.",
    )
    p.add_argument(
        "--allow-missing-severity",
        action="store_true",
        help="Do not require OA/O-MAE/QWK/SER columns to be populated.",
    )
    p.add_argument("--no-audit-files", action="store_true", help="Skip training_history/run_config checks.")
    args = p.parse_args(argv)

    report = validate_results_tree(
        args.output_base.resolve(),
        require_severity=not args.allow_missing_severity,
        strict_full_models=args.strict_full_models,
        check_audit_files=not args.no_audit_files,
    )
    print(format_report(report))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
