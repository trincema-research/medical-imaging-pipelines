"""Run the post-NAS performance pipeline (5 repeats × best config per condition)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from rsna2024_lumbar.best_config_runs.config import filter_config_paths, iter_entries, load_best_config
from rsna2024_lumbar.best_config_runs.jobs import PipelineJob, read_jobs_file
from rsna2024_lumbar.best_config_runs.paths import (
    DEFAULT_SPLIT_SEEDS,
    model_results_dir,
    run_output_dir,
)
from rsna2024_lumbar.best_config_runs.results import write_per_run_best_epoch_csv, write_pipeline_results_csv
from rsna2024_lumbar.best_config_runs.train import run_training


def run_is_complete(run_dir: Path, epochs: int) -> bool:
    if (run_dir / "best_epoch_results.csv").is_file():
        return True
    metrics_path = run_dir / "training_metrics.csv"
    if not metrics_path.is_file():
        return False
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return False
    if df.empty or "epoch" not in df.columns:
        return False
    if int(df["epoch"].max()) >= int(epochs):
        return True
    # Early-stopped runs write training_history.json when the loop exits.
    return (run_dir / "training_history.json").is_file()


def collect_model_result_rows(
    *,
    config_stem: str,
    config: dict[str, Any],
    model_label: str,
    repeat_seeds: list[int],
    epochs: int,
    output_base: Path | None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in iter_entries(config, condition=condition):
        for repeat_index, split_seed in enumerate(repeat_seeds, start=1):
            run_dir = run_output_dir(
                config_stem,
                str(entry["condition"]),
                repeat_index,
                output_base=output_base,
            )
            metrics_path = run_dir / "training_metrics.csv"
            if not run_is_complete(run_dir, epochs):
                continue
            meta = {
                "model_config": config_stem,
                "model_label": model_label,
                "family": entry.get("family", ""),
                "archive_layout": entry.get("archive_layout", ""),
                "condition": entry.get("condition", ""),
                "repeat_index": repeat_index,
                "split_seed": split_seed,
                "nas_trial_id": entry.get("trial_id", ""),
            }
            rows.append(
                write_per_run_best_epoch_csv(
                    metrics_path,
                    run_dir / "best_epoch_results.csv",
                    meta=meta,
                )
            )
    return rows


def resolve_repeat_seeds(
    repeats: int,
    seeds: Sequence[int] | None,
) -> list[int]:
    if seeds is not None:
        if len(seeds) < repeats:
            raise ValueError(f"Need {repeats} seeds, got {len(seeds)}")
        return list(seeds[:repeats])
    default = list(DEFAULT_SPLIT_SEEDS)
    if repeats <= len(default):
        return default[:repeats]
    extra = [default[-1] + 100 * i for i in range(1, repeats - len(default) + 1)]
    return default + extra


def run_pipeline_jobs(
    jobs: list[PipelineJob],
    *,
    data_root: Path | None = None,
    crops_root: Path | None = None,
    epochs: int = 50,
    output_base: Path | None = None,
    max_studies: int | None = None,
    dry_run: bool = False,
    early_stop_patience: int | None = None,
    save_checkpoints: bool = False,
) -> int:
    rc = 0
    for job in jobs:
        cfg_path = Path(job.best_config)
        config = load_best_config(cfg_path)
        entry = None
        for candidate in iter_entries(config, condition=job.condition):
            entry = candidate
            break
        if entry is None:
            raise ValueError(f"No entry for condition {job.condition!r} in {cfg_path}")
        run_dir = run_output_dir(
            cfg_path.stem,
            job.condition,
            job.repeat_index,
            output_base=output_base,
        )
        code = run_training(
            entry,
            source_config=cfg_path,
            output_dir=run_dir,
            data_root=data_root,
            crops_root=crops_root,
            epochs=epochs,
            max_studies=max_studies,
            split_seed=job.split_seed,
            seed=job.split_seed,
            repeat_index=job.repeat_index,
            dry_run=dry_run,
            early_stop_patience=early_stop_patience,
            save_checkpoints=save_checkpoints,
        )
        if code != 0:
            rc = code
    return rc


def run_jobs_file(
    jobs_path: Path,
    **kwargs: Any,
) -> int:
    return run_pipeline_jobs(read_jobs_file(jobs_path), **kwargs)


def finalize_all_model_summaries(
    config_dir: Path,
    *,
    repeats: int,
    seeds: Sequence[int] | None,
    only: list[str] | None,
    epochs: int,
    output_base: Path,
) -> None:
    """Write per-model pipeline_results.csv after parallel or sequential runs."""
    repeat_seeds = resolve_repeat_seeds(repeats, seeds)
    for cfg_path in filter_config_paths(config_dir, only):
        config = load_best_config(cfg_path)
        all_rows = collect_model_result_rows(
            config_stem=cfg_path.stem,
            config=config,
            model_label=str(config.get("label") or cfg_path.stem),
            repeat_seeds=repeat_seeds,
            epochs=epochs,
            output_base=output_base,
        )
        if not all_rows:
            continue
        out_root = model_results_dir(cfg_path.stem, output_base=output_base)
        write_pipeline_results_csv(all_rows, out_root / "pipeline_results.csv")
        with (out_root / "pipeline_run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "best_config": str(cfg_path.resolve()),
                    "model_label": config.get("label") or cfg_path.stem,
                    "repeats": repeats,
                    "seeds": repeat_seeds,
                    "run_count": len(all_rows),
                },
                fh,
                indent=2,
            )
        print(f"Wrote {out_root / 'pipeline_results.csv'} ({len(all_rows)} rows)")


def run_best_config_pipeline(
    best_config_path: Path,
    *,
    data_root: Path | None = None,
    crops_root: Path | None = None,
    epochs: int = 50,
    repeats: int = 5,
    seeds: Sequence[int] | None = None,
    condition: str | None = None,
    output_base: Path | None = None,
    max_studies: int | None = None,
    dry_run: bool = False,
    skip_completed: bool = False,
    early_stop_patience: int | None = None,
    save_checkpoints: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    config = load_best_config(best_config_path)
    config_stem = best_config_path.stem
    model_label = config.get("label") or config_stem
    out_root = model_results_dir(config_stem, output_base=output_base)
    repeat_seeds = resolve_repeat_seeds(repeats, seeds)

    rc = 0

    for entry in iter_entries(config, condition=condition):
        for repeat_index, split_seed in enumerate(repeat_seeds, start=1):
            run_dir = run_output_dir(
                config_stem,
                str(entry["condition"]),
                repeat_index,
                output_base=output_base,
            )
            if skip_completed and run_is_complete(run_dir, epochs):
                print(f"Skip completed run: {run_dir}")
                continue
            code = run_training(
                entry,
                source_config=best_config_path,
                output_dir=run_dir,
                data_root=data_root,
                crops_root=crops_root,
                epochs=epochs,
                max_studies=max_studies,
                split_seed=split_seed,
                seed=split_seed,
                repeat_index=repeat_index,
                dry_run=dry_run,
                early_stop_patience=early_stop_patience,
                save_checkpoints=save_checkpoints,
            )
            if code != 0:
                rc = code
                continue
            if dry_run:
                continue
            metrics_path = run_dir / "training_metrics.csv"
            if not metrics_path.is_file():
                continue

    if dry_run:
        return rc, []

    all_rows = collect_model_result_rows(
        config_stem=config_stem,
        config=config,
        model_label=model_label,
        repeat_seeds=repeat_seeds,
        epochs=epochs,
        output_base=output_base,
        condition=condition,
    )

    if all_rows and not dry_run:
        write_pipeline_results_csv(all_rows, out_root / "pipeline_results.csv")
        with (out_root / "pipeline_run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "best_config": str(best_config_path.resolve()),
                    "model_label": model_label,
                    "repeats": repeats,
                    "seeds": repeat_seeds,
                    "run_count": len(all_rows),
                },
                fh,
                indent=2,
            )
        print(f"Wrote {out_root / 'pipeline_results.csv'} ({len(all_rows)} rows)")
    return rc, all_rows
