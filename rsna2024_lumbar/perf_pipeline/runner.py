"""Run the post-NAS performance pipeline (5 repeats × best config per condition)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rsna2024_lumbar.perf_pipeline.config import iter_entries, load_best_config
from rsna2024_lumbar.perf_pipeline.paths import (
    DEFAULT_SPLIT_SEEDS,
    model_results_dir,
    run_output_dir,
)
from rsna2024_lumbar.perf_pipeline.results import write_per_run_best_epoch_csv, write_pipeline_results_csv
from rsna2024_lumbar.perf_pipeline.train import run_training


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
) -> tuple[int, list[dict[str, Any]]]:
    config = load_best_config(best_config_path)
    config_stem = best_config_path.stem
    model_label = config.get("label") or config_stem
    out_root = model_results_dir(config_stem, output_base=output_base)
    repeat_seeds = resolve_repeat_seeds(repeats, seeds)

    all_rows: list[dict[str, Any]] = []
    rc = 0

    for entry in iter_entries(config, condition=condition):
        for repeat_index, split_seed in enumerate(repeat_seeds, start=1):
            run_dir = run_output_dir(
                config_stem,
                str(entry["condition"]),
                repeat_index,
                output_base=output_base,
            )
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
            )
            if code != 0:
                rc = code
                continue
            if dry_run:
                continue
            metrics_path = run_dir / "training_metrics.csv"
            if not metrics_path.is_file():
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
            row = write_per_run_best_epoch_csv(
                metrics_path,
                run_dir / "best_epoch_results.csv",
                meta=meta,
            )
            all_rows.append(row)

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
