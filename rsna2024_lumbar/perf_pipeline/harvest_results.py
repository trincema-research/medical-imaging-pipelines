"""Build pipeline_results.csv from repeat_*/training_metrics.csv (e.g. after cloud download)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna2024_lumbar.perf_pipeline.config import list_best_config_files, load_best_config
from rsna2024_lumbar.perf_pipeline.paths import DEFAULT_BEST_CONFIG_DIR, RESULTS_DIR
from rsna2024_lumbar.perf_pipeline.results import row_from_training_metrics, write_pipeline_results_csv


def harvest_model_dir(
    model_dir: Path,
    *,
    config_stem: str,
    model_label: str,
    best_config_path: Path,
) -> list[dict]:
    rows: list[dict] = []
    if not model_dir.is_dir():
        return rows
    for metrics_path in sorted(model_dir.rglob("training_metrics.csv")):
        repeat_dir = metrics_path.parent
        if repeat_dir.name.startswith("repeat_") is False:
            continue
        condition = repeat_dir.parent.name
        repeat_index = int(repeat_dir.name.replace("repeat_", ""))
        manifest_path = repeat_dir / "pipeline_run_manifest.json"
        family = ""
        archive_layout = ""
        split_seed = ""
        trial_id = ""
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            family = manifest.get("family", "")
            split_seed = manifest.get("split_seed", "")
            trial_id = manifest.get("nas_trial_id", "")
            hp = manifest.get("hyperparameters") or {}
            archive_layout = hp.get("input_layout", "")
        meta = {
            "model_config": config_stem,
            "model_label": model_label,
            "family": family,
            "archive_layout": archive_layout,
            "condition": condition,
            "repeat_index": repeat_index,
            "split_seed": split_seed,
            "nas_trial_id": trial_id,
        }
        row = row_from_training_metrics(metrics_path, meta=meta)
        row["run_dir"] = f"{config_stem}/{condition}/repeat_{repeat_index:02d}"
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("condition", "")), int(r.get("repeat_index", 0))))
    return rows


def harvest_tree(
    results_root: Path,
    *,
    config_dir: Path,
    output_base: Path,
) -> list[dict]:
    combined: list[dict] = []
    for cfg_path in list_best_config_files(config_dir):
        config = load_best_config(cfg_path)
        model_dir = results_root / cfg_path.stem
        rows = harvest_model_dir(
            model_dir,
            config_stem=cfg_path.stem,
            model_label=str(config.get("label") or cfg_path.stem),
            best_config_path=cfg_path,
        )
        if not rows:
            print(f"Skip {cfg_path.stem}: no training_metrics.csv")
            continue
        out_dir = output_base / cfg_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        write_pipeline_results_csv(rows, out_dir / "pipeline_results.csv")
        summary = {
            "best_config": str(cfg_path.resolve()),
            "model_label": config.get("label") or cfg_path.stem,
            "repeats": 5,
            "run_count": len(rows),
            "harvested_from": str(results_root.resolve()),
        }
        (out_dir / "pipeline_run_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(f"{cfg_path.stem}: {len(rows)} rows -> {out_dir / 'pipeline_results.csv'}")
        combined.extend(rows)
    if combined:
        combined_path = output_base / "pipeline_results_all_models.csv"
        write_pipeline_results_csv(combined, combined_path)
        print(f"Combined: {len(combined)} rows -> {combined_path}")
    return combined


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Harvest perf_pipeline summary CSVs from run folders.")
    p.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing nas_best_* model result folders.",
    )
    p.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    p.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    harvest_tree(
        args.results_root.resolve(),
        config_dir=args.config_dir.resolve(),
        output_base=args.output_base.resolve(),
    )


if __name__ == "__main__":
    main()
