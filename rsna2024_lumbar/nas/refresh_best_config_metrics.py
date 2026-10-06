"""Refresh best-config CSV/JSON with full NAS classification metrics from ``nas_compact`` histories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rsna2024_lumbar.nas.history_metrics import enrich_row_from_compact_history
from rsna2024_lumbar.nas.paths import YEAR_ROOT
from rsna2024_lumbar.nas.report import row_to_json_entry, write_best_csv, write_best_json


def _entry_to_row(entry: dict, *, document: dict) -> dict[str, Any]:

    row: dict[str, Any] = {
        "selection": entry.get("selection"),
        "model_group": entry.get("model_group"),
        "archive_layout": entry.get("archive_layout"),
        "archive_label": entry.get("archive_label"),
        "family": entry.get("family"),
        "results_root": document.get("archive_root", ""),
        "rank_metric": entry.get("rank_metric") or document.get("rank_metric"),
        "rank_metric_value": entry.get("rank_metric_value", ""),
        "condition": entry.get("condition"),
        "trial_id": entry.get("trial_id"),
        "grid_index": entry.get("grid_index", ""),
        "shared_mean_rank_metric": entry.get("shared_mean_rank_metric", ""),
        "trainable_params": entry.get("trainable_params", ""),
        "total_params": entry.get("total_params", ""),
        "checkpoint_path": entry.get("checkpoint_path", ""),
        "result_path": entry.get("result_path", ""),
        "trial_dir": entry.get("trial_dir", ""),
    }
    hp = entry.get("hyperparameters") or {}
    for key, value in hp.items():
        row[key] = value
    metrics = entry.get("metrics") or {}
    for key, value in metrics.items():
        row[key] = value
    return row


def refresh_json_path(
    json_path: Path,
    *,
    compact_root: Path,
) -> int:
    document = json.loads(json_path.read_text(encoding="utf-8"))
    entries = document.get("entries") or []
    rows: list[dict] = []
    for entry in entries:
        row = _entry_to_row(entry, document=document)
        enrich_row_from_compact_history(row, compact_root=compact_root)
        refreshed = row_to_json_entry(row)
        entry["metrics"] = refreshed["metrics"]
        rows.append(row)
    write_best_json(
        json_path,
        rows,
        rank_metric=str(document.get("rank_metric", "val_acc")),
        archive_root=document.get("archive_root"),
        label=str(document.get("label", "")),
    )
    csv_path = json_path.with_suffix(".csv")
    write_best_csv(csv_path, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Add NAS epoch classification metrics to best_configs from nas_compact histories.",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=YEAR_ROOT / "data" / "nas_compact" / "best_configs",
    )
    p.add_argument(
        "--compact-root",
        type=Path,
        default=YEAR_ROOT / "data" / "nas_compact",
    )
    args = p.parse_args(argv)
    config_dir = args.config_dir.resolve()
    compact_root = args.compact_root.resolve()
    skip = {"nas_best_all", "nas_best_vit_maxvit_convnext_all"}
    total = 0
    for json_path in sorted(config_dir.glob("nas_best_*.json")):
        if json_path.stem in skip:
            continue
        n = refresh_json_path(json_path, compact_root=compact_root)
        print(f"Updated {json_path.name} ({n} entries)")
        total += n
    # Combined table
    all_csv = config_dir / "nas_best_all.csv"
    if all_csv.is_file():
        rows: list[dict] = []
        with all_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                enrich_row_from_compact_history(row, compact_root=compact_root)
                rows.append(row)
        write_best_csv(all_csv, rows)
        write_best_json(
            all_csv.with_suffix(".json"),
            rows,
            rank_metric="val_acc",
            label="All NAS model groups (best per condition)",
        )
        print(f"Updated {all_csv.name} ({len(rows)} rows)")
    print(f"Done. Refreshed metrics for {total} best-config entries.")


if __name__ == "__main__":
    main()
