"""CLI: scan NAS result trees and report best trials per model / condition."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rsna2024_lumbar.nas.families import FAMILIES, require_family
from rsna2024_lumbar.nas.rank import (
    DEFAULT_RANK_METRIC,
    METRIC_HELP,
    RANK_METRICS,
    best_per_condition,
    best_shared_hyperparams,
    higher_is_better,
    metric_value,
    top_trials,
)
from rsna2024_lumbar.nas.results import (
    attach_grid_indices,
    default_archive_root,
    filter_records,
    iter_trial_records,
    resolve_results_root,
    summarize_scan,
)
from rsna2024_lumbar.nas.report import (
    default_csv_path,
    rows_best_per_condition,
    rows_shared_config,
    write_best_csv,
)
from rsna2024_lumbar.preprocessing.constants import CONDITIONS, CROP_POLICIES, CROP_POLICY_CENTERED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find best NAS trial(s) from result.json trees (legacy archive or runs/)."
    )
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Directory to scan recursively for result.json (overrides --archive-root).",
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Legacy ``runs/NAS results`` parent (default: LUMBAR_NAS_RESULTS_ROOT or known paths).",
    )
    p.add_argument(
        "--layout",
        choices=["2d", "3d"],
        default=None,
        help="When using --archive-root, descend into ViT/MaxViT/Convnext/2d|3d.",
    )
    p.add_argument("--condition", choices=list(CONDITIONS), default=None)
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    metric_help = "; ".join(f"{name}: {METRIC_HELP[name]}" for name in RANK_METRICS)
    p.add_argument(
        "--metric",
        choices=RANK_METRICS,
        default=DEFAULT_RANK_METRIC,
        help=f"Ranking criterion (default: {DEFAULT_RANK_METRIC}). {metric_help}",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Print top N trials for one --condition (0 = off).",
    )
    p.add_argument(
        "--shared",
        action="store_true",
        help="Also report best hyperparameters shared across all five conditions.",
    )
    p.add_argument(
        "--attach-grid-index",
        action="store_true",
        help="Match each trial to the NAS grid index from expand_family.",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Write summary JSON to this path.")
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV path (default when saving: runs/nas_best_<family>[_<layout>].csv).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV (overrides default save for best-per-condition runs).",
    )
    return p.parse_args(argv)


def _format_record(record, metric: str) -> str:
    try:
        score_str = f"{metric_value(record, metric):.4f}"
    except ValueError:
        score_str = "n/a"
    grid = f"grid={record.grid_index}" if record.grid_index is not None else "grid=?"
    return (
        f"  trial {record.trial_id:4d}  {grid}  {metric}={score_str}  "
        f"{record.variant}  bs={record.batch_size}  lr={record.learning_rate:g}  "
        f"wd={record.weight_decay:g}  head={record.head_depth}  "
        f"{record.optimizer_type}  epoch={record.best_epoch}  "
        f"path={record.result_path.parent.name}"
    )


def record_to_dict(record) -> dict:
    data = asdict(record)
    data["result_path"] = str(record.result_path)
    return data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = require_family(args.family)
    archive = args.archive_root
    if archive is None and args.results_root is None:
        archive = default_archive_root()

    root = resolve_results_root(
        args.family,
        results_root=args.results_root,
        archive_root=archive,
        layout=args.layout,
    )
    records = list(iter_trial_records(root))
    records = filter_records(records, family=args.family, condition=args.condition)
    if not records:
        raise SystemExit(f"No trials found under {root} for family={args.family!r}.")

    if args.attach_grid_index:
        records = attach_grid_indices(records, family=args.family, crop_policy=args.crop_policy)

    scan = summarize_scan(root, records)
    print(f"Family:      {spec.name} ({spec.model_type})")
    print(f"Results:     {root}")
    print(f"Trials:      {scan['trial_count']}  model_types={scan['model_types']}")
    print(f"Metric:      {args.metric} ({'higher' if higher_is_better(args.metric) else 'lower'} is better)")
    print()

    payload: dict = {
        "family": spec.name,
        "model_type": spec.model_type,
        "results_root": str(root),
        "metric": args.metric,
        "scan": scan,
        "best_per_condition": {},
    }

    csv_rows: list[dict] = []
    best_map: dict[str, object] = {}

    if args.condition and args.top > 0:
        subset = [r for r in records if r.condition == args.condition]
        print(f"Top {args.top} — {args.condition}")
        for record in top_trials(subset, args.metric, limit=args.top):
            print(_format_record(record, args.metric))
        print()
        payload["top"] = [record_to_dict(r) for r in top_trials(subset, args.metric, limit=args.top)]
    else:
        best_map = best_per_condition(records, args.metric)
        print("Best per condition:")
        for condition, record in best_map.items():
            print(f"{condition}:")
            print(_format_record(record, args.metric))
            payload["best_per_condition"][condition] = record_to_dict(record)
        print()
        csv_rows.extend(
            rows_best_per_condition(
                best_map,
                family=spec.name,
                results_root=root,
                rank_metric=args.metric,
            )
        )

    shared = None
    if args.shared:
        shared = best_shared_hyperparams(records, args.metric)
        if shared is None:
            print("Shared config: no hyperparameter set completed all five conditions.")
        else:
            direction = "higher" if higher_is_better(args.metric) else "lower"
            print(
                f"Best shared hyperparameters (mean {args.metric}={shared.mean_score:.4f}, "
                f"{direction} is better):"
            )
            sample = next(iter(shared.per_condition.values()))
            print(
                f"  {sample.variant}  bs={sample.batch_size}  lr={sample.learning_rate:g}  "
                f"wd={sample.weight_decay:g}  head={sample.head_depth}  "
                f"{sample.optimizer_type}  {sample.image_width}x{sample.image_height}  "
                f"layout={sample.input_layout}"
            )
            payload["best_shared"] = {
                "mean_score": shared.mean_score,
                "sample": record_to_dict(sample),
                "per_condition": {k: record_to_dict(v) for k, v in shared.per_condition.items()},
            }
            csv_rows.extend(
                rows_shared_config(
                    shared,
                    family=spec.name,
                    results_root=root,
                    rank_metric=args.metric,
                )
            )

    write_csv = not args.no_csv and csv_rows
    if write_csv and args.csv_out is None:
        args.csv_out = default_csv_path(spec.name, args.layout)
    if write_csv and args.csv_out is not None:
        write_best_csv(Path(args.csv_out), csv_rows)
        print(f"Wrote {Path(args.csv_out).resolve()} ({len(csv_rows)} row(s))")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
