"""Batch-extract best NAS configs for ViT, MaxViT, and ConvNeXt (2D + 3D) from legacy archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.nas.catalog import ARCHIVE_RESULT_TARGETS, MODEL_GROUPS, targets_for_groups
from rsna2024_lumbar.nas.extract import extract_best_for_target
from rsna2024_lumbar.nas.rank import DEFAULT_RANK_METRIC, METRIC_HELP, RANK_METRICS
from rsna2024_lumbar.nas.report import (
    COMBINED_BEST_CONFIG_STEM,
    LEGACY_COMBINED_BEST_CONFIG_STEM,
    combined_best_config_csv_path,
    default_csv_path,
    default_json_path,
    merge_combined_best_rows,
    rows_best_per_condition,
    rows_shared_config,
    write_best_csv,
    write_best_json,
)
from rsna2024_lumbar.nas.history_metrics import enrich_row_from_compact_history
from rsna2024_lumbar.nas.paths import YEAR_ROOT
from rsna2024_lumbar.nas.results import default_archive_root
from rsna2024_lumbar.preprocessing.constants import CROP_POLICIES, CROP_POLICY_CENTERED

DEFAULT_COMPACT_ROOT = YEAR_ROOT / "data" / "nas_compact"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract best-per-condition NAS trials for ViT, MaxViT, and ConvNeXt "
            "(2D and 3D) from runs/NAS results/."
        )
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Parent of ViT/, MaxViT/, Convnext/ (default: LUMBAR_NAS_RESULTS_ROOT).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_GROUPS),
        default=None,
        metavar="GROUP",
        help=f"Subset of model groups (default: all). Groups: {', '.join(MODEL_GROUPS)}.",
    )
    p.add_argument(
        "--layouts",
        nargs="+",
        choices=["2d", "3d"],
        default=None,
        help="Only these layouts (default: both 2d and 3d).",
    )
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    metric_help = "; ".join(f"{name}: {METRIC_HELP[name]}" for name in RANK_METRICS)
    p.add_argument(
        "--metric",
        choices=RANK_METRICS,
        default=DEFAULT_RANK_METRIC,
        help=f"Ranking criterion (default: {DEFAULT_RANK_METRIC}). {metric_help}",
    )
    p.add_argument("--attach-grid-index", action="store_true")
    p.add_argument("--shared", action="store_true", help="Append shared-hyperparameter rows per model/layout.")
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for per-model CSV files (default: runs/).",
    )
    p.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Also write one CSV with all models (default: runs/nas_best_vit_maxvit_convnext_all.csv).",
    )
    p.add_argument(
        "--no-combined-csv",
        action="store_true",
        help="Skip the combined all-models CSV.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected archive is missing or empty (default: skip missing).",
    )
    p.add_argument(
        "--compact-root",
        type=Path,
        default=DEFAULT_COMPACT_ROOT,
        help="nas_compact tree for training_history.csv (enriches best-config metrics).",
    )
    return p.parse_args(argv)


def filter_targets(args: argparse.Namespace) -> list:
    targets = targets_for_groups(args.models)
    if args.layouts:
        layouts = {layout.lower() for layout in args.layouts}
        targets = [t for t in targets if t.layout in layouts]
    return targets


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive = args.archive_root or default_archive_root()
    if archive is None:
        raise SystemExit(
            "No archive root. Pass --archive-root or set LUMBAR_NAS_RESULTS_ROOT "
            "to the folder that contains ViT/, MaxViT/, Convnext/."
        )
    archive = Path(archive).resolve()
    targets = filter_targets(args)
    if not targets:
        raise SystemExit("No archive targets selected.")

    combined_rows: list[dict] = []
    wrote = 0
    errors: list[str] = []

    for target in targets:
        label = target.label
        try:
            result = extract_best_for_target(
                target,
                archive_root=archive,
                metric=args.metric,
                attach_grid_index=args.attach_grid_index,
                crop_policy=args.crop_policy,
                include_shared=args.shared,
            )
        except (FileNotFoundError, OSError) as exc:
            msg = f"{label}: {exc}"
            if args.strict:
                raise SystemExit(msg) from exc
            print(f"SKIP  {msg}")
            errors.append(msg)
            continue

        rows = rows_best_per_condition(
            result.best,
            family=target.family,
            results_root=result.results_root,
            rank_metric=args.metric,
        )
        compact_root = Path(args.compact_root).resolve()
        for row in rows:
            row["model_group"] = target.model_group
            row["archive_layout"] = target.layout
            row["archive_label"] = target.label
            if compact_root.is_dir():
                enrich_row_from_compact_history(row, compact_root=compact_root)
        if result.shared is not None:
            shared_rows = rows_shared_config(
                result.shared,
                family=target.family,
                results_root=result.results_root,
                rank_metric=args.metric,
            )
            for row in shared_rows:
                row["model_group"] = target.model_group
                row["archive_layout"] = target.layout
                row["archive_label"] = target.label
                if compact_root.is_dir():
                    enrich_row_from_compact_history(row, compact_root=compact_root)
            rows.extend(shared_rows)

        out_path = args.csv_dir / default_csv_path(target.family, target.layout).name
        write_best_csv(out_path, rows)
        json_path = args.csv_dir / default_json_path(target.family, target.layout).name
        write_best_json(
            json_path,
            rows,
            rank_metric=args.metric,
            archive_root=archive,
            label=label,
        )
        combined_rows.extend(rows)
        wrote += 1
        n_trials = len(result.records)
        print(
            f"OK    {label}: {n_trials} trials -> {out_path} "
            f"({len(result.best)} best-per-condition rows)"
        )

    if not args.no_combined_csv and combined_rows:
        combined_path = args.combined_csv or combined_best_config_csv_path(args.csv_dir)
        existing_rows: list[dict] = []
        if combined_path.is_file():
            import csv as _csv

            with combined_path.open(encoding="utf-8", newline="") as fh:
                existing_rows = list(_csv.DictReader(fh))
        merged = merge_combined_best_rows(existing_rows, combined_rows)
        write_best_csv(combined_path, merged)
        write_best_json(
            combined_path.with_suffix(".json"),
            merged,
            rank_metric=args.metric,
            archive_root=archive,
            label="All NAS model groups (best per condition)",
        )
        legacy = args.csv_dir / f"{LEGACY_COMBINED_BEST_CONFIG_STEM}.csv"
        write_best_csv(legacy, merged)
        write_best_json(
            legacy.with_suffix(".json"),
            merged,
            rank_metric=args.metric,
            archive_root=archive,
            label="All NAS model groups (best per condition)",
        )
        print(
            f"Wrote combined CSV: {combined_path.resolve()} ({len(merged)} rows, "
            f"{COMBINED_BEST_CONFIG_STEM})"
        )

    if errors and not wrote:
        raise SystemExit(f"No archives processed. {len(errors)} skipped.")
    print(f"Done. {wrote}/{len(targets)} archive target(s) written under {args.csv_dir.resolve()}.")


if __name__ == "__main__":
    main()
