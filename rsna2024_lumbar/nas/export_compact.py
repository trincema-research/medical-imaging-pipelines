"""Export compact NAS artifacts (training histories + trial config JSON) for git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna2024_lumbar.nas.catalog import MODEL_GROUPS, targets_for_groups
from rsna2024_lumbar.nas.compact import (
    estimate_history_bytes,
    export_trials,
    load_best_trial_dirs,
)
from rsna2024_lumbar.nas.extract import extract_best_for_target
from rsna2024_lumbar.nas.rank import DEFAULT_RANK_METRIC, RANK_METRICS
from rsna2024_lumbar.nas.report import (
    rows_best_per_condition,
    write_best_csv,
)
from rsna2024_lumbar.nas.results import default_archive_root
from rsna2024_lumbar.nas.summarize import filter_targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Copy training_history.csv (+ trial_config.json from result.json) into "
            "rsna2024_lumbar/data/nas_compact/<ViT|MaxViT|Convnext>/<2d|3d>/..."
        )
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Legacy NAS results parent (ViT/, MaxViT/, Convnext/).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("rsna2024_lumbar/data/nas_compact"),
        help="Destination repo tree (default: rsna2024_lumbar/data/nas_compact).",
    )
    p.add_argument(
        "--mode",
        choices=["best", "all"],
        default="best",
        help="best = 5 trials per model/layout (git-friendly); all = full NAS export (~800MB histories).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_GROUPS),
        default=None,
    )
    p.add_argument("--layouts", nargs="+", choices=["2d", "3d"], default=None)
    p.add_argument("--metric", choices=RANK_METRICS, default=DEFAULT_RANK_METRIC)
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print size estimates and exit (no copy).",
    )
    p.add_argument(
        "--yes-all",
        action="store_true",
        help="Required with --mode all to confirm large export.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive = args.archive_root or default_archive_root()
    if archive is None:
        raise SystemExit("Pass --archive-root or set LUMBAR_NAS_RESULTS_ROOT.")
    archive = Path(archive).resolve()
    output_base = Path(args.output_root).resolve()
    trial_output = output_base / "best" if args.mode == "best" else output_base
    targets = filter_targets(args)

    if args.estimate_only or args.mode == "all":
        grand_count = 0
        grand_bytes = 0
        print(f"Estimate training_history.csv under {archive}")
        for target in targets:
            count, size = estimate_history_bytes(archive, target)
            grand_count += count
            grand_bytes += size
            print(f"  {target.label}: {count} trials, {size / (1024 * 1024):.1f} MB")
        print(f"  TOTAL: {grand_count} trials, {grand_bytes / (1024 * 1024):.1f} MB")
        if args.estimate_only:
            return
        if args.mode == "all" and not args.yes_all:
            raise SystemExit("Refusing --mode all without --yes-all (histories are hundreds of MB).")

    best_rows: list[dict] = []
    manifest_targets: list[dict] = []

    for target in targets:
        only_dirs: set[Path] | None
        if args.mode == "best":
            only_dirs = load_best_trial_dirs(target, archive, args.metric)
        else:
            only_dirs = None

        stats = export_trials(
            target=target,
            archive_root=archive,
            output_root=trial_output,
            only_trial_dirs=only_dirs,
        )
        manifest_targets.append(stats)
        mb = int(stats["bytes_written"]) / (1024 * 1024)
        print(
            f"OK {target.label}: exported {stats['exported_trials']} trials "
            f"({mb:.2f} MB) -> {stats['dest_root']}"
        )

        if args.mode == "best":
            result = extract_best_for_target(
                target,
                archive_root=archive,
                metric=args.metric,
                include_shared=False,
            )
            rows = rows_best_per_condition(
                result.best,
                family=target.family,
                results_root=result.results_root,
                rank_metric=args.metric,
            )
            for row in rows:
                row["model_group"] = target.model_group
                row["archive_layout"] = target.layout
                row["archive_label"] = target.label
            best_rows.extend(rows)
            csv_path = output_base / "best_configs" / f"nas_best_{target.family}_{target.layout}.csv"
            write_best_csv(csv_path, rows)

    manifest = {
        "mode": args.mode,
        "metric": args.metric,
        "archive_root": str(archive),
        "output_root": str(output_base),
        "trial_tree": str(trial_output),
        "targets": manifest_targets,
    }
    output_base.mkdir(parents=True, exist_ok=True)
    manifest_path = output_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if best_rows:
        combined = output_base / "best_configs" / "nas_best_vit_maxvit_convnext_all.csv"
        write_best_csv(combined, best_rows)
        print(f"Wrote best-config CSVs under {output_base / 'best_configs'}")

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
