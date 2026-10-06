"""CLI for the post-NAS performance pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.best_config_runs.config import (
    iter_entries,
    list_best_config_files,
    load_best_config,
)
from rsna2024_lumbar.best_config_runs.nas_snapshot import (
    snapshot_all_best_configs,
    write_nas_snapshot_csv,
)
from rsna2024_lumbar.best_config_runs.paths import (
    DEFAULT_BEST_CONFIG_DIR,
    DEFAULT_REPEATS,
    RESULTS_DIR,
    run_output_dir,
)
from rsna2024_lumbar.best_config_runs.results import (
    load_combined_pipeline_results,
    write_pipeline_results_csv,
)
from rsna2024_lumbar.best_config_runs.runner import (
    resolve_repeat_seeds,
    run_best_config_pipeline,
    run_is_complete,
)


def _filter_config_paths(config_dir: Path, only: list[str] | None) -> list[Path]:
    paths = list_best_config_files(config_dir)
    if not only:
        return paths
    want = {s.removesuffix(".json") for s in only}
    return [p for p in paths if p.stem in want]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train NAS best hyperparameters with severity metrics "
            "(OA, O-MAE, QWK, SER) on train/val/test."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="Run pipeline for one nas_best_*.json (5 repeats per condition by default).",
    )
    run.add_argument("--best-config", type=Path, required=True)
    run.add_argument("--condition", type=str, default=None)
    run.add_argument("--epochs", type=int, default=50)
    run.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    run.add_argument("--seeds", type=int, nargs="+", default=None)
    run.add_argument("--data-root", type=Path, default=None)
    run.add_argument("--crops-root", type=Path, default=None)
    run.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    run.add_argument("--max-studies", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip repeats whose run dir already has training_metrics.csv through --epochs.",
    )
    run.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N epochs without val improvement (NAS default on cloud: 5).",
    )

    run_all = sub.add_parser("run-all", help="Run pipeline for every per-model nas_best_*.json.")
    run_all.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    run_all.add_argument("--epochs", type=int, default=50)
    run_all.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    run_all.add_argument("--data-root", type=Path, default=None)
    run_all.add_argument("--crops-root", type=Path, default=None)
    run_all.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    run_all.add_argument("--max-studies", type=int, default=None)
    run_all.add_argument("--dry-run", action="store_true")
    run_all.add_argument("--skip-completed", action="store_true")
    run_all.add_argument("--early-stop-patience", type=int, default=None, metavar="N")
    run_all.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STEM",
        help="Limit to config stems (e.g. nas_best_vit_2d nas_best_efficientnet_2d).",
    )

    status = sub.add_parser(
        "status",
        help="Count completed best_config_runs repeats (for resume / second GPU).",
    )
    status.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    status.add_argument("--epochs", type=int, default=50)
    status.add_argument("--repeats", type=int, default=3)
    status.add_argument("--output-base", type=Path, default=RESULTS_DIR)

    snap = sub.add_parser(
        "nas-snapshot",
        help="OA-only rows from NAS compact histories (no GPU training).",
    )
    snap.add_argument("--best-config", type=Path, required=True)
    snap.add_argument(
        "--compact-root",
        type=Path,
        default=Path("rsna2024_lumbar/data/nas_compact"),
    )
    snap.add_argument("--output-csv", type=Path, default=None)

    snap_all = sub.add_parser(
        "nas-snapshot-all",
        help="OA snapshots for every per-model nas_best_*.json + combined CSV.",
    )
    snap_all.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    snap_all.add_argument(
        "--compact-root",
        type=Path,
        default=Path("rsna2024_lumbar/data/nas_compact"),
    )
    snap_all.add_argument("--output-base", type=Path, default=RESULTS_DIR)

    list_cfg = sub.add_parser("list-configs", help="List per-model nas_best_*.json files.")
    list_cfg.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)

    return p.parse_args(argv)


def cmd_list_configs(args: argparse.Namespace) -> int:
    for path in list_best_config_files(args.config_dir):
        print(path)
    return 0


def cmd_nas_snapshot_all(args: argparse.Namespace) -> int:
    rows = snapshot_all_best_configs(
        config_dir=args.config_dir,
        compact_root=args.compact_root,
        output_base=args.output_base,
    )
    combined = args.output_base / "nas_snapshots_all_models.csv"
    print(f"Wrote {len(rows)} rows -> {combined}")
    return 0


def cmd_nas_snapshot(args: argparse.Namespace) -> int:
    out = args.output_csv or (
        RESULTS_DIR
        / args.best_config.stem
        / f"nas_snapshot_{args.best_config.stem}.csv"
    )
    write_nas_snapshot_csv(
        args.best_config,
        compact_root=args.compact_root,
        output_csv=out,
    )
    print(f"Wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rc, _ = run_best_config_pipeline(
        args.best_config,
        data_root=args.data_root,
        crops_root=args.crops_root,
        epochs=args.epochs,
        repeats=args.repeats,
        seeds=args.seeds,
        condition=args.condition,
        output_base=args.output_base,
        max_studies=args.max_studies,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
        early_stop_patience=args.early_stop_patience,
    )
    return rc


def cmd_run_all(args: argparse.Namespace) -> int:
    rc = 0
    for cfg_path in _filter_config_paths(args.config_dir, args.only):
        code, _ = run_best_config_pipeline(
            cfg_path,
            data_root=args.data_root,
            crops_root=args.crops_root,
            epochs=args.epochs,
            repeats=args.repeats,
            output_base=args.output_base,
            max_studies=args.max_studies,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
            early_stop_patience=args.early_stop_patience,
        )
        if code != 0:
            rc = code
    if not args.dry_run:
        combined = load_combined_pipeline_results(args.output_base)
        if combined:
            out = args.output_base / "pipeline_results_all_models.csv"
            write_pipeline_results_csv(combined, out)
            print(f"Wrote combined {out} ({len(combined)} rows)")
    return rc


def cmd_status(args: argparse.Namespace) -> int:
    total_expected = 0
    total_done = 0
    for cfg_path in list_best_config_files(args.config_dir):
        config = load_best_config(cfg_path)
        repeat_seeds = resolve_repeat_seeds(args.repeats, None)
        n_entries = sum(1 for _ in iter_entries(config))
        expected = n_entries * len(repeat_seeds)
        done = 0
        for entry in iter_entries(config):
            for repeat_index, _ in enumerate(repeat_seeds, start=1):
                run_dir = run_output_dir(
                    cfg_path.stem,
                    str(entry["condition"]),
                    repeat_index,
                    output_base=args.output_base,
                )
                if run_is_complete(run_dir, args.epochs):
                    done += 1
        total_expected += expected
        total_done += done
        print(f"{cfg_path.stem}: {done}/{expected} repeats complete")
    print(f"TOTAL: {total_done}/{total_expected} repeats (target {args.repeats} per condition)")
    return 0 if total_done >= total_expected else 1


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "list-configs":
        raise SystemExit(cmd_list_configs(args))
    if args.command == "nas-snapshot":
        raise SystemExit(cmd_nas_snapshot(args))
    if args.command == "nas-snapshot-all":
        raise SystemExit(cmd_nas_snapshot_all(args))
    if args.command == "run":
        raise SystemExit(cmd_run(args))
    if args.command == "run-all":
        raise SystemExit(cmd_run_all(args))
    if args.command == "status":
        raise SystemExit(cmd_status(args))
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
