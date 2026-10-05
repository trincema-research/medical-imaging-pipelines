"""CLI for the post-NAS performance pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.perf_pipeline.config import list_best_config_files
from rsna2024_lumbar.perf_pipeline.nas_snapshot import (
    snapshot_all_best_configs,
    write_nas_snapshot_csv,
)
from rsna2024_lumbar.perf_pipeline.paths import (
    DEFAULT_BEST_CONFIG_DIR,
    DEFAULT_REPEATS,
    RESULTS_DIR,
)
from rsna2024_lumbar.perf_pipeline.results import write_pipeline_results_csv
from rsna2024_lumbar.perf_pipeline.runner import run_best_config_pipeline


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

    run_all = sub.add_parser("run-all", help="Run pipeline for every per-model nas_best_*.json.")
    run_all.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    run_all.add_argument("--epochs", type=int, default=50)
    run_all.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    run_all.add_argument("--data-root", type=Path, default=None)
    run_all.add_argument("--crops-root", type=Path, default=None)
    run_all.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    run_all.add_argument("--max-studies", type=int, default=None)
    run_all.add_argument("--dry-run", action="store_true")

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
    )
    return rc


def cmd_run_all(args: argparse.Namespace) -> int:
    rc = 0
    combined: list[dict] = []
    for cfg_path in list_best_config_files(args.config_dir):
        code, rows = run_best_config_pipeline(
            cfg_path,
            data_root=args.data_root,
            crops_root=args.crops_root,
            epochs=args.epochs,
            repeats=args.repeats,
            output_base=args.output_base,
            max_studies=args.max_studies,
            dry_run=args.dry_run,
        )
        if code != 0:
            rc = code
        combined.extend(rows)
    if combined and not args.dry_run:
        write_pipeline_results_csv(
            combined,
            args.output_base / "pipeline_results_all_models.csv",
        )
        print(f"Wrote combined {args.output_base / 'pipeline_results_all_models.csv'}")
    return rc


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
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
