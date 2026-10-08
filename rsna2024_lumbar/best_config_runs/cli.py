"""CLI for the post-NAS performance pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.best_config_runs.config import (
    filter_config_paths,
    iter_entries,
    list_best_config_files,
    load_best_config,
)
from rsna2024_lumbar.best_config_runs.jobs import list_pipeline_jobs
from rsna2024_lumbar.best_config_runs.parallel import run_parallel_workers
from rsna2024_lumbar.best_config_runs.nas_snapshot import (
    snapshot_all_best_configs,
    write_nas_snapshot_csv,
)
from rsna2024_lumbar.best_config_runs.condition_specific import write_condition_specific_outputs
from rsna2024_lumbar.best_config_runs.paths import (
    CONDITION_SPECIFIC_SPLIT_SEEDS,
    DEFAULT_BEST_CONFIG_DIR,
    DEFAULT_REPEATS,
    RESULTS_DIR,
    run_output_dir,
)
from rsna2024_lumbar.best_config_runs.results import (
    load_combined_pipeline_results,
    write_pipeline_results_csv,
)
from rsna2024_lumbar.best_config_runs.article_summary import write_article_summaries
from rsna2024_lumbar.best_config_runs.validate_results import format_report, validate_results_tree
from rsna2024_lumbar.best_config_runs.runner import (
    finalize_all_model_summaries,
    resolve_repeat_seeds,
    run_best_config_pipeline,
    run_is_complete,
    run_jobs_file,
    run_pipeline_jobs,
)
from rsna2024_lumbar.nas.gpus import count_visible_cuda_devices


def _add_run_training_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_model.pt / last_model.pt (default: metrics only).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        metavar="N",
        help="Parallel GPUs: 1=sequential, 0=auto (2/4/8), or 2/4/8.",
    )
    parser.add_argument(
        "--visible-gpus",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )


def _worker_argv_from_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.data_root is not None:
        argv.extend(["--data-root", str(args.data_root)])
    if args.crops_root is not None:
        argv.extend(["--crops-root", str(args.crops_root)])
    argv.extend(["--output-base", str(args.output_base)])
    argv.extend(["--epochs", str(args.epochs)])
    if args.max_studies is not None:
        argv.extend(["--max-studies", str(args.max_studies)])
    if args.early_stop_patience is not None:
        argv.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if getattr(args, "save_checkpoints", False):
        argv.append("--save-checkpoints")
    return argv


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
    run.add_argument("--config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    run.add_argument("--condition", type=str, default=None)
    run.add_argument("--epochs", type=int, default=50)
    run.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    run.add_argument("--seeds", type=int, nargs="+", default=None)
    run.add_argument(
        "--condition-specific-seeds",
        action="store_true",
        help=f"Use split seeds {' '.join(str(s) for s in CONDITION_SPECIFIC_SPLIT_SEEDS)} (Beyond-Accuracy protocol).",
    )
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
    _add_run_training_flags(run)

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
    run_all.add_argument("--seeds", type=int, nargs="+", default=None)
    run_all.add_argument(
        "--condition-specific-seeds",
        action="store_true",
        help=f"Use split seeds {' '.join(str(s) for s in CONDITION_SPECIFIC_SPLIT_SEEDS)} (Beyond-Accuracy protocol).",
    )
    _add_run_training_flags(run_all)

    worker = sub.add_parser(
        "worker",
        help="Run a JSON job list on one visible GPU (used by --num-gpus sharding).",
    )
    worker.add_argument("--jobs-file", type=Path, required=True)
    worker.add_argument("--data-root", type=Path, default=None)
    worker.add_argument("--crops-root", type=Path, default=None)
    worker.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    worker.add_argument("--epochs", type=int, default=50)
    worker.add_argument("--max-studies", type=int, default=None)
    worker.add_argument("--early-stop-patience", type=int, default=None, metavar="N")
    worker.add_argument("--save-checkpoints", action="store_true")
    worker.add_argument("--dry-run", action="store_true")

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

    article = sub.add_parser(
        "article-summary",
        help="Mean ± std over repeats at best val epoch (article CSV tables).",
    )
    article.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    article.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places in mean ± std strings (default: 4).",
    )

    cond = sub.add_parser(
        "condition-specific",
        help="Beyond-Accuracy condition-macro / best / worst / range / std tables.",
    )
    cond.add_argument("--output-base", type=Path, default=RESULTS_DIR)

    validate = sub.add_parser(
        "validate-results",
        help="Check pipeline CSVs, article tables, and repeat audit JSON under output-base.",
    )
    validate.add_argument("--output-base", type=Path, default=RESULTS_DIR)
    validate.add_argument(
        "--strict-full-models",
        action="store_true",
        help="Require full 5×5 repeat grids for the six complete retrain layouts.",
    )
    validate.add_argument(
        "--allow-missing-severity",
        action="store_true",
        help="Do not require populated OA/O-MAE/QWK/SER columns.",
    )
    validate.add_argument(
        "--no-audit-files",
        action="store_true",
        help="Skip training_history.json / run_config.json checks.",
    )

    return p.parse_args(argv)


def _cli_seeds(args: argparse.Namespace) -> list[int] | None:
    if args.seeds:
        return list(args.seeds)
    if getattr(args, "condition_specific_seeds", False):
        return list(CONDITION_SPECIFIC_SPLIT_SEEDS)
    return None


def _write_condition_specific(output_base: Path) -> None:
    try:
        paths = write_condition_specific_outputs(output_base)
    except ValueError as exc:
        print(f"Skip condition-specific: {exc}")
        return
    print(f"Wrote {paths['runs']}")
    print(f"Wrote {paths['by_condition']}")
    print(f"Wrote {paths['pipeline_summary']}")
    print(f"Wrote {paths['paper_table']}")
    print(f"Wrote {paths['summary_json']}")


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
    if args.num_gpus != 1 and not args.dry_run:
        jobs = list_pipeline_jobs(
            args.config_dir,
            repeats=args.repeats,
            seeds=_cli_seeds(args),
            only=[args.best_config.stem],
            output_base=args.output_base,
            epochs=args.epochs,
            skip_completed=args.skip_completed,
        )
        if args.condition is not None:
            jobs = [j for j in jobs if j.condition == args.condition]
        visible = args.visible_gpus if args.visible_gpus is not None else count_visible_cuda_devices()
        par_rc = run_parallel_workers(
            jobs,
            num_gpus=args.num_gpus,
            visible_gpus=visible,
            worker_argv_base=_worker_argv_from_args(args),
        )
        if par_rc >= 0:
            if par_rc != 0:
                return par_rc
            finalize_all_model_summaries(
                args.config_dir,
                repeats=args.repeats,
                seeds=_cli_seeds(args),
                only=[args.best_config.stem],
                epochs=args.epochs,
                output_base=args.output_base,
            )
            return 0
    rc, _ = run_best_config_pipeline(
        args.best_config,
        data_root=args.data_root,
        crops_root=args.crops_root,
        epochs=args.epochs,
        repeats=args.repeats,
        seeds=_cli_seeds(args),
        condition=args.condition,
        output_base=args.output_base,
        max_studies=args.max_studies,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
        early_stop_patience=args.early_stop_patience,
        save_checkpoints=args.save_checkpoints,
    )
    return rc


def cmd_worker(args: argparse.Namespace) -> int:
    return run_jobs_file(
        args.jobs_file.resolve(),
        data_root=args.data_root,
        crops_root=args.crops_root,
        epochs=args.epochs,
        output_base=args.output_base,
        max_studies=args.max_studies,
        dry_run=args.dry_run,
        early_stop_patience=args.early_stop_patience,
        save_checkpoints=args.save_checkpoints,
    )


def cmd_run_all(args: argparse.Namespace) -> int:
    if args.num_gpus != 1 and not args.dry_run:
        jobs = list_pipeline_jobs(
            args.config_dir,
            repeats=args.repeats,
            seeds=_cli_seeds(args),
            only=args.only,
            output_base=args.output_base,
            epochs=args.epochs,
            skip_completed=args.skip_completed,
        )
        visible = args.visible_gpus if args.visible_gpus is not None else count_visible_cuda_devices()
        par_rc = run_parallel_workers(
            jobs,
            num_gpus=args.num_gpus,
            visible_gpus=visible,
            worker_argv_base=_worker_argv_from_args(args),
        )
        if par_rc >= 0:
            if par_rc != 0:
                return par_rc
            finalize_all_model_summaries(
                args.config_dir,
                repeats=args.repeats,
                seeds=_cli_seeds(args),
                only=args.only,
                epochs=args.epochs,
                output_base=args.output_base,
            )
            if not args.dry_run:
                combined = load_combined_pipeline_results(args.output_base)
                if combined:
                    out = args.output_base / "pipeline_results_all_models.csv"
                    write_pipeline_results_csv(combined, out)
                    print(f"Wrote combined {out} ({len(combined)} rows)")
                _write_condition_specific(args.output_base)
            return 0

    rc = 0
    for cfg_path in filter_config_paths(args.config_dir, args.only):
        code, _ = run_best_config_pipeline(
            cfg_path,
            data_root=args.data_root,
            crops_root=args.crops_root,
            epochs=args.epochs,
            repeats=args.repeats,
            seeds=_cli_seeds(args),
            output_base=args.output_base,
            max_studies=args.max_studies,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
            early_stop_patience=args.early_stop_patience,
            save_checkpoints=args.save_checkpoints,
        )
        if code != 0:
            rc = code
    if not args.dry_run:
        combined = load_combined_pipeline_results(args.output_base)
        if combined:
            out = args.output_base / "pipeline_results_all_models.csv"
            write_pipeline_results_csv(combined, out)
            print(f"Wrote combined {out} ({len(combined)} rows)")
        _write_condition_specific(args.output_base)
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
    if args.command == "worker":
        raise SystemExit(cmd_worker(args))
    if args.command == "status":
        raise SystemExit(cmd_status(args))
    if args.command == "article-summary":
        wide, long = write_article_summaries(
            args.output_base,
            decimals=args.decimals,
        )
        print(f"Wrote {wide}")
        print(f"Wrote {long}")
        _write_condition_specific(args.output_base)
        raise SystemExit(0)
    if args.command == "condition-specific":
        _write_condition_specific(args.output_base)
        raise SystemExit(0)
    if args.command == "validate-results":
        report = validate_results_tree(
            args.output_base.resolve(),
            require_severity=not args.allow_missing_severity,
            strict_full_models=args.strict_full_models,
            check_audit_files=not args.no_audit_files,
        )
        print(format_report(report))
        raise SystemExit(0 if report.ok else 1)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
