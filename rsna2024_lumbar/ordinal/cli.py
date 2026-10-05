"""CLI: refit NAS best configs and export ordinal metric CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from rsna2024_lumbar.ordinal.config import iter_entries, load_best_config
from rsna2024_lumbar.ordinal.export import (
    best_epoch_ordinal_row,
    epoch_history_to_ordinal_csv,
)
from rsna2024_lumbar.ordinal.paths import DEFAULT_BEST_CONFIG_DIR, ORDINAL_RESULTS_DIR
from rsna2024_lumbar.ordinal.refit import (
    extract_best_epoch_summary,
    refit_slug,
    run_entry_refit,
    write_refit_summary_csv,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refit NAS best hyperparameters and log ordinal severity metrics."
    )
    sub = p.add_subparsers(dest="command", required=True)

    refit = sub.add_parser("refit", help="Train from a nas_best_*.json entry set.")
    refit.add_argument(
        "--best-config",
        type=Path,
        required=True,
        help="Path to nas_best_*.json (or CSV) under data/nas_compact/best_configs.",
    )
    refit.add_argument("--condition", type=str, default=None)
    refit.add_argument("--epochs", type=int, default=50)
    refit.add_argument("--data-root", type=Path, default=None)
    refit.add_argument("--crops-root", type=Path, default=None)
    refit.add_argument("--output-base", type=Path, default=ORDINAL_RESULTS_DIR)
    refit.add_argument("--max-studies", type=int, default=None)
    refit.add_argument("--dry-run", action="store_true")

    export = sub.add_parser("export", help="Extract ordinal CSVs from an existing run dir.")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument(
        "--write-summary",
        action="store_true",
        help="Also write ordinal_best_epoch.csv in run-dir.",
    )

    list_cfg = sub.add_parser("list-configs", help="List default best-config JSON files.")
    list_cfg.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_BEST_CONFIG_DIR,
    )

    return p.parse_args(argv)


def cmd_list_configs(args: argparse.Namespace) -> int:
    d = args.config_dir
    if not d.is_dir():
        print(f"Missing config dir: {d}")
        return 1
    for path in sorted(d.glob("nas_best_*.json")):
        print(path)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    metrics = run_dir / "training_metrics.csv"
    if not metrics.is_file():
        print(f"No training_metrics.csv in {run_dir}")
        return 1
    out_epochs = run_dir / "ordinal_epoch_metrics.csv"
    epoch_history_to_ordinal_csv(metrics, out_epochs)
    print(f"Wrote {out_epochs}")
    if args.write_summary:
        summary_path = run_dir / "ordinal_best_epoch.csv"
        row = best_epoch_ordinal_row(metrics)
        import pandas as pd

        pd.DataFrame([row]).to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
    return 0


def cmd_refit(args: argparse.Namespace) -> int:
    config = load_best_config(args.best_config)
    summary_rows: list[dict] = []
    rc = 0
    for entry in iter_entries(config, condition=args.condition):
        code = run_entry_refit(
            entry,
            source_config=args.best_config,
            data_root=args.data_root,
            crops_root=args.crops_root,
            epochs=args.epochs,
            output_base=args.output_base,
            max_studies=args.max_studies,
            dry_run=args.dry_run,
        )
        if code != 0:
            rc = code
            continue
        if args.dry_run:
            continue
        slug = refit_slug(entry)
        run_dir = args.output_base / Path(slug)
        metrics_path = run_dir / "training_metrics.csv"
        if metrics_path.is_file():
            try:
                epoch_history_to_ordinal_csv(
                    metrics_path, run_dir / "ordinal_epoch_metrics.csv"
                )
            except ValueError as exc:
                print(f"Warning: {exc}")
            summary = extract_best_epoch_summary(run_dir) or {}
            summary.update(
                {
                    "condition": entry.get("condition"),
                    "family": entry.get("family"),
                    "run_dir": str(run_dir),
                    "source_trial_id": entry.get("trial_id"),
                }
            )
            summary_rows.append(summary)

    if summary_rows and not args.dry_run:
        label = args.best_config.stem
        summary_path = args.output_base / f"ordinal_refit_summary_{label}.csv"
        write_refit_summary_csv(summary_rows, summary_path)
        print(f"Wrote aggregate summary: {summary_path}")
    return rc


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "list-configs":
        raise SystemExit(cmd_list_configs(args))
    if args.command == "export":
        raise SystemExit(cmd_export(args))
    if args.command == "refit":
        raise SystemExit(cmd_refit(args))
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
