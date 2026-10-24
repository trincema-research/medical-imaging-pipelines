"""Run a contiguous 1-based landmark NAS trial range on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import BATCH_SIZE_DEFAULT, CONDITIONS, DENSE_COORDINATES_DIRNAME
from rsna2024_lumbar.landmarks.search import FAMILIES, FAMILY_FULL, expand_trials
from rsna2024_lumbar.landmarks.train import train_condition

HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HERE / "data" / "processed" / DENSE_COORDINATES_DIRNAME / "nas"
DEFAULT_RAW = HERE / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one GPU shard of the landmark NAS grid.")
    p.add_argument("--family", choices=list(FAMILIES), default=FAMILY_FULL)
    p.add_argument("--condition", choices=list(CONDITIONS), required=True)
    p.add_argument("--start-trial", type=int, required=True)
    p.add_argument("--end-trial", type=int, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help=f"Fixed for every trial (default {BATCH_SIZE_DEFAULT}).")
    p.add_argument("--max-studies", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true")
    return p.parse_args(argv)


def history_path(output_dir: Path, condition: str, backbone: str) -> Path:
    return output_dir / f"{condition}_{backbone}_history.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trials = expand_trials(
        args.condition, family=args.family, epochs=args.epochs, batch_size=args.batch_size
    )
    start, end = int(args.start_trial), int(args.end_trial)
    if start < 1 or end < start or end > len(trials):
        raise SystemExit(f"Trial range {start}-{end} is outside 1-{len(trials)}.")
    slice_ = trials[start - 1 : end]
    print(f"worker {args.condition} trials {start}-{end} of {len(trials)}", flush=True)
    for trial in slice_:
        out = Path(args.output_base) / args.family / trial.condition / trial.slug
        hist = history_path(out, trial.condition, trial.backbone)
        if args.skip_completed and hist.is_file():
            print(f"skip {trial.slug}", flush=True)
            continue
        print(f"trial {trial.slug}", flush=True)
        train_condition(
            args.data_root,
            out,
            trial.condition,
            backbone=trial.backbone,
            image_size=trial.image_size,
            epochs=trial.epochs,
            batch_size=trial.batch_size,
            lr=trial.lr,
            weight_decay=trial.weight_decay,
            seed=trial.seed,
            pretrained=trial.pretrained,
            max_studies=args.max_studies,
        )
        (out / "trial.json").write_text(
            json.dumps(
                {
                    "trial_id": trial.trial_id,
                    "slug": trial.slug,
                    "condition": trial.condition,
                    "backbone": trial.backbone,
                    "image_size": trial.image_size,
                    "batch_size": trial.batch_size,
                    "family": trial.family,
                    "lr": trial.lr,
                    "weight_decay": trial.weight_decay,
                    "pretrained": trial.pretrained,
                    "epochs": trial.epochs,
                    "seed": trial.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
