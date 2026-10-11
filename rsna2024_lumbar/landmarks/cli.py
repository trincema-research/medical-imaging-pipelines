"""CLI: python -m rsna2024_lumbar.landmarks {densify,train,infer,nas,pack,unzip,deploy}"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import (
    BACKBONE_CONVNEXT_TINY,
    BACKBONES,
    CONDITIONS,
    DENSE_COORDINATES_DIRNAME,
    IMAGE_SIZE_DEFAULT,
    SPLIT_SEED_DEFAULT,
)
from rsna2024_lumbar.landmarks.densify import DEFAULT_OUTPUT, DEFAULT_RAW, densify_coordinates

HERE = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = HERE / "data" / "processed" / DENSE_COORDINATES_DIRNAME / "runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete lumbar (x, y) coordinates for every series slice.",
        epilog="Cloud: nas | pack | unzip | deploy (see those modules for flags).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    densify = sub.add_parser("densify", help="Project each labeled mark onto every slice in its series.")
    densify.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    densify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    densify.add_argument("--max-studies", type=int, default=None)
    densify.add_argument("--study-id", type=int, action="append", dest="study_ids")
    densify.add_argument("--progress", action="store_true")

    train = sub.add_parser("train", help="Train a 2D ConvNeXt / EfficientNetV2 (x, y) regressor (1st-place style).")
    train.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS)
    train.add_argument("--condition", choices=list(CONDITIONS), required=True)
    train.add_argument("--backbone", choices=list(BACKBONES), default=BACKBONE_CONVNEXT_TINY)
    train.add_argument("--image-size", type=int, default=IMAGE_SIZE_DEFAULT)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=SPLIT_SEED_DEFAULT)
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--max-studies", type=int, default=None)

    infer = sub.add_parser("infer", help="Predict (x, y) and project onto the rest of each series.")
    infer.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output-csv", type=Path, required=True)
    infer.add_argument("--study-id", type=int, action="append", dest="study_ids")
    infer.add_argument("--no-project", action="store_true")
    return p.parse_args(argv)


CLOUD_COMMANDS = {
    "nas": "rsna2024_lumbar.landmarks.launch",
    "pack": "rsna2024_lumbar.landmarks.pack",
    "unzip": "rsna2024_lumbar.landmarks.unzip",
    "deploy": "rsna2024_lumbar.landmarks.deploy",
}


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in CLOUD_COMMANDS:
        return int(importlib.import_module(CLOUD_COMMANDS[args_list[0]]).main(args_list[1:]))
    args = parse_args(argv)
    if args.command == "densify":
        result = densify_coordinates(
            args.data_root,
            args.output_root,
            max_studies=args.max_studies,
            study_ids=args.study_ids,
            progress=args.progress,
        )
        print(
            f"Wrote {result['csv_path']}  rows={result['n_rows']}  "
            f"labeled={result['n_labeled']} projected={result['n_projected']} copied={result['n_copied']}"
        )
        return 0
    if args.command == "train":
        from rsna2024_lumbar.landmarks.train import train_condition

        meta = train_condition(
            args.data_root,
            args.output_dir,
            args.condition,
            backbone=args.backbone,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            pretrained=args.pretrained,
            max_studies=args.max_studies,
        )
        print(f"Wrote {meta['checkpoint']}  train={meta['n_train']} val={meta['n_val']}")
        return 0
    if args.command == "infer":
        from rsna2024_lumbar.landmarks.infer import predict_condition_csv

        meta = predict_condition_csv(
            args.data_root,
            args.checkpoint,
            args.output_csv,
            study_ids=args.study_ids,
            project_series=not args.no_project,
        )
        print(f"Wrote {meta['csv']}  rows={meta['n_rows']}")
        return 0
    raise SystemExit(f"Unknown command {args.command}")
