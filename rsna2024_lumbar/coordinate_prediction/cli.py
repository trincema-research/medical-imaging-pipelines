"""CLI: python -m rsna2024_lumbar.coordinate_prediction {densify,train,infer,nas,pack,unzip,deploy}"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from rsna2024_lumbar.coordinate_prediction.constants import (
    BACKBONE_CONVNEXT_TINY,
    BACKBONES,
    CONDITIONS,
    DENSE_COORDINATES_DIRNAME,
    IMAGE_SIZE_DEFAULT,
    SPLIT_SEED_DEFAULT,
)
from rsna2024_lumbar.coordinate_prediction.densify import DEFAULT_OUTPUT, DEFAULT_RAW, densify_coordinates
from rsna2024_lumbar.coordinate_prediction.resume import wants_resume

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
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=SPLIT_SEED_DEFAULT)
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--max-studies", type=int, default=None)
    train.add_argument(
        "--all-slices",
        action="store_true",
        help="Train on every densified DICOM (not only the median labeled instance).",
    )
    train.add_argument(
        "--dense-csv",
        type=Path,
        default=None,
        help="Dense coordinate table from `coordinate_prediction densify` (required with --all-slices).",
    )
    train.add_argument("--resume", action="store_true", default=True, help="Continue from last.pt after a crash (default).")
    train.add_argument("--no-resume", action="store_true", help="Ignore last.pt and start this run from epoch 1.")

    infer = sub.add_parser("infer", help="Predict (x, y) and project onto the rest of each series.")
    infer.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output-csv", type=Path, required=True)
    infer.add_argument("--study-id", type=int, action="append", dest="study_ids")
    infer.add_argument("--no-project", action="store_true")
    infer.add_argument(
        "--per-slice",
        action="store_true",
        help="Run the net on every DICOM in the series instead of median-slice + project.",
    )
    return p.parse_args(argv)


CLOUD_COMMANDS = {
    "nas": "rsna2024_lumbar.coordinate_prediction.launch",
    "pack": "rsna2024_lumbar.coordinate_prediction.pack",
    "unzip": "rsna2024_lumbar.coordinate_prediction.unzip",
    "deploy": "rsna2024_lumbar.coordinate_prediction.deploy",
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
        from rsna2024_lumbar.coordinate_prediction.train import train_condition

        meta = train_condition(
            args.data_root,
            args.output_dir,
            args.condition,
            backbone=args.backbone,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            pretrained=args.pretrained,
            max_studies=args.max_studies,
            all_slices=args.all_slices,
            dense_csv=args.dense_csv,
            resume=wants_resume(args),
        )
        print(f"Wrote {meta['checkpoint']}  train={meta['n_train']} val={meta['n_val']}")
        return 0
    if args.command == "infer":
        from rsna2024_lumbar.coordinate_prediction.infer import predict_condition_csv

        meta = predict_condition_csv(
            args.data_root,
            args.checkpoint,
            args.output_csv,
            study_ids=args.study_ids,
            project_series=not args.no_project,
            per_slice=args.per_slice,
        )
        print(f"Wrote {meta['csv']}  rows={meta['n_rows']}")
        return 0
    raise SystemExit(f"Unknown command {args.command}")
