"""CLI for the RSNA 2018 lung-segmentation pipeline.

```
python -m rsna2018_cxr.segmentation build-manifest
python -m rsna2018_cxr.segmentation train-vgg
python -m rsna2018_cxr.segmentation train
python -m rsna2018_cxr.segmentation segment
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rsna2018_cxr.segmentation.constants import EXPECTED_COUNTS, NATIVE_HW
from rsna2018_cxr.segmentation.data.severity import build_manifest, count_by_class, count_mismatches

HERE = Path(__file__).resolve().parents[1]
DEFAULT_SEVERITY_ROOT = Path(r"F:\severity_train")
DEFAULT_VGG_ROOT = HERE / "data" / "iteration-1" / "vgg11"
DEFAULT_CHECKPOINT = DEFAULT_VGG_ROOT / "unet_vgg11_1024_best.pth"
DEFAULT_LABELED_IMAGES = Path(r"F:\CXR_segmentation\dataset\images")
DEFAULT_LABELED_MASKS = DEFAULT_VGG_ROOT / "masks"
DEFAULT_VGG_SPLITS = DEFAULT_VGG_ROOT / "splits.csv"
DEFAULT_MANIFEST = HERE / "data" / "manifests" / "master.csv"
DEFAULT_REPORTS = HERE / "reports" / "qc"
DEFAULT_SEGMENT_ROOT = HERE / "data" / "segmented"
DEFAULT_TRAIN_CHECKPOINT = HERE / "data" / "models" / "compact_unet.pt"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required to read config files.") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSNA 2018 severity-stratified lung segmentation.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="Index the severity split and verify class counts.")
    build.add_argument("--severity-root", type=Path, default=DEFAULT_SEVERITY_ROOT)
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--metadata-csv", type=Path, default=None)
    build.add_argument("--allow-count-mismatch", action="store_true")
    build.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)

    vgg = sub.add_parser("train-vgg", help="Train the VGG11-BN U-Net at 1024 on the iteration-1 split.")
    vgg.add_argument("--images-dir", type=Path, default=DEFAULT_LABELED_IMAGES)
    vgg.add_argument("--masks-dir", type=Path, default=DEFAULT_LABELED_MASKS)
    vgg.add_argument("--splits", type=Path, default=DEFAULT_VGG_SPLITS)
    vgg.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    vgg.add_argument("--epochs", type=int, default=100)
    vgg.add_argument("--batch-size", type=int, default=4)
    vgg.add_argument("--lr", type=float, default=5e-4)
    vgg.add_argument("--num-workers", type=int, default=0)

    train = sub.add_parser("train", help="Train the compact U-Net. This did not segment the cohort.")
    train.add_argument("--images-dir", type=Path, default=DEFAULT_LABELED_IMAGES)
    train.add_argument("--masks-dir", type=Path, default=DEFAULT_LABELED_MASKS)
    train.add_argument("--output", type=Path, default=DEFAULT_TRAIN_CHECKPOINT)
    train.add_argument("--image-size", type=int, default=NATIVE_HW[0])
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--config", type=Path, default=None)

    segment = sub.add_parser("segment", help="Segment the cohort and save masks under data/.")
    segment.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    segment.add_argument("--output-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    segment.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    segment.add_argument("--threshold", type=float, default=0.5)
    segment.add_argument("--batch-size", type=int, default=8)
    segment.add_argument("--limit", type=int, default=None)
    segment.add_argument("--inspection-per-class", type=int, default=10)
    segment.add_argument("--fusion", choices=("majority", "staple", "probability"), default="majority")
    segment.add_argument("--no-classical", action="store_true")
    segment.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _write_cohort_report(frame: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = count_by_class(frame)
    lines = [
        "# Severity-split cohort",
        "",
        "S1, S2, and S3 are annotation-derived radiographic burden, not clinical severity.",
        f"Grid: {NATIVE_HW[0]}x{NATIVE_HW[1]}.",
        "",
        "| class | studies |",
        "|---|---:|",
    ]
    for name, count in counts.items():
        lines.append(f"| {name} | {count} |")
    lines.append(f"| total | {int(sum(counts.values()))} |")
    lines.append("")
    text = "\n".join(lines)
    (report_dir / "cohort_counts.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_build(args: argparse.Namespace) -> int:
    frame = build_manifest(args.severity_root, metadata_csv=args.metadata_csv)
    mismatches = count_mismatches(frame, EXPECTED_COUNTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    counts = count_by_class(frame)
    print("class counts:", counts)
    print(f"wrote {args.output} ({len(frame)} studies)")
    _write_cohort_report(frame, args.report_dir)
    if mismatches and not args.allow_count_mismatch:
        print("count mismatch:", mismatches)
        return 1
    return 0


def cmd_train_vgg(args: argparse.Namespace) -> int:
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive.")
    from rsna2018_cxr.segmentation.training.vgg import train_vgg11

    result = train_vgg11(
        args.images_dir,
        args.masks_dir,
        args.splits,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    print(
        f"best epoch {result['best_epoch']} val loss {result['best_val_loss']:.4f} "
        f"on {result['n_train']} train / {result['n_val']} val -> {result['checkpoint']}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.config:
        config = _load_yaml(args.config)
        args.images_dir = Path(config.get("images_dir", args.images_dir))
        args.masks_dir = Path(config.get("masks_dir", args.masks_dir))
        args.output = Path(config.get("output", args.output))
        args.image_size = int(config.get("image_size", args.image_size))
        args.epochs = int(config.get("epochs", args.epochs))
        args.batch_size = int(config.get("batch_size", args.batch_size))
        args.lr = float(config.get("learning_rate", args.lr))
    if args.image_size != NATIVE_HW[0]:
        raise SystemExit(
            f"Training input is {NATIVE_HW[0]}x{NATIVE_HW[1]} to match the severity cohort, got {args.image_size}."
        )
    from rsna2018_cxr.segmentation.training.train import train_compact_unet

    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive.")
    result = train_compact_unet(
        args.images_dir,
        args.masks_dir,
        args.output,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    print(
        f"best val dice {result['best_val_dice']:.4f} on {result['n_pairs']} labeled pairs -> {result['checkpoint']}"
    )
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    from rsna2018_cxr.segmentation.pipeline import segment_cohort

    frame = pd.read_csv(args.manifest)
    updated = segment_cohort(
        frame,
        args.output_root,
        unet_checkpoint=args.checkpoint,
        unet_threshold=args.threshold,
        batch_size=args.batch_size,
        fusion=args.fusion,
        use_classical=not args.no_classical,
        inspection_per_class=args.inspection_per_class,
        limit=args.limit,
        device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_manifest = args.output_root / "manifest.csv"
    updated.to_csv(out_manifest, index=False)
    counts = updated["label_status"].value_counts().to_dict()
    print("label_status:", counts)
    print(f"masks under {args.output_root / 'masks'}")
    print(f"inspection overlays under {args.output_root / 'inspection'}")
    print(f"manifest {out_manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-manifest":
        return cmd_build(args)
    if args.command == "train-vgg":
        return cmd_train_vgg(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "segment":
        return cmd_segment(args)
    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
