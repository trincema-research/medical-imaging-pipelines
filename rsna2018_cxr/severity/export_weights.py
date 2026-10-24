"""Strip optimizer state from the study checkpoints so they can be released."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsna2018_cxr.severity.constants import MODELS_DIR, SPECIALIST_CHECKPOINTS


KEEP_KEYS = (
    "epoch",
    "model_state",
    "best_val_loss",
    "config",
    "class_to_idx",
    "train_acc",
    "val_acc",
    "positive_class",
    "split_csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export inference-only specialist checkpoints.")
    p.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"F:\severity_train\output_retrain"),
        help="Folder with the original 1 GB study checkpoints.",
    )
    return p.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def export_one(source: Path, dest: Path) -> dict:
    import torch

    obj = torch.load(source, map_location="cpu", weights_only=False)
    slim = {key: obj[key] for key in KEEP_KEYS if key in obj}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dest)
    return {
        "name": dest.name,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "positive_class": slim.get("positive_class"),
        "val_acc": slim.get("val_acc"),
        "train_acc": slim.get("train_acc"),
        "epoch": slim.get("epoch"),
        "source": str(source),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for spec in SPECIALIST_CHECKPOINTS:
        source = args.source_dir / spec["source_name"]
        if not source.is_file():
            raise SystemExit(f"Missing study checkpoint: {source}")
        dest = args.output_dir / spec["name"]
        print(f"export {source.name} -> {dest.name}")
        rows.append(export_one(source, dest))
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
