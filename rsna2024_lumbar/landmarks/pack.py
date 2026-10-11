"""Zip landmark code, label CSVs, training scripts, and optionally the full DICOM dump."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import CONDITIONS, DENSE_COORDINATES_DIRNAME
from rsna2024_lumbar.landmarks.search import expand_trials
from rsna2024_lumbar.nas.paths import REPO_ROOT, YEAR_ROOT
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS

DEFAULT_ZIP = REPO_ROOT / "lumbar_landmarks_cloud.zip"
SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "runs",
    "training_bundle",
    DENSE_COORDINATES_DIRNAME,
}
STORE_SUFFIXES = {".dcm", ".npy", ".zip"}
UNZIP_NAME = "unzip_lumbar_landmarks_cloud.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zip the landmark pipeline (and optionally train_images/) for cloud deploy."
    )
    p.add_argument("--output", type=Path, default=DEFAULT_ZIP)
    p.add_argument(
        "--include-images",
        action="store_true",
        help="Pack data/raw/train_images/ (large; needed to train on the cloud).",
    )
    p.add_argument("--data-root", type=Path, default=YEAR_ROOT / "data" / "raw")
    return p.parse_args(argv)


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if SKIP_DIR_NAMES & set(path.relative_to(root).parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return files


def collect_pack_files(*, data_root: Path, include_images: bool) -> list[Path]:
    files: list[Path] = []
    for path in (
        YEAR_ROOT / "landmarks",
        YEAR_ROOT / "preprocessing",
        YEAR_ROOT / "nas",
        YEAR_ROOT / "__init__.py",
        YEAR_ROOT / "docs" / "landmark_coordinates.md",
        REPO_ROOT / "common",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements.txt",
    ):
        if path.exists():
            files.extend(_iter_files(path))
    data_root = Path(data_root)
    for name in LABEL_CSVS:
        csv = data_root / name
        if csv.is_file():
            files.append(csv)
    if include_images:
        images = data_root / "train_images"
        if not images.is_dir():
            raise SystemExit(f"No train_images/ at {images}. Omit --include-images or copy the dump.")
        files.extend(_iter_files(images))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def cloud_run_text(*, include_images: bool) -> str:
    images = "yes (train_images/ in the zip)" if include_images else "no — copy train_images/ onto the box"
    return "\n".join(
        [
            "# Unzip, then from the repo root (directory with pyproject.toml):",
            "",
            f"python {UNZIP_NAME} --zip lumbar_landmarks_cloud.zip --dest ./landmarks_cloud",
            "cd landmarks_cloud",
            "pip install -e '.[landmarks]'",
            "",
            "# 8-GPU NAS (32 trials × 5 conditions, sequential-by-condition):",
            "python -m rsna2024_lumbar.landmarks deploy --num-gpus 8 --all-conditions --skip-completed",
            "",
            "# Single condition / print plan only:",
            "python -m rsna2024_lumbar.landmarks nas --num-gpus 8 --condition spinal_canal_stenosis",
            "python -m rsna2024_lumbar.landmarks nas --list",
            "",
            f"DICOMs in zip: {images}",
            "Labels: rsna2024_lumbar/data/raw/*.csv",
            "Training: python -m rsna2024_lumbar.landmarks train  |  nas/worker",
            "",
        ]
    )


def _arcname(path: Path, repo_root: Path, data_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(data_root.resolve())
        return (Path("rsna2024_lumbar") / "data" / "raw" / rel).as_posix()
    except ValueError:
        return path.name


def write_zip(output: Path, files: list[Path], *, data_root: Path, include_images: bool) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = REPO_ROOT.resolve()
    unzip_src = (YEAR_ROOT / "landmarks" / "unzip.py").read_text(encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "include_images": include_images,
        "allowed_num_gpus": [1, 2, 4, 8],
        "trials_per_condition": len(expand_trials(next(iter(CONDITIONS)))),
        "conditions": len(CONDITIONS),
        "file_count": len(files),
        "note": "Landmark 2D (x, y) NAS. Do not commit this zip.",
    }
    with zipfile.ZipFile(output, "w") as zf:
        for path in files:
            arc = _arcname(path, repo, Path(data_root))
            compress = zipfile.ZIP_STORED if path.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED
            zf.write(path, arc, compress_type=compress)
        zf.writestr("landmarks_deploy_manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("CLOUD_RUN.txt", cloud_run_text(include_images=include_images))
        zf.writestr(UNZIP_NAME, unzip_src)
    manifest["zip_path"] = output.as_posix()
    manifest["zip_mb"] = output.stat().st_size / (1024 * 1024)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = collect_pack_files(data_root=args.data_root, include_images=args.include_images)
    meta = write_zip(args.output, files, data_root=args.data_root, include_images=args.include_images)
    print(f"Wrote {meta['zip_path']} ({meta['zip_mb']:.1f} MB)")
    print(f"  files: {meta['file_count']}")
    print(f"  images: {'yes' if args.include_images else 'no'}")
    print(f"  unzip: python {UNZIP_NAME} --zip {Path(args.output).name} --dest ./landmarks_cloud")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
