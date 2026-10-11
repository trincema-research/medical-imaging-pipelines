#!/usr/bin/env python3
"""Extract lumbar_landmarks_cloud.zip with stdlib only (no package install).

Shipped inside the zip as unzip_lumbar_landmarks_cloud.py.

  python unzip_lumbar_landmarks_cloud.py --zip lumbar_landmarks_cloud.zip --dest ./landmarks_cloud
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

MANIFEST_NAME = "landmarks_deploy_manifest.json"


def repo_root_from_extracted(extract_dir: Path) -> Path:
    extract_dir = extract_dir.resolve()
    if (extract_dir / "pyproject.toml").is_file():
        return extract_dir
    for candidate in extract_dir.iterdir():
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise SystemExit(f"No pyproject.toml under {extract_dir}.")


def extract_zip(zip_path: Path, dest: Path, *, force: bool = False) -> Path:
    zip_path = zip_path.resolve()
    dest = dest.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"Zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / "pyproject.toml").is_file():
        raise SystemExit(f"{dest} already contains pyproject.toml. Use --force or another --dest.")
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return repo_root_from_extracted(dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unzip the landmark cloud archive (stdlib only).")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    repo = extract_zip(args.zip, args.dest, force=args.force)
    manifest_path = repo / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Manifest: {manifest_path}")
        print(f"  images: {manifest.get('include_images')}")
        print(f"  files: {manifest.get('file_count')}")
    print(f"Repo root: {repo}")
    print()
    print("Next:")
    print(f"  cd {repo}")
    print("  pip install -e '.[landmarks]'")
    print("  python -m rsna2024_lumbar.landmarks deploy --num-gpus 8 --all-conditions --skip-completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
