#!/usr/bin/env python3
"""Extract lumbar_best_config_runs_cloud.zip with stdlib only (no rsna2024_lumbar install).

Copy this file next to the zip on a cloud VM, or use the copy inside a newer zip archive.

Usage:
  python unzip_lumbar_best_config_runs_cloud.py --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

MANIFEST_NAME = "best_config_runs_deploy_manifest.json"


def repo_root_from_extracted(extract_dir: Path) -> Path:
    extract_dir = extract_dir.resolve()
    if (extract_dir / "pyproject.toml").is_file():
        return extract_dir
    for candidate in extract_dir.iterdir():
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise SystemExit(
        f"No pyproject.toml under {extract_dir}. Pick an empty --dest or check the zip path."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Unzip best_config_runs cloud archive (stdlib only).")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    zip_path = args.zip.resolve()
    dest = args.dest.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"Zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    if not args.force and (dest / "pyproject.toml").is_file():
        raise SystemExit(f"{dest} already contains pyproject.toml. Use --force or another --dest.")

    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    repo_root = repo_root_from_extracted(dest)
    manifest_path = repo_root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Manifest: {manifest_path}")
        print(f"  expected rows: {manifest.get('expected_result_rows', '?')}")
        print(f"  repeats default: {manifest.get('repeats_default', '?')}")
    print(f"Repo root: {repo_root}")
    print()
    print("Next:")
    print(f"  cd {repo_root}")
    print("  python -m rsna2024_lumbar.best_config_runs.deploy_cloud --repo-root . --repeats 5 --skip-completed")


if __name__ == "__main__":
    main()
