"""Extract a perf_pipeline cloud zip and verify layout."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from rsna2024_lumbar.perf_pipeline.cloud_common import MANIFEST_NAME, read_manifest, repo_root_from_extracted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unpack lumbar perf_pipeline cloud zip.")
    p.add_argument("--zip", type=Path, required=True, help="Archive from pack_cloud.")
    p.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Empty or existing directory; contents are extracted here.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Extract even if dest already contains pyproject.toml.",
    )
    return p.parse_args(argv)


def unpack(zip_path: Path, dest: Path, *, force: bool = False) -> Path:
    zip_path = zip_path.resolve()
    dest = dest.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"Zip not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / "pyproject.toml").is_file():
        raise SystemExit(f"{dest} already looks like a repo root. Use --force or pick an empty --dest.")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    repo_root = repo_root_from_extracted(dest)
    manifest = read_manifest(repo_root)
    if manifest:
        print(f"Manifest: {repo_root / MANIFEST_NAME}")
        print(f"  expected rows: {manifest.get('expected_result_rows', '?')}")
        print(f"  repeats (pack default): {manifest.get('repeats_default', '?')}")
    print(f"Repo root: {repo_root}")
    return repo_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    unpack(args.zip, args.dest, force=args.force)


if __name__ == "__main__":
    main()
