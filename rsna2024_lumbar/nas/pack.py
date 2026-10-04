"""Build a cloud-deploy zip for lumbar NAS (code + labels, optional PNG crops)."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from rsna2024_lumbar.nas.families import FAMILIES, FAMILY_SPECS
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS

YEAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = YEAR_ROOT.parent

PROFILES: dict[str, dict[str, object]] = {
    "all": {
        "families": list(FAMILIES),
        "default_output": "lumbar_nas_all_cloud.zip",
    },
    "efficientnet_families": {
        "families": ["efficientnet", "efficientnet3d"],
        "default_output": "lumbar_efficientnet_nas_2d3d_cloud.zip",
    },
    "convnext_families": {
        "families": ["convnext", "convnext3d"],
        "default_output": "lumbar_convnext_nas_2d3d_cloud.zip",
    },
    "transformers": {
        "families": ["vit", "maxvit"],
        "default_output": "lumbar_nas_transformers_cloud.zip",
    },
}

CODE_PATHS = (
    YEAR_ROOT / "nas",
    YEAR_ROOT / "preprocessing",
    YEAR_ROOT / "__init__.py",
    REPO_ROOT / "common",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
)

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "runs"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zip lumbar NAS code for cloud deploy.")
    p.add_argument("--profile", choices=sorted(PROFILES), default="all")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--include-crops",
        action="store_true",
        help="Also pack data/processed/<crop-policy>/ (large).",
    )
    p.add_argument("--crop-policy", default="centered")
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


def collect_pack_files(*, include_crops: bool, crop_policy: str) -> list[Path]:
    files: list[Path] = []
    for path in CODE_PATHS:
        if path.exists():
            files.extend(_iter_files(path))
    raw = YEAR_ROOT / "data" / "raw"
    for name in LABEL_CSVS:
        csv = raw / name
        if csv.is_file():
            files.append(csv)
    readme = YEAR_ROOT / "nas" / "README.md"
    if readme.is_file():
        files.append(readme)
    if include_crops:
        crops = YEAR_ROOT / "data" / "processed" / crop_policy
        if not crops.is_dir():
            raise SystemExit(f"No crop cache at {crops}. Export first or omit --include-crops.")
        files.extend(_iter_files(crops))
    # De-dupe while keeping order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def cloud_run_text(families: list[str]) -> str:
    lines = [
        "pip install -e .",
        "",
        "Shard plan (no spawn):",
    ]
    for family in families:
        lines.append(
            f"python -m rsna2024_lumbar.nas.launch --family {family} "
            "--num-gpus 0 --visible-gpus 8 --all-conditions"
        )
    lines.extend(
        [
            "",
            "Spawn one process per GPU (1/2/4/8; 0 = auto):",
        ]
    )
    for family in families:
        lines.append(
            f"python -m rsna2024_lumbar.nas.launch --family {family} "
            "--num-gpus 0 --all-conditions --spawn --wait"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    families = list(profile["families"])
    output = args.output or (REPO_ROOT / str(profile["default_output"]))
    output = output.resolve()
    files = collect_pack_files(include_crops=args.include_crops, crop_policy=args.crop_policy)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "families": families,
        "allowed_num_gpus": [1, 2, 4, 8],
        "include_crops": args.include_crops,
        "file_count": len(files),
        "family_output_bases": {name: FAMILY_SPECS[name].output_base for name in families},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            try:
                arc = path.resolve().relative_to(REPO_ROOT.resolve())
            except ValueError:
                arc = Path(path.name)
            zf.write(path, arc.as_posix())
        zf.writestr("deploy_manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("CLOUD_RUN.txt", cloud_run_text(families))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")
    print(f"  profile: {args.profile}")
    print(f"  families: {', '.join(families)}")
    print(f"  files: {len(files)}")
    print(f"  crops: {'yes' if args.include_crops else 'no'}")


if __name__ == "__main__":
    main()
