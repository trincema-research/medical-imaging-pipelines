"""Run one GPU shard of NAS trials via the bundled ``vit_nas_lumbar.py``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rsna2024_lumbar.nas.families import require_family
from rsna2024_lumbar.nas.paths import (
    BUNDLE_DIR,
    default_crops_root,
    default_data_root,
    bundle_ready,
)
from rsna2024_lumbar.nas.search_space import expand_family
from rsna2024_lumbar.preprocessing.constants import CONDITIONS, CROP_POLICIES, CROP_POLICY_CENTERED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one NAS trial shard on the visible GPU.")
    p.add_argument("--family", required=True)
    p.add_argument("--condition", choices=list(CONDITIONS), required=True)
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    p.add_argument("--start-trial", type=int, required=True)
    p.add_argument("--end-trial", type=int, required=True)
    p.add_argument("--output-base", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--crops-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def _bundle_config_rel(spec) -> str:
    return f"configs/{spec.config_file.name}"


def build_vit_nas_command(args: argparse.Namespace, spec) -> list[str]:
    if not bundle_ready():
        raise SystemExit(
            "Training bundle missing (vit_nas_lumbar.py). "
            "Pack with: python -m rsna2024_lumbar.nas.pack --legacy-root /path/to/lumbar --include-crops"
        )
    vit_nas = BUNDLE_DIR / "vit_nas_lumbar.py"
    data_root = (args.data_root or default_data_root()).resolve()
    crops_root = (args.crops_root or default_crops_root(args.crop_policy)).resolve()
    output_base = args.output_base or Path(spec.output_base)
    cmd = [
        sys.executable,
        str(vit_nas),
        "--data-root",
        str(data_root),
        "--model-type",
        spec.model_type,
        "--condition",
        args.condition,
        "--crop-policy",
        args.crop_policy,
        "--image-source",
        "png",
        "--crops-root",
        str(crops_root),
        "--config-file",
        _bundle_config_rel(spec),
        "--output-base",
        str(output_base),
        "--epochs",
        str(args.epochs),
        "--start-trial",
        str(args.start_trial),
        "--end-trial",
        str(args.end_trial),
    ]
    if spec.input_layout != "2d":
        cmd.extend(["--input-layout", spec.input_layout])
    if args.no_amp:
        cmd.append("--no-amp")
    else:
        cmd.append("--amp")
    if args.no_resume:
        cmd.append("--no-resume")
    elif args.resume:
        cmd.append("--resume")
    if args.progress:
        cmd.append("--progress")
    if args.early_stop_patience is not None:
        cmd.extend(["--early-stop-patience", str(args.early_stop_patience)])
    return cmd


def run_shard(args: argparse.Namespace) -> int:
    if args.start_trial < 1 or args.end_trial < args.start_trial:
        raise SystemExit("--start-trial/--end-trial must be 1-based and start <= end.")
    spec, _space, trials = expand_family(
        args.family,
        condition=args.condition,
        crop_policy=args.crop_policy,
    )
    if args.end_trial > len(trials):
        raise SystemExit(f"--end-trial {args.end_trial} exceeds {len(trials)} trials.")
    cmd = build_vit_nas_command(args, spec)
    env = os.environ.copy()
    bundle = str(BUNDLE_DIR.resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = bundle if not prev else f"{bundle}{os.pathsep}{prev}"
    env.setdefault("MPLBACKEND", "Agg")
    print(f"Shard {args.start_trial}-{args.end_trial} ({args.condition}): {' '.join(cmd[2:])}")
    completed = subprocess.run(cmd, env=env, cwd=bundle, check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    code = run_shard(args)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
