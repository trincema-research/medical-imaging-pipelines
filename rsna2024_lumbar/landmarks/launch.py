"""Shard the landmark NAS grid across 1/2/4/8 GPUs (same formula as severity NAS)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rsna2024_lumbar.landmarks.constants import BATCH_SIZE_DEFAULT, CONDITIONS, DENSE_COORDINATES_DIRNAME
from rsna2024_lumbar.landmarks.search import FAMILIES, FAMILY_FULL, expand_trials
from rsna2024_lumbar.nas.gpus import ALLOWED_NAS_GPU_COUNTS, GpuShard, resolve_num_gpus, shard_trial_range

HERE = Path(__file__).resolve().parents[1]
DEFAULT_RAW = HERE / "data" / "raw"
DEFAULT_OUTPUT = HERE / "data" / "processed" / DENSE_COORDINATES_DIRNAME / "nas"


@dataclass(frozen=True)
class LandmarkLaunchPlan:
    family: str
    condition: str
    n_trials: int
    num_gpus: int
    shards: list[GpuShard]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shard landmark (x, y) NAS across 1, 2, 4, or 8 GPUs.")
    p.add_argument("--family", choices=list(FAMILIES), default=FAMILY_FULL)
    p.add_argument("--condition", choices=list(CONDITIONS), default="spinal_canal_stenosis")
    p.add_argument("--all-conditions", action="store_true")
    p.add_argument("--num-gpus", type=int, default=0, help="1, 2, 4, or 8. 0 = auto.")
    p.add_argument("--visible-gpus", type=int, default=None)
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_RAW,
        help="Kaggle-shaped raw dump (train.csv + train_images/). Default: data/raw/.",
    )
    p.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Fixed batch size for every trial (default {BATCH_SIZE_DEFAULT}). Not a search axis.",
    )
    p.add_argument("--max-studies", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--list", action="store_true", help="Print the grid and exit.")
    p.add_argument("--spawn", action="store_true")
    p.add_argument("--wait", action="store_true")
    return p.parse_args(argv)


def build_plan(
    condition: str,
    num_gpus: int,
    *,
    family: str = FAMILY_FULL,
    visible: int | None = None,
) -> LandmarkLaunchPlan:
    trials = expand_trials(condition, family=family)
    resolved = resolve_num_gpus(num_gpus, visible=visible)
    shards = shard_trial_range(len(trials), resolved)
    return LandmarkLaunchPlan(
        family=family,
        condition=condition,
        n_trials=len(trials),
        num_gpus=resolved,
        shards=shards,
    )


def format_plan(plan: LandmarkLaunchPlan) -> str:
    lines = [
        f"Family:     {plan.family}",
        f"Condition:  {plan.condition}",
        f"Trials:     {plan.n_trials}",
        f"GPUs:       {plan.num_gpus}  (allowed {ALLOWED_NAS_GPU_COUNTS})",
    ]
    for shard in plan.shards:
        lines.append(
            f"  GPU {shard.gpu_id}: trials {shard.start_trial}-{shard.end_trial} ({shard.n_trials})"
        )
    return "\n".join(lines)


def spawn_shard(plan: LandmarkLaunchPlan, shard: GpuShard, args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard.gpu_id)
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.landmarks.worker",
        "--family",
        plan.family,
        "--condition",
        plan.condition,
        "--start-trial",
        str(shard.start_trial),
        "--end-trial",
        str(shard.end_trial),
        "--output-base",
        str(args.output_base),
    ]
    cmd.extend(["--data-root", str(args.data_root)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_studies is not None:
        cmd.extend(["--max-studies", str(args.max_studies)])
    if args.skip_completed:
        cmd.append("--skip-completed")
    print(f"GPU {shard.gpu_id}: trials {shard.start_trial}-{shard.end_trial}", flush=True)
    return subprocess.Popen(cmd, env=env)


def _wait(processes: list[subprocess.Popen]) -> None:
    codes = [proc.wait() for proc in processes]
    failed = sum(code != 0 for code in codes)
    if failed:
        raise SystemExit(f"{failed} GPU shard(s) exited with non-zero status.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = list(CONDITIONS) if args.all_conditions else [args.condition]
    if args.list:
        trials = expand_trials(conditions[0], family=args.family, batch_size=args.batch_size)
        backs = sorted({t.backbone for t in trials})
        sizes = sorted({t.image_size for t in trials})
        lrs = sorted({t.lr for t in trials})
        wds = sorted({t.weight_decay for t in trials})
        print(
            f"family={args.family}  {len(trials)} trials per condition  |  "
            f"{len(trials) * len(list(CONDITIONS))} if all 5"
        )
        print(f"  models     {backs}")
        print(f"  image_size {sizes}  (fixed)")
        print(f"  lr         {lrs}")
        print(f"  wd         {wds}")
        print(f"  batch      {trials[0].batch_size}  epochs={trials[0].epochs}  pretrained=ImageNet")
        print(f"  product    {len(backs)}×{len(sizes)}×{len(lrs)}×{len(wds)} = {len(trials)}")
        for trial in trials:
            print(f"  {trial.slug}")
        return 0
    plans = [
        build_plan(c, args.num_gpus, family=args.family, visible=args.visible_gpus) for c in conditions
    ]
    for plan in plans:
        print(format_plan(plan))
        print()
    if not args.spawn:
        return 0
    if args.all_conditions and args.wait:
        for plan in plans:
            print(f"=== {plan.condition} ({plan.n_trials} trials, {plan.num_gpus} GPU(s)) ===")
            _wait([spawn_shard(plan, shard, args) for shard in plan.shards])
        print("All GPU shards finished.")
        return 0
    processes = [spawn_shard(plan, shard, args) for plan in plans for shard in plan.shards]
    if not args.wait:
        print(f"Started {len(processes)} background shard(s).")
        return 0
    _wait(processes)
    print("All GPU shards finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
