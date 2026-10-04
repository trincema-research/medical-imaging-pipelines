"""Shard a family grid across 1/2/4/8 GPUs. Default prints the plan (no spawn)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

from rsna2024_lumbar.nas.families import FAMILIES, FamilySpec
from rsna2024_lumbar.nas.gpus import ALLOWED_NAS_GPU_COUNTS, GpuShard, resolve_num_gpus, shard_trial_range
from rsna2024_lumbar.nas.search_space import expand_family
from rsna2024_lumbar.preprocessing.constants import CONDITIONS, CROP_POLICIES, CROP_POLICY_CENTERED


@dataclass(frozen=True)
class LaunchPlan:
    spec: FamilySpec
    condition: str
    crop_policy: str
    n_trials: int
    num_gpus: int
    shards: list[GpuShard]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shard a lumbar NAS family across 1, 2, 4, or 8 GPUs."
    )
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--condition", choices=list(CONDITIONS), default="spinal_canal_stenosis")
    p.add_argument("--all-conditions", action="store_true")
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    p.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        metavar="N",
        help="1, 2, 4, or 8. 0 = auto from visible CUDA devices.",
    )
    p.add_argument(
        "--visible-gpus",
        type=int,
        default=None,
        metavar="N",
        help="Override visible CUDA count (tests / dry-run). Default: probe torch.",
    )
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--crops-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--spawn", action="store_true", help="Start one subprocess per GPU shard.")
    p.add_argument("--wait", action="store_true", help="With --spawn, block until shards exit.")
    return p.parse_args(argv)


def build_plan(
    family: str,
    *,
    condition: str,
    crop_policy: str,
    num_gpus: int,
    visible: int | None = None,
) -> LaunchPlan:
    spec, _space, trials = expand_family(family, condition=condition, crop_policy=crop_policy)
    resolved = resolve_num_gpus(num_gpus, visible=visible)
    shards = shard_trial_range(len(trials), resolved)
    return LaunchPlan(
        spec=spec,
        condition=condition,
        crop_policy=crop_policy,
        n_trials=len(trials),
        num_gpus=resolved,
        shards=shards,
    )


def format_plan(plan: LaunchPlan) -> str:
    lines = [
        f"Family:      {plan.spec.name} ({plan.spec.model_type})",
        f"Output base: {plan.spec.output_base}",
        f"Condition:   {plan.condition}",
        f"Crop policy: {plan.crop_policy}",
        f"Trials:      {plan.n_trials} per condition",
        f"GPUs:        {plan.num_gpus}  (allowed {ALLOWED_NAS_GPU_COUNTS})",
    ]
    for shard in plan.shards:
        lines.append(
            f"  GPU {shard.gpu_id}: trials {shard.start_trial}-{shard.end_trial} "
            f"({shard.n_trials})"
        )
    return "\n".join(lines)


def _worker_training_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = [
        "--epochs",
        str(args.epochs),
    ]
    if args.data_root:
        flags.extend(["--data-root", args.data_root])
    if args.crops_root:
        flags.extend(["--crops-root", args.crops_root])
    if args.no_amp:
        flags.append("--no-amp")
    elif args.amp:
        flags.append("--amp")
    if args.no_resume:
        flags.append("--no-resume")
    elif args.resume:
        flags.append("--resume")
    if args.no_progress:
        flags.append("--no-progress")
    elif args.progress:
        flags.append("--progress")
    if args.early_stop_patience is not None:
        flags.extend(["--early-stop-patience", str(args.early_stop_patience)])
    return flags


def spawn_shard(plan: LaunchPlan, shard: GpuShard, args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard.gpu_id)
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [
        sys.executable,
        "-m",
        "rsna2024_lumbar.nas.worker",
        "--family",
        plan.spec.name,
        "--condition",
        plan.condition,
        "--crop-policy",
        plan.crop_policy,
        "--start-trial",
        str(shard.start_trial),
        "--end-trial",
        str(shard.end_trial),
        "--output-base",
        plan.spec.output_base,
        *_worker_training_flags(args),
    ]
    print(f"GPU {shard.gpu_id}: trials {shard.start_trial}-{shard.end_trial}")
    return subprocess.Popen(cmd, env=env)


def _wait_processes(processes: list[subprocess.Popen]) -> None:
    codes = [proc.wait() for proc in processes]
    failed = sum(code != 0 for code in codes)
    if failed:
        raise SystemExit(f"{failed} GPU shard(s) exited with non-zero status.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conditions = list(CONDITIONS) if args.all_conditions else [args.condition]
    plans = [
        build_plan(
            args.family,
            condition=condition,
            crop_policy=args.crop_policy,
            num_gpus=args.num_gpus,
            visible=args.visible_gpus,
        )
        for condition in conditions
    ]
    for plan in plans:
        print(format_plan(plan))
        print()
    if not args.spawn:
        return

    sequential = args.all_conditions and args.wait
    if sequential:
        for plan in plans:
            print(f"=== {plan.condition} ({plan.n_trials} trials, {plan.num_gpus} GPU(s)) ===")
            processes = [spawn_shard(plan, shard, args) for shard in plan.shards]
            _wait_processes(processes)
        print("All GPU shards finished.")
        return

    processes: list[subprocess.Popen] = []
    for plan in plans:
        for shard in plan.shards:
            processes.append(spawn_shard(plan, shard, args))
    if not args.wait:
        print(f"Started {len(processes)} background shard(s).")
        return
    _wait_processes(processes)
    print("All GPU shards finished.")


if __name__ == "__main__":
    main()
