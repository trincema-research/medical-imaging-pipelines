"""CLI: python -m rsna2024_lumbar.nas --family vit --list"""

from __future__ import annotations

import argparse

from rsna2024_lumbar.nas.families import FAMILIES
from rsna2024_lumbar.nas.search_space import TrialConfig, expand_family
from rsna2024_lumbar.preprocessing.constants import CONDITIONS, CROP_POLICIES, CROP_POLICY_CENTERED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expand a lumbar NAS family grid (dry-run; no GPU / no training)."
    )
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--condition", choices=list(CONDITIONS), default="spinal_canal_stenosis")
    p.add_argument("--crop-policy", choices=CROP_POLICIES, default=CROP_POLICY_CENTERED)
    p.add_argument("--list", action="store_true", help="Print the first --limit trials.")
    p.add_argument(
        "--limit",
        type=int,
        default=8,
        metavar="N",
        help="Rows to print with --list (0 = all). Default: 8.",
    )
    return p.parse_args(argv)


def format_trial(index: int, trial: TrialConfig) -> str:
    hidden = "none" if trial.head_hidden_dim is None else str(trial.head_hidden_dim)
    return (
        f"{index:4d}  {trial.variant}  {trial.optimizer_type}  "
        f"lr={trial.learning_rate:g}  wd={trial.weight_decay:g}  "
        f"head={trial.head_depth}/{hidden}  bs={trial.batch_size}  "
        f"{trial.image_width}x{trial.image_height}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0.")
    spec, _space, trials = expand_family(
        args.family,
        condition=args.condition,
        crop_policy=args.crop_policy,
    )
    print(f"Family:      {spec.name} ({spec.model_type})")
    print(f"Config:      {spec.config_file.name}")
    print(f"Output base: {spec.output_base}")
    print(f"Condition:   {args.condition}")
    print(f"Crop policy: {args.crop_policy} -> {trials[0].image_width}x{trials[0].image_height}")
    print(f"Trials:      {len(trials)}")
    if not args.list:
        return
    shown = trials if args.limit == 0 else trials[: args.limit]
    for i, trial in enumerate(shown, start=1):
        print(format_trial(i, trial))
    if args.limit and args.limit < len(trials):
        print(f"... {len(trials) - args.limit} more")
