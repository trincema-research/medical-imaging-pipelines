"""One GPU shard. Training loop is not in this package yet — records the shard plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.start_trial < 1 or args.end_trial < args.start_trial:
        raise SystemExit("--start-trial/--end-trial must be 1-based and start <= end.")
    spec, _space, trials = expand_family(
        args.family,
        condition=args.condition,
        crop_policy=args.crop_policy,
    )
    if args.end_trial > len(trials):
        raise SystemExit(f"--end-trial {args.end_trial} exceeds {len(trials)} trials.")
    output_base = args.output_base or Path(spec.output_base)
    dest = output_base / args.condition
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": spec.name,
        "model_type": spec.model_type,
        "condition": args.condition,
        "crop_policy": args.crop_policy,
        "start_trial": args.start_trial,
        "end_trial": args.end_trial,
        "n_trials": args.end_trial - args.start_trial + 1,
        "grid_size": len(trials),
        "status": "shard_recorded",
        "note": "GPU shard is assigned; the training loop is not packaged here yet.",
    }
    path = dest / f"shard_{args.start_trial:04d}_{args.end_trial:04d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
