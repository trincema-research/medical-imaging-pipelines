"""Shard a NAS trial list across 1, 2, 4, or 8 GPUs (no torch required)."""

from __future__ import annotations

import math
from dataclasses import dataclass

ALLOWED_NAS_GPU_COUNTS = (1, 2, 4, 8)


@dataclass(frozen=True)
class GpuShard:
    gpu_id: int
    start_trial: int
    end_trial: int

    @property
    def n_trials(self) -> int:
        return self.end_trial - self.start_trial + 1


def count_visible_cuda_devices() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def resolve_num_gpus(requested: int, *, visible: int | None = None) -> int:
    """
    ``requested=0`` → largest allowed count that fits visible CUDA devices.
    Explicit 1 / 2 / 4 / 8 must not exceed the visible device count.
    """
    if visible is None:
        visible = count_visible_cuda_devices()
    if requested == 0:
        chosen = max((n for n in ALLOWED_NAS_GPU_COUNTS if n <= visible), default=0)
        if chosen < 1:
            raise SystemExit(
                f"No usable GPUs (visible={visible}; need 1, 2, 4, or 8)."
            )
        return chosen
    if requested not in ALLOWED_NAS_GPU_COUNTS:
        raise SystemExit(
            f"--num-gpus must be 0 (auto) or one of {ALLOWED_NAS_GPU_COUNTS}, got {requested}."
        )
    if requested > visible:
        raise SystemExit(
            f"--num-gpus {requested} requested but only {visible} CUDA device(s) visible."
        )
    return requested


def shard_trial_range(n_trials: int, num_gpus: int) -> list[GpuShard]:
    """Even 1-based shards. Same formula as the lumbar 8-GPU launcher."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if num_gpus not in ALLOWED_NAS_GPU_COUNTS:
        raise ValueError(f"num_gpus must be one of {ALLOWED_NAS_GPU_COUNTS}, got {num_gpus}.")
    per_gpu = int(math.ceil(n_trials / float(num_gpus)))
    shards: list[GpuShard] = []
    for gpu in range(num_gpus):
        start = gpu * per_gpu + 1
        end = min((gpu + 1) * per_gpu, n_trials)
        if start > n_trials:
            break
        shards.append(GpuShard(gpu_id=gpu, start_trial=start, end_trial=end))
    return shards


def covered_trial_ids(shards: list[GpuShard]) -> list[int]:
    ids: list[int] = []
    for shard in shards:
        ids.extend(range(shard.start_trial, shard.end_trial + 1))
    return ids
