"""Study-level 70/15/15 split. Never split slices from the same patient."""

from __future__ import annotations

import math

import numpy as np


def split_study_ids(
    study_ids: list[int],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    ids = sorted({int(v) for v in study_ids})
    if not ids:
        return {"train": [], "val": [], "holdout": []}
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(ids).tolist()
    n = len(order)
    n_train = max(1, int(math.floor(n * ratios[0]))) if n >= 3 else max(1, n - 2)
    n_val = int(math.floor(n * ratios[1])) if n >= 3 else (1 if n >= 2 else 0)
    if n >= 3 and n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_hold = n - n_train - n_val
    train = [int(v) for v in order[:n_train]]
    val = [int(v) for v in order[n_train : n_train + n_val]]
    holdout = [int(v) for v in order[n_train + n_val : n_train + n_val + n_hold]]
    return {"train": sorted(train), "val": sorted(val), "holdout": sorted(holdout)}
