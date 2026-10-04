"""Load NAS archives and compute best trials (shared by ``best`` and ``summarize`` CLIs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rsna2024_lumbar.nas.catalog import ArchiveResultTarget
from rsna2024_lumbar.nas.rank import (
    SharedConfigPick,
    best_per_condition,
    best_shared_hyperparams,
)
from rsna2024_lumbar.nas.results import (
    attach_grid_indices,
    filter_records,
    iter_trial_records,
    resolve_results_root,
)
from rsna2024_lumbar.preprocessing.constants import CROP_POLICY_CENTERED


@dataclass(frozen=True)
class ExtractResult:
    target: ArchiveResultTarget
    results_root: Path
    records: list
    best: dict[str, object]
    shared: SharedConfigPick | None


def extract_best_for_target(
    target: ArchiveResultTarget,
    *,
    archive_root: Path,
    metric: str,
    attach_grid_index: bool = False,
    crop_policy: str = CROP_POLICY_CENTERED,
    include_shared: bool = False,
) -> ExtractResult:
    root = resolve_results_root(
        target.family,
        archive_root=archive_root,
        layout=target.layout,
    )
    records = list(iter_trial_records(root))
    records = filter_records(records, family=target.family)
    if not records:
        raise FileNotFoundError(
            f"No trials under {root} for family={target.family!r} ({target.label})."
        )
    if attach_grid_index:
        records = attach_grid_indices(records, family=target.family, crop_policy=crop_policy)
    best = best_per_condition(records, metric)
    shared = best_shared_hyperparams(records, metric) if include_shared else None
    return ExtractResult(
        target=target,
        results_root=root,
        records=records,
        best=best,
        shared=shared,
    )
