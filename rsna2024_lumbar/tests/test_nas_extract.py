"""Unit tests for ``nas.extract``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.nas.catalog import ARCHIVE_RESULT_TARGETS
from rsna2024_lumbar.nas.extract import extract_best_for_target
from rsna2024_lumbar.nas.rank import METRIC_VAL_ACC
from rsna2024_lumbar.tests.nas_fixtures import write_legacy_archive_trial


def test_extract_best_for_target(tmp_path: Path):
    archive = tmp_path / "NAS results"
    target = next(t for t in ARCHIVE_RESULT_TARGETS if t.family == "convnext")
    for trial_id, val_acc, name in (
        (1, 0.5, "trial_a"),
        (2, 0.99, "trial_b"),
    ):
        root = archive / "Convnext" / "2d" / "spinal_canal_stenosis" / name
        from rsna2024_lumbar.tests.nas_fixtures import write_nas_result_at_grid_index

        write_nas_result_at_grid_index(
            root,
            family="convnext",
            condition="spinal_canal_stenosis",
            trial_id=trial_id,
            grid_index=0,
            val_acc_at_best=val_acc,
        )
    result = extract_best_for_target(
        target,
        archive_root=archive,
        metric=METRIC_VAL_ACC,
        attach_grid_index=True,
    )
    assert result.best["spinal_canal_stenosis"].trial_id == 2
    assert result.records[0].grid_index is not None


def test_extract_best_empty_raises(tmp_path: Path):
    archive = tmp_path / "NAS results"
    target = next(t for t in ARCHIVE_RESULT_TARGETS if t.family == "vit")
    (archive / "ViT" / "2d").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No trials"):
        extract_best_for_target(target, archive_root=archive, metric=METRIC_VAL_ACC)
