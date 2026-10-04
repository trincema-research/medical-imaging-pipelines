"""Unit tests for ``nas.results``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.nas.results import (
    attach_grid_indices,
    default_archive_root,
    filter_records,
    iter_trial_records,
    load_trial_record,
    resolve_results_root,
    summarize_scan,
)
from rsna2024_lumbar.tests.nas_fixtures import (
    write_nas_result,
    write_nas_result_at_grid_index,
)


def test_resolve_results_root_explicit(tmp_path: Path):
    write_nas_result(tmp_path / "t1", trial_id=1, condition="spinal_canal_stenosis")
    root = resolve_results_root("vit", results_root=tmp_path)
    assert root == tmp_path.resolve()


def test_resolve_results_root_archive_layout(tmp_path: Path):
    archive = tmp_path / "NAS results"
    trial = archive / "ViT" / "2d" / "scs" / "trial_1"
    write_nas_result(trial, trial_id=1, condition="spinal_canal_stenosis")
    root = resolve_results_root("vit", archive_root=archive, layout="2d")
    assert root == (archive / "ViT" / "2d").resolve()


def test_resolve_results_root_layout_alias_3d(tmp_path: Path):
    archive = tmp_path / "NAS results"
    (archive / "ViT" / "3D").mkdir(parents=True)
    write_nas_result(
        archive / "ViT" / "3D" / "t1",
        trial_id=1,
        condition="spinal_canal_stenosis",
        model_type="vit3d",
        input_layout="3d",
    )
    root = resolve_results_root("vit3d", archive_root=archive, layout="3d")
    assert root.name in ("3d", "3D")


def test_resolve_results_root_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        resolve_results_root("vit", results_root=tmp_path / "missing")


def test_default_archive_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LUMBAR_NAS_RESULTS_ROOT", str(tmp_path))
    assert default_archive_root() == tmp_path.resolve()
    monkeypatch.delenv("LUMBAR_NAS_RESULTS_ROOT", raising=False)


def test_iter_and_filter_records(tmp_path: Path):
    write_nas_result(
        tmp_path / "a",
        trial_id=1,
        condition="spinal_canal_stenosis",
        model_type="convnext",
    )
    write_nas_result(
        tmp_path / "b",
        trial_id=2,
        condition="left_neural_foraminal_narrowing",
        model_type="vit",
    )
    records = list(iter_trial_records(tmp_path))
    assert len(records) == 2
    vit_only = filter_records(records, family="vit")
    assert len(vit_only) == 1
    scs = filter_records(records, condition="spinal_canal_stenosis")
    assert len(scs) == 1


def test_load_trial_record_without_best_metrics(tmp_path: Path):
    write_nas_result(
        tmp_path,
        trial_id=1,
        condition="spinal_canal_stenosis",
        include_best_metrics=False,
        max_val_acc=0.77,
    )
    record = load_trial_record(tmp_path / "result.json")
    assert record.val_acc is None
    assert record.val_loss is None
    assert record.max_val_acc == pytest.approx(0.77)


def test_attach_grid_indices(tmp_path: Path):
    write_nas_result_at_grid_index(
        tmp_path / "t1",
        family="vit",
        condition="spinal_canal_stenosis",
        trial_id=1,
        grid_index=0,
    )
    records = list(iter_trial_records(tmp_path))
    indexed = attach_grid_indices(records, family="vit")
    assert indexed[0].grid_index == 1


def test_summarize_scan(tmp_path: Path):
    write_nas_result(tmp_path / "t1", trial_id=1, condition="spinal_canal_stenosis")
    records = list(iter_trial_records(tmp_path))
    summary = summarize_scan(tmp_path, records)
    assert summary["trial_count"] == 1
    assert "spinal_canal_stenosis" in summary["conditions"]
