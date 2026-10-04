"""Unit tests for ``nas.rank``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.nas.rank import (
    METRIC_MAX_VAL_ACC,
    METRIC_VAL_ACC,
    METRIC_VAL_F1,
    METRIC_VAL_LOSS,
    best_shared_hyperparams,
    higher_is_better,
    metric_value,
    sort_trials,
    top_trials,
)
from rsna2024_lumbar.nas.results import load_trial_record
from rsna2024_lumbar.tests.nas_fixtures import write_nas_result


def _record(tmp_path: Path, name: str, **kwargs: object):
    write_nas_result(tmp_path / name, **kwargs)  # type: ignore[arg-type]
    return load_trial_record(tmp_path / name / "result.json")


def test_metric_value_variants(tmp_path: Path):
    r = _record(
        tmp_path,
        "a",
        trial_id=1,
        condition="spinal_canal_stenosis",
        val_f1=0.6,
        val_acc_at_best=0.91,
        max_val_acc=0.88,
    )
    assert metric_value(r, METRIC_VAL_ACC) == pytest.approx(0.91)
    assert metric_value(r, METRIC_MAX_VAL_ACC) == pytest.approx(0.88)
    assert metric_value(r, METRIC_VAL_F1) == pytest.approx(0.6)


def test_metric_val_acc_falls_back_to_max(tmp_path: Path):
    write_nas_result(
        tmp_path,
        trial_id=1,
        condition="spinal_canal_stenosis",
        include_best_metrics=False,
        max_val_acc=0.82,
    )
    r = load_trial_record(tmp_path / "result.json")
    assert metric_value(r, METRIC_VAL_ACC) == pytest.approx(0.82)


def test_metric_val_loss_missing_raises(tmp_path: Path):
    write_nas_result(
        tmp_path,
        trial_id=1,
        condition="spinal_canal_stenosis",
        include_best_metrics=False,
    )
    r = load_trial_record(tmp_path / "result.json")
    with pytest.raises(ValueError, match="val_loss"):
        metric_value(r, METRIC_VAL_LOSS)


def test_sort_and_top_trials_val_loss(tmp_path: Path):
    low = _record(
        tmp_path,
        "low",
        trial_id=1,
        condition="spinal_canal_stenosis",
        val_loss_at_best=0.1,
    )
    high = _record(
        tmp_path,
        "high",
        trial_id=2,
        condition="spinal_canal_stenosis",
        val_loss_at_best=0.9,
    )
    assert not higher_is_better(METRIC_VAL_LOSS)
    ordered = sort_trials([high, low], METRIC_VAL_LOSS)
    assert ordered[0].trial_id == 1
    assert top_trials([high, low], METRIC_VAL_LOSS, limit=1)[0].trial_id == 1


def test_best_shared_partial_conditions(tmp_path: Path):
    hp = dict(variant="vit_b_16", batch_size=8, lr=1e-4)
    r1 = _record(
        tmp_path,
        "c1",
        trial_id=1,
        condition="spinal_canal_stenosis",
        val_f1=0.5,
        **hp,
    )
    r2 = _record(
        tmp_path,
        "c2",
        trial_id=2,
        condition="left_neural_foraminal_narrowing",
        val_f1=0.6,
        **hp,
    )
    shared = best_shared_hyperparams(
        [r1, r2], METRIC_VAL_F1, require_all_conditions=False
    )
    assert shared is not None
    assert shared.trial_count == 2


def test_unknown_metric_raises(tmp_path: Path):
    r = _record(tmp_path, "a", trial_id=1, condition="spinal_canal_stenosis")
    with pytest.raises(ValueError, match="Unknown metric"):
        metric_value(r, "not_a_metric")
