"""NAS result ranking (synthetic result.json only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.nas.best import main as best_main
from rsna2024_lumbar.nas.rank import (
    METRIC_VAL_ACC,
    METRIC_VAL_F1,
    best_per_condition,
    best_shared_hyperparams,
    metric_value,
)
from rsna2024_lumbar.nas.results import load_trial_record
from rsna2024_lumbar.tests.nas_fixtures import write_nas_result as _write_result


def test_best_per_condition_picks_highest_val_acc(tmp_path: Path):
    _write_result(tmp_path / "a", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.9)
    _write_result(tmp_path / "b", trial_id=2, condition="spinal_canal_stenosis", val_f1=0.4)
    records = [load_trial_record(p / "result.json") for p in (tmp_path / "a", tmp_path / "b")]
    assert metric_value(records[0], METRIC_VAL_ACC) > metric_value(records[1], METRIC_VAL_ACC)
    best = best_per_condition(records, METRIC_VAL_ACC)
    assert best["spinal_canal_stenosis"].trial_id == 1


def test_best_per_condition_picks_highest_f1(tmp_path: Path):
    _write_result(tmp_path / "a", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.4)
    _write_result(tmp_path / "b", trial_id=2, condition="spinal_canal_stenosis", val_f1=0.9)
    records = [load_trial_record(p / "result.json") for p in (tmp_path / "a", tmp_path / "b")]
    best = best_per_condition(records, METRIC_VAL_F1)
    assert best["spinal_canal_stenosis"].trial_id == 2


def test_shared_config_requires_all_conditions(tmp_path: Path):
    hp = dict(variant="vit_b_16", batch_size=8, lr=1e-4)
    for i, condition in enumerate(
        [
            "spinal_canal_stenosis",
            "left_neural_foraminal_narrowing",
            "right_neural_foraminal_narrowing",
            "left_subarticular_stenosis",
            "right_subarticular_stenosis",
        ],
        start=1,
    ):
        _write_result(
            tmp_path / condition,
            trial_id=i,
            condition=condition,
            val_f1=0.5 + i * 0.01,
            **hp,
        )
    records = [load_trial_record(p / "result.json") for p in tmp_path.iterdir() if p.is_dir()]
    shared = best_shared_hyperparams(records, METRIC_VAL_F1)
    assert shared is not None
    assert shared.trial_count == 5


def test_write_best_csv(tmp_path: Path):
    from rsna2024_lumbar.nas.rank import METRIC_VAL_ACC, best_per_condition
    from rsna2024_lumbar.nas.report import rows_best_per_condition, write_best_csv
    import csv

    _write_result(tmp_path / "scs", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.7)
    records = [load_trial_record(tmp_path / "scs" / "result.json")]
    best = best_per_condition(records, METRIC_VAL_ACC)
    rows = rows_best_per_condition(
        best, family="vit", results_root=tmp_path, rank_metric=METRIC_VAL_ACC
    )
    out = tmp_path / "best.csv"
    write_best_csv(out, rows)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    assert row["condition"] == "spinal_canal_stenosis"
    assert row["variant"] == "vit_b_16"
    assert float(row["val_f1_macro_levels"]) == pytest.approx(0.7)


def test_best_cli_smoke(tmp_path: Path, capsys):
    _write_result(tmp_path / "scs", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.7)
    csv_path = tmp_path / "out.csv"
    best_main(
        [
            "--family",
            "vit",
            "--results-root",
            str(tmp_path),
            "--condition",
            "spinal_canal_stenosis",
            "--csv-out",
            str(csv_path),
        ]
    )
    out = capsys.readouterr().out
    assert "spinal_canal_stenosis" in out
    assert "trial" in out
    assert csv_path.is_file()
