"""Unit tests for ``nas.report``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rsna2024_lumbar.nas.rank import METRIC_VAL_ACC, best_per_condition
from rsna2024_lumbar.nas.report import (
    CSV_FIELDNAMES,
    default_csv_path,
    rows_best_per_condition,
    rows_shared_config,
    write_best_csv,
)
from rsna2024_lumbar.nas.results import load_trial_record
from rsna2024_lumbar.nas.rank import best_shared_hyperparams, METRIC_VAL_F1
from rsna2024_lumbar.tests.nas_fixtures import write_nas_result


def test_default_csv_path():
    assert default_csv_path("vit") == Path("runs/nas_best_vit.csv")
    assert default_csv_path("vit", "2d") == Path("runs/nas_best_vit_2d.csv")


def test_rows_shared_config(tmp_path: Path):
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
        write_nas_result(
            tmp_path / condition,
            trial_id=i,
            condition=condition,
            val_f1=0.5 + i * 0.01,
            **hp,
        )
    records = [
        load_trial_record(p / "result.json") for p in tmp_path.iterdir() if p.is_dir()
    ]
    shared = best_shared_hyperparams(records, METRIC_VAL_F1)
    assert shared is not None
    rows = rows_shared_config(
        shared,
        family="vit",
        results_root=tmp_path,
        rank_metric=METRIC_VAL_F1,
    )
    assert len(rows) == 5
    assert rows[0]["selection"] == "shared_hyperparams"
    assert rows[0]["trainable_params"] == 100


def test_write_best_csv_fieldnames(tmp_path: Path):
    write_nas_result(
        tmp_path / "scs",
        trial_id=1,
        condition="spinal_canal_stenosis",
        val_f1=0.7,
    )
    records = [load_trial_record(tmp_path / "scs" / "result.json")]
    best = best_per_condition(records, METRIC_VAL_ACC)
    rows = rows_best_per_condition(
        best, family="vit", results_root=tmp_path, rank_metric=METRIC_VAL_ACC
    )
    out = tmp_path / "out.csv"
    write_best_csv(out, rows)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(CSV_FIELDNAMES)
        row = next(reader)
    assert row["total_params"] == "200"
    assert float(row["final_train_acc"]) == 0.55
    assert row["train_acc_at_best_epoch"] == "0.58"
    assert row["test_acc_at_best_epoch"] == "0.51"


def test_write_best_json(tmp_path: Path):
    from rsna2024_lumbar.nas.report import write_best_json

    write_nas_result(
        tmp_path / "scs",
        trial_id=1,
        condition="spinal_canal_stenosis",
        val_f1=0.7,
    )
    records = [load_trial_record(tmp_path / "scs" / "result.json")]
    best = best_per_condition(records, METRIC_VAL_ACC)
    rows = rows_best_per_condition(
        best, family="vit", results_root=tmp_path, rank_metric=METRIC_VAL_ACC
    )
    out = tmp_path / "best.json"
    write_best_json(out, rows, rank_metric=METRIC_VAL_ACC, archive_root=tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert "hyperparameters" in payload["entries"][0]
    assert payload["entries"][0]["metrics"]["train_acc_at_best_epoch"] == 0.58
