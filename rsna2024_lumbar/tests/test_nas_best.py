"""NAS result ranking (synthetic result.json only)."""

from __future__ import annotations

import json
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


def _write_result(
    dest: Path,
    *,
    trial_id: int,
    condition: str,
    val_f1: float,
    variant: str = "vit_b_16",
    batch_size: int = 8,
    lr: float = 1e-4,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "condition": condition,
        "variant": variant,
        "image_width": 64,
        "image_height": 64,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": 0.01,
        "freeze_backbone": False,
        "use_pretrained": True,
        "amp": True,
        "head_depth": 1,
        "head_hidden_dim": None,
        "head_dropout": 0.0,
        "head_activation": "gelu",
        "optimizer_type": "AdamW",
        "scheduler_type": "CosineAnnealingLR",
        "final_train_acc": 0.5,
        "final_val_acc": 0.5,
        "final_test_acc": 0.5,
        "max_train_acc": 0.5,
        "max_val_acc": 0.5,
        "max_test_acc": 0.5,
        "best_epoch": 1,
        "duration_sec": 1.0,
        "output_dir": str(dest),
        "trainable_params": 1,
        "total_params": 1,
        "epochs_ran": 1,
        "val_f1_macro_levels": val_f1,
        "test_f1_macro_levels": val_f1,
        "checkpoint_path": "",
        "model_type": "vit",
        "input_layout": "2d",
    }
    (dest / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (dest / "best_epoch_metrics.json").write_text(
        json.dumps({"metrics": {"val_loss": 1.0 - val_f1, "val_acc": 0.5 + val_f1 * 0.1}}),
        encoding="utf-8",
    )


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
