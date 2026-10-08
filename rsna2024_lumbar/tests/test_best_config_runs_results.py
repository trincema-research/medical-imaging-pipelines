from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna2024_lumbar.best_config_runs.results import (
    PIPELINE_RESULTS_COLUMNS,
    PIPELINE_RESULTS_SPINAL_COLUMNS,
    row_from_training_metrics,
)


def test_row_from_training_metrics(tmp_path: Path):
    csv = tmp_path / "training_metrics.csv"
    csv.write_text(
        "epoch,val_acc,train_acc,val_acc,test_acc,"
        "train_accuracy_overall,val_accuracy_overall,test_accuracy_overall,"
        "val_oa_overall,val_omae_overall,"
        "val_accuracy_l1_l2,val_oa_L1_L2,val_f1_macro_l4_l5\n"
        "1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.1,0.4,0.4,0.3\n"
        "2,0.9,0.8,0.9,0.85,0.88,0.9,0.85,0.9,0.05,0.91,0.91,0.77\n",
        encoding="utf-8",
    )
    row = row_from_training_metrics(
        csv,
        meta={
            "model_config": "nas_best_vit_2d",
            "condition": "spinal_canal_stenosis",
            "repeat_index": 1,
            "split_seed": 42,
        },
    )
    assert row["best_epoch"] == 2
    assert row["val_oa_overall"] == 0.9
    assert row["val_omae_overall"] == 0.05
    assert row["val_accuracy_l1_l2"] == 0.91
    assert row["val_oa_L1_L2"] == 0.91
    assert row["val_f1_macro_l4_l5"] == 0.77
    assert "val_accuracy_l1_l2" in PIPELINE_RESULTS_SPINAL_COLUMNS
    df = pd.DataFrame([row], columns=list(PIPELINE_RESULTS_COLUMNS))
    assert len(df.columns) == len(PIPELINE_RESULTS_COLUMNS)
