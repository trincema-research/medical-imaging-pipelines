from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna2024_lumbar.nas.history_metrics import (
    enrich_row_from_compact_history,
    metrics_at_best_val_acc_row,
)


def test_metrics_at_best_val_acc_row_picks_max_val_acc(tmp_path: Path):
    hist = tmp_path / "training_history.csv"
    hist.write_text(
        "epoch,val_acc,train_loss,val_accuracy_overall,train_f1_macro_overall,train_f1_macro_levels\n"
        "1,0.5,1.0,0.5,0.4,0.3\n"
        "2,0.9,0.2,0.88,0.7,0.65\n",
        encoding="utf-8",
    )
    m = metrics_at_best_val_acc_row(hist)
    assert m["val_accuracy_overall"] == 0.88
    assert m["train_f1_macro_overall"] == 0.7
    assert m["train_loss"] == 0.2
    assert m["train_f1_macro_levels_at_best"] == 0.65


def test_enrich_row_from_compact_history(tmp_path: Path):
    compact = tmp_path / "compact"
    hist = (
        compact
        / "EfficientNet"
        / "2d"
        / "spinal_canal_stenosis"
        / "trial_0093"
        / "training_history.csv"
    )
    hist.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "val_acc": [0.5, 0.92],
            "test_accuracy_overall": [0.4, 0.91],
        }
    ).to_csv(hist, index=False)
    row = {
        "family": "efficientnet",
        "archive_layout": "2d",
        "condition": "spinal_canal_stenosis",
        "trial_id": 93,
    }
    enrich_row_from_compact_history(row, compact_root=compact)
    assert row["test_accuracy_overall"] == 0.91
