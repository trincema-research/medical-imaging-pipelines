from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna2024_lumbar.best_config_runs.severity_backfill import (
    backfill_severity_on_series,
    severity_from_val_confusion_matrices,
)


def test_severity_from_val_confusion_matrices(tmp_path: Path):
    for level in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"):
        (tmp_path / f"confusion_matrix_val_{level}.csv").write_text(
            "actual\\pred,Normal/Mild,Moderate,Severe\n"
            "Normal/Mild,10,0,0\n"
            "Moderate,1,8,1\n"
            "Severe,0,1,9\n",
            encoding="utf-8",
        )
    m = severity_from_val_confusion_matrices(tmp_path)
    assert m["val_oa_overall"] > 0.8
    assert 0 <= m["val_omae_overall"] <= 1.0


def test_backfill_oa_from_accuracy():
    row = pd.Series({"train_accuracy_overall": 0.88, "val_accuracy_overall": 0.9})
    out = backfill_severity_on_series(row, repeat_dir=None)
    assert out["train_oa_overall"] == 0.88
    assert out["val_oa_overall"] == 0.9
