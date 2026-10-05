from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.ordinal.nas_history import summarize_entry_from_nas_history


def test_summarize_entry_from_compact_history(tmp_path: Path):
    compact = tmp_path / "nas_compact"
    hist = (
        compact
        / "EfficientNet"
        / "2d"
        / "spinal_canal_stenosis"
        / "trial_0001"
        / "training_history.csv"
    )
    hist.parent.mkdir(parents=True)
    hist.write_text(
        "epoch,val_acc,val_accuracy_overall\n1,0.5,0.5\n2,0.9,0.88\n",
        encoding="utf-8",
    )
    entry = {
        "family": "efficientnet",
        "archive_layout": "2d",
        "condition": "spinal_canal_stenosis",
        "trial_id": 1,
    }
    row = summarize_entry_from_nas_history(entry, compact_root=compact)
    assert row["best_epoch"] == 2
    assert row["val_oa_overall"] == 0.88
