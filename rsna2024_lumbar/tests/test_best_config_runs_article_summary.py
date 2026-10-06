from __future__ import annotations

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.article_summary import aggregate_pipeline_results


def test_aggregate_mean_std_pm():
    df = pd.DataFrame(
        [
            {
                "model_config": "nas_best_vit_2d",
                "model_label": "ViT 2D",
                "family": "vit",
                "archive_layout": "2d",
                "condition": "spinal_canal_stenosis",
                "repeat_index": 1,
                "val_acc": 0.90,
                "val_oa_overall": 0.91,
            },
            {
                "model_config": "nas_best_vit_2d",
                "model_label": "ViT 2D",
                "family": "vit",
                "archive_layout": "2d",
                "condition": "spinal_canal_stenosis",
                "repeat_index": 2,
                "val_acc": 0.92,
                "val_oa_overall": 0.93,
            },
        ]
    )
    out = aggregate_pipeline_results(df, metric_columns=("val_acc", "val_oa_overall"))
    assert len(out) == 1
    assert out.iloc[0]["n_repeats"] == 2
    assert out.iloc[0]["val_acc_mean"] == 0.91
    assert out.iloc[0]["val_acc_std"] == pytest.approx(0.02, abs=0.01)
    assert out.iloc[0]["val_acc_pm"] == "0.9100 ± 0.0141"
