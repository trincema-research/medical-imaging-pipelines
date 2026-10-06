from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.best_config_runs.runner import run_best_config_pipeline


def test_run_best_config_dry_run(tmp_path: Path):
    cfg = {
        "label": "Test",
        "entries": [
            {
                "condition": "spinal_canal_stenosis",
                "family": "vit",
                "archive_layout": "2d",
                "trial_id": 1,
                "hyperparameters": {"model_type": "vit", "variant": "vit_b_16"},
            }
        ],
    }
    path = tmp_path / "nas_best_vit_2d.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    rc, rows = run_best_config_pipeline(
        path,
        repeats=2,
        epochs=1,
        output_base=tmp_path / "results",
        dry_run=True,
    )
    assert rc == 0
    assert rows == []
