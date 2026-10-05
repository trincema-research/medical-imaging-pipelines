from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.ordinal.config import iter_entries, load_best_config


def test_load_best_config_json(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "entries": [
            {"condition": "spinal_canal_stenosis", "hyperparameters": {"model_type": "vit"}},
            {"condition": "left_neural_foraminal_narrowing", "hyperparameters": {}},
        ],
    }
    path = tmp_path / "nas_best_vit_2d.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = load_best_config(path)
    assert len(list(iter_entries(cfg))) == 2
    assert len(list(iter_entries(cfg, condition="spinal_canal_stenosis"))) == 1
