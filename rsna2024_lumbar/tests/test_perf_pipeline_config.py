from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.perf_pipeline.config import iter_entries, list_best_config_files, load_best_config


def test_load_best_config_json(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "entries": [{"condition": "spinal_canal_stenosis", "hyperparameters": {}}],
    }
    path = tmp_path / "nas_best_vit_2d.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(list(iter_entries(load_best_config(path)))) == 1


def test_list_best_config_skips_combined(tmp_path: Path):
    (tmp_path / "nas_best_all.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nas_best_vit_2d.json").write_text("{}", encoding="utf-8")
    names = [p.name for p in list_best_config_files(tmp_path)]
    assert "nas_best_vit_2d.json" in names
    assert "nas_best_all.json" not in names
