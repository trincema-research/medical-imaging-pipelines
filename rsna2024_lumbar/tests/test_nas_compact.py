"""Compact NAS history export."""

from __future__ import annotations

import json
from pathlib import Path

from rsna2024_lumbar.nas.compact import export_trial_dir, trial_config_from_result
from rsna2024_lumbar.nas.export_compact import main as export_main
from rsna2024_lumbar.tests.test_nas_best import _write_result


def test_trial_config_from_result(tmp_path: Path):
    _write_result(tmp_path, trial_id=3, condition="spinal_canal_stenosis", val_f1=0.5)
    config = trial_config_from_result(tmp_path / "result.json")
    assert config["trial_id"] == 3
    assert config["variant"] == "vit_b_16"
    assert "val_acc_at_best_epoch" in config


def test_export_trial_dir_copies_history(tmp_path: Path):
    _write_result(tmp_path / "trial_a", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.5)
    (tmp_path / "trial_a" / "training_history.csv").write_text("epoch,val_acc\n1,0.5\n", encoding="utf-8")
    dest = tmp_path / "out" / "trial_a"
    written = export_trial_dir(tmp_path / "trial_a", dest)
    assert (dest / "training_history.csv").is_file()
    assert (dest / "trial_config.json").is_file()
    assert len(written) >= 2


def test_export_compact_best_mode(tmp_path: Path):
    archive = tmp_path / "NAS results"
    root = archive / "Convnext" / "2d" / "spinal_canal_stenosis" / "trial_001_long_slug_ignored"
    _write_result(root, trial_id=1, condition="spinal_canal_stenosis", val_f1=0.9)
    payload = json.loads((root / "result.json").read_text(encoding="utf-8"))
    payload["model_type"] = "convnext"
    payload["max_val_acc"] = 0.99
    payload["val_acc"] = 0.99
    (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "training_history.csv").write_text("epoch,val_acc\n1,0.99\n", encoding="utf-8")
    (root / "best_epoch_metrics.json").write_text(
        json.dumps({"metrics": {"val_acc": 0.99, "val_loss": 0.1}}),
        encoding="utf-8",
    )

    out = tmp_path / "compact"
    export_main(
        [
            "--archive-root",
            str(archive),
            "--output-root",
            str(out),
            "--models",
            "convnext",
            "--layouts",
            "2d",
            "--mode",
            "best",
        ]
    )
    assert (
        out
        / "best"
        / "Convnext"
        / "2d"
        / "spinal_canal_stenosis"
        / "trial_0001"
        / "training_history.csv"
    ).is_file()
    assert (out / "best_configs" / "nas_best_convnext_2d.csv").is_file()
