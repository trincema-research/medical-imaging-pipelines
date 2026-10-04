"""Compact NAS history export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsna2024_lumbar.nas.compact import (
    compact_relative_dir,
    dest_layout_root,
    estimate_history_bytes,
    export_trial_dir,
    export_trials,
    trial_config_from_result,
)
from rsna2024_lumbar.nas.export_compact import main as export_main
from rsna2024_lumbar.nas.catalog import ARCHIVE_RESULT_TARGETS
from rsna2024_lumbar.tests.nas_fixtures import write_nas_result as _write_result


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


def test_compact_relative_dir_uses_trial_id(tmp_path: Path):
    trial = tmp_path / "long_original_folder_name"
    _write_result(trial, trial_id=42, condition="spinal_canal_stenosis")
    rel = compact_relative_dir(trial, tmp_path)
    assert rel == Path("spinal_canal_stenosis") / "trial_0042"


def test_export_trials_and_estimate(tmp_path: Path):
    archive = tmp_path / "NAS results"
    target = next(t for t in ARCHIVE_RESULT_TARGETS if t.family == "convnext")
    trial = archive / "Convnext" / "2d" / "spinal_canal_stenosis" / "trial_x"
    _write_result(trial, trial_id=1, condition="spinal_canal_stenosis", model_type="convnext")
    history = "epoch,val_acc\n1,0.9\n"
    (trial / "training_history.csv").write_text(history, encoding="utf-8")
    history_path = trial / "training_history.csv"
    count, size = estimate_history_bytes(archive, target)
    assert count == 1
    assert size == history_path.stat().st_size
    out = tmp_path / "out"
    stats = export_trials(target=target, archive_root=archive, output_root=out)
    assert stats["exported_trials"] == 1
    dest = dest_layout_root(out, target) / "spinal_canal_stenosis" / "trial_0001"
    assert (dest / "training_history.csv").is_file()


def test_export_trial_dir_missing_history_raises(tmp_path: Path):
    _write_result(tmp_path / "t", trial_id=1, condition="spinal_canal_stenosis")
    with pytest.raises(FileNotFoundError, match="training_history"):
        export_trial_dir(tmp_path / "t", tmp_path / "dest")
