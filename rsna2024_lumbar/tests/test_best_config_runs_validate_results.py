from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.paths import DEFAULT_SPLIT_SEEDS, RESULTS_DIR
from rsna2024_lumbar.best_config_runs.results import PIPELINE_RESULTS_COLUMNS
from rsna2024_lumbar.best_config_runs.validate_results import (
    validate_pipeline_results_dataframe,
    validate_results_tree,
)


def _minimal_pipeline_row(**overrides) -> dict:
    row = {c: "" for c in PIPELINE_RESULTS_COLUMNS}
    row.update(
        {
            "model_config": "nas_best_vit_2d",
            "model_label": "ViT 2D",
            "family": "vit",
            "archive_layout": "2d",
            "condition": "spinal_canal_stenosis",
            "repeat_index": 1,
            "split_seed": DEFAULT_SPLIT_SEEDS[0],
            "nas_trial_id": "1",
            "best_epoch": 3,
            "train_acc": 0.8,
            "val_acc": 0.85,
            "test_acc": 0.84,
            "train_oa_overall": 0.8,
            "val_oa_overall": 0.85,
            "test_oa_overall": 0.84,
            "train_omae_overall": 0.1,
            "val_omae_overall": 0.12,
            "test_omae_overall": 0.11,
            "train_qwk_overall": 0.5,
            "val_qwk_overall": 0.55,
            "test_qwk_overall": 0.52,
            "train_ser_overall": 0.01,
            "val_ser_overall": 0.02,
            "test_ser_overall": 0.015,
            "run_dir": "nas_best_vit_2d/spinal_canal_stenosis/repeat_01",
        }
    )
    row.update(overrides)
    return row


def test_validate_pipeline_ok_single_row():
    df = pd.DataFrame([_minimal_pipeline_row()])
    report = validate_pipeline_results_dataframe(df, require_severity=True)
    assert report.ok


def test_validate_pipeline_missing_severity_fails():
    df = pd.DataFrame([_minimal_pipeline_row(val_oa_overall="")])
    report = validate_pipeline_results_dataframe(df, require_severity=True)
    assert not report.ok
    assert any(i.code == "missing_severity" for i in report.issues)


def test_validate_tree_with_audit_json(tmp_path: Path):
    model = tmp_path / "nas_best_vit_2d"
    repeat = model / "spinal_canal_stenosis" / "repeat_01"
    repeat.mkdir(parents=True)
    row = _minimal_pipeline_row()
    pd.DataFrame([row]).to_csv(model / "pipeline_results.csv", index=False)
    pd.DataFrame([row]).to_csv(tmp_path / "pipeline_results_all_models.csv", index=False)

    (repeat / "run_config.json").write_text(
        json.dumps({"condition": "spinal_canal_stenosis", "seed": DEFAULT_SPLIT_SEEDS[0]}),
        encoding="utf-8",
    )
    history = [
        {"epoch": 1, "val_acc": 0.5, "val_oa_overall": 0.5},
        {"epoch": 3, "val_acc": 0.85, "val_oa_overall": 0.85},
    ]
    (repeat / "training_history.json").write_text(json.dumps(history), encoding="utf-8")

    report = validate_results_tree(tmp_path, check_audit_files=True)
    assert report.ok


@pytest.mark.skipif(
    not (RESULTS_DIR / "pipeline_results_all_models.csv").is_file(),
    reason="committed results tree not present",
)
def test_committed_results_pass_default_validation():
    report = validate_results_tree(RESULTS_DIR, strict_full_models=False)
    assert report.ok, report.issues


@pytest.mark.skipif(
    not (RESULTS_DIR / "pipeline_results_all_models.csv").is_file(),
    reason="committed results tree not present",
)
def test_committed_full_models_strict():
    report = validate_results_tree(RESULTS_DIR, strict_full_models=True)
    assert report.ok, report.issues
