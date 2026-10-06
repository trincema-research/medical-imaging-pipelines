from __future__ import annotations

from pathlib import Path

from rsna2024_lumbar.best_config_runs.train import build_train_command


def test_build_train_command_skips_empty_head_hidden_dim(tmp_path: Path):
    entry = {
        "condition": "spinal_canal_stenosis",
        "hyperparameters": {
            "model_type": "convnext3d",
            "input_layout": "3d_level_stack",
            "variant": "convnext_small",
            "head_hidden_dim": "",
        },
    }
    cmd = build_train_command(
        entry,
        data_root=tmp_path,
        crops_root=tmp_path,
        epochs=1,
        output_dir=tmp_path / "out",
        split_seed=42,
    )
    assert "--head-hidden-dim" not in cmd
    assert "--log-pipeline-metrics" in cmd
    assert "--save-checkpoints" not in cmd
