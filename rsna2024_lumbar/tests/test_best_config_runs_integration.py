"""Integration tests for best_config_runs (repo best configs + nas_compact)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.cli import main as cli_main
from rsna2024_lumbar.best_config_runs.config import list_best_config_files
from rsna2024_lumbar.best_config_runs.nas_snapshot import (
    snapshot_all_best_configs,
    summarize_entry_from_nas_history,
    write_nas_snapshot_csv,
)
from rsna2024_lumbar.best_config_runs.paths import DEFAULT_BEST_CONFIG_DIR
from rsna2024_lumbar.best_config_runs.runner import resolve_repeat_seeds, run_best_config_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
BEST_CONFIG_DIR = REPO_ROOT / "rsna2024_lumbar" / "data" / "nas_compact" / "best_configs"
COMPACT_ROOT = REPO_ROOT / "rsna2024_lumbar" / "data" / "nas_compact"


@pytest.mark.skipif(not BEST_CONFIG_DIR.is_dir(), reason="best_configs not in checkout")
def test_repo_has_eight_per_model_best_configs():
    configs = list_best_config_files(BEST_CONFIG_DIR)
    assert len(configs) == 8
    stems = {p.stem for p in configs}
    assert "nas_best_vit_2d" in stems
    assert "nas_best_efficientnet_2d" in stems
    assert "nas_best_all" not in stems


@pytest.mark.skipif(not COMPACT_ROOT.is_dir(), reason="nas_compact not in checkout")
def test_nas_snapshot_vit_2d_from_repo(tmp_path: Path):
    cfg = BEST_CONFIG_DIR / "nas_best_vit_2d.json"
    if not cfg.is_file():
        pytest.skip("nas_best_vit_2d.json missing")
    rows = write_nas_snapshot_csv(
        cfg,
        compact_root=COMPACT_ROOT,
        output_csv=tmp_path / "snap.csv",
    )
    assert len(rows) == 5
    assert rows[0].get("val_oa_overall") not in ("", None)


def test_snapshot_all_best_configs_tmp(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    compact = tmp_path / "compact"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "nas_best_vit_2d.json"
    cfg_path.write_text(
        '{"label":"ViT 2D","entries":[{"family":"vit","archive_layout":"2d",'
        '"condition":"spinal_canal_stenosis","trial_id":1}]}',
        encoding="utf-8",
    )
    hist = (
        compact
        / "ViT"
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
    out = tmp_path / "results"
    rows = snapshot_all_best_configs(
        config_dir=cfg_dir,
        compact_root=compact,
        output_base=out,
    )
    assert len(rows) == 1
    assert (out / "nas_snapshots_all_models.csv").is_file()
    df = pd.read_csv(out / "nas_snapshots_all_models.csv")
    assert df.loc[0, "val_oa_overall"] == pytest.approx(0.88)


def test_cli_nas_snapshot_all_does_not_raise(tmp_path: Path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    compact = tmp_path / "compact"
    cfg_dir.mkdir()
    (cfg_dir / "nas_best_vit_2d.json").write_text(
        '{"entries":[{"family":"vit","archive_layout":"2d","condition":"c","trial_id":1}]}',
        encoding="utf-8",
    )
    hist = compact / "ViT" / "2d" / "c" / "trial_0001" / "training_history.csv"
    hist.parent.mkdir(parents=True)
    hist.write_text("epoch,val_acc,val_accuracy_overall\n1,0.8,0.8\n", encoding="utf-8")
    out = tmp_path / "results"
    with pytest.raises(SystemExit) as exc:
        cli_main(
            [
                "nas-snapshot-all",
                "--config-dir",
                str(cfg_dir),
                "--compact-root",
                str(compact),
                "--output-base",
                str(out),
            ]
        )
    assert exc.value.code == 0


def test_run_all_dry_run_uses_repo_configs():
    if not BEST_CONFIG_DIR.is_dir():
        pytest.skip("best_configs missing")
    rc, rows = run_best_config_pipeline(
        BEST_CONFIG_DIR / "nas_best_vit_2d.json",
        repeats=2,
        epochs=1,
        dry_run=True,
    )
    assert rc == 0
    assert rows == []


def test_resolve_repeat_seeds_defaults():
    assert resolve_repeat_seeds(5, None) == [42, 142, 242, 342, 442]
    assert resolve_repeat_seeds(2, [1, 2]) == [1, 2]


def test_summarize_entry_missing_history():
    row = summarize_entry_from_nas_history(
        {"family": "vit", "archive_layout": "2d", "condition": "c", "trial_id": 99},
        compact_root=Path("/nonexistent"),
    )
    assert "note" in row
