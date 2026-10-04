"""Training bundle staging and worker command line (no GPU)."""

from __future__ import annotations

import argparse

import pytest

from rsna2024_lumbar.nas.bundle import default_legacy_root, stage_training_bundle
from rsna2024_lumbar.nas.paths import BUNDLE_DIR, bundle_ready
from rsna2024_lumbar.nas.worker import build_vit_nas_command


@pytest.mark.skipif(default_legacy_root() is None, reason="legacy lumbar repo not on this machine")
def test_stage_training_bundle_from_legacy():
    legacy = default_legacy_root()
    assert legacy is not None
    stage_training_bundle(legacy)
    assert bundle_ready()
    assert (BUNDLE_DIR / "configs" / "efficientnet_nas_search_space_pruned.json").is_file()


@pytest.mark.skipif(not bundle_ready(), reason="run test_stage_training_bundle_from_legacy first")
def test_worker_builds_vit_nas_command():
    from rsna2024_lumbar.nas.families import require_family

    spec = require_family("efficientnet")
    args = argparse.Namespace(
        family="efficientnet",
        condition="spinal_canal_stenosis",
        crop_policy="centered",
        start_trial=1,
        end_trial=2,
        output_base=None,
        data_root=None,
        crops_root=None,
        epochs=50,
        early_stop_patience=5,
        amp=True,
        no_amp=False,
        progress=True,
        no_progress=False,
        resume=True,
        no_resume=False,
    )
    cmd = build_vit_nas_command(args, spec)
    joined = " ".join(cmd)
    assert "vit_nas_lumbar.py" in joined
    assert "--image-source png" in joined
    assert "--start-trial 1" in joined
    assert "efficientnet_nas_search_space_pruned.json" in joined
