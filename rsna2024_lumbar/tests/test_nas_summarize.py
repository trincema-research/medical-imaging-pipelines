"""Batch NAS summarize (synthetic archives)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsna2024_lumbar.nas.catalog import ARCHIVE_RESULT_TARGETS, targets_for_groups
from rsna2024_lumbar.nas.summarize import filter_targets, main as summarize_main
from rsna2024_lumbar.tests.nas_fixtures import write_nas_result as _write_result


def test_targets_for_groups():
    all_six = targets_for_groups(None)
    assert len(all_six) == len(ARCHIVE_RESULT_TARGETS) == 6
    vit_only = targets_for_groups(["vit"])
    assert len(vit_only) == 2
    assert {t.layout for t in vit_only} == {"2d", "3d"}


def test_targets_for_groups_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model group"):
        targets_for_groups(["not_a_model"])


def test_filter_targets_layouts():
    from rsna2024_lumbar.nas.summarize import parse_args

    args = parse_args(["--layouts", "2d", "--models", "convnext"])
    targets = filter_targets(args)
    assert len(targets) == 1
    assert targets[0].family == "convnext"


def test_summarize_writes_combined_csv(tmp_path: Path):
    archive = tmp_path / "NAS results"
    root = archive / "Convnext" / "2d" / "spinal_canal_stenosis"
    _write_result(root / "t1", trial_id=1, condition="spinal_canal_stenosis", val_f1=0.8)
    payload = json.loads((root / "t1" / "result.json").read_text(encoding="utf-8"))
    payload["model_type"] = "convnext"
    (root / "t1" / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out"
    summarize_main(
        [
            "--archive-root",
            str(archive),
            "--models",
            "convnext",
            "--layouts",
            "2d",
            "--csv-dir",
            str(out_dir),
            "--combined-csv",
            str(out_dir / "all.csv"),
        ]
    )
    assert (out_dir / "nas_best_convnext_2d.csv").is_file()
    assert (out_dir / "all.csv").is_file()
