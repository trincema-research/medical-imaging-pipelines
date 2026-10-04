"""1/2/4/8 GPU sharding. CPU only — no torch."""

from __future__ import annotations

import zipfile

import pytest

from rsna2024_lumbar.nas.gpus import (
    ALLOWED_NAS_GPU_COUNTS,
    covered_trial_ids,
    resolve_num_gpus,
    shard_trial_range,
)
from rsna2024_lumbar.nas.launch import build_plan, format_plan, main as launch_main
from rsna2024_lumbar.nas.pack import collect_pack_files, main as pack_main
from rsna2024_lumbar.nas.search_space import expand_family


@pytest.mark.parametrize(
    "requested,visible,expected",
    [
        (0, 8, 8),
        (0, 6, 4),
        (0, 3, 2),
        (0, 1, 1),
        (1, 1, 1),
        (2, 4, 2),
        (4, 8, 4),
        (8, 8, 8),
    ],
)
def test_resolve_num_gpus(requested, visible, expected):
    assert resolve_num_gpus(requested, visible=visible) == expected


def test_resolve_rejects_disallowed_count():
    with pytest.raises(SystemExit, match="1, 2, 4, 8"):
        resolve_num_gpus(3, visible=8)


def test_resolve_rejects_more_than_visible():
    with pytest.raises(SystemExit, match="only 2"):
        resolve_num_gpus(8, visible=2)


def test_resolve_auto_with_no_gpu():
    with pytest.raises(SystemExit, match="No usable GPUs"):
        resolve_num_gpus(0, visible=0)


@pytest.mark.parametrize("n_trials,num_gpus", [
    (144, 1),
    (144, 2),
    (144, 4),
    (144, 8),
    (384, 1),
    (384, 2),
    (384, 4),
    (384, 8),
    (2304, 8),
    (1152, 8),
])
def test_shards_cover_every_trial_once(n_trials, num_gpus):
    shards = shard_trial_range(n_trials, num_gpus)
    assert {s.gpu_id for s in shards} <= set(range(num_gpus))
    ids = covered_trial_ids(shards)
    assert ids == list(range(1, n_trials + 1))
    assert len(shards) <= num_gpus
    assert all(shard.n_trials >= 1 for shard in shards)


def test_convnext_144_on_8_gpus_is_18_each():
    shards = shard_trial_range(144, 8)
    assert len(shards) == 8
    assert all(shard.n_trials == 18 for shard in shards)
    assert shards[0].start_trial == 1
    assert shards[-1].end_trial == 144


def test_allowed_gpu_counts():
    assert ALLOWED_NAS_GPU_COUNTS == (1, 2, 4, 8)


def test_launch_plan_efficientnet_4_gpus():
    plan = build_plan("efficientnet", condition="spinal_canal_stenosis", crop_policy="centered", num_gpus=4, visible=8)
    assert plan.n_trials == 144
    assert plan.num_gpus == 4
    assert len(plan.shards) == 4
    assert covered_trial_ids(plan.shards) == list(range(1, 145))
    text = format_plan(plan)
    assert "GPUs:        4" in text
    assert "GPU 0: trials 1-36" in text


def test_launch_cli_dry_run(capsys):
    launch_main(
        [
            "--family",
            "convnext3d",
            "--num-gpus",
            "8",
            "--visible-gpus",
            "8",
        ]
    )
    out = capsys.readouterr().out
    assert "Trials:      384 per condition" in out
    assert "GPUs:        8" in out
    assert "GPU 7: trials" in out


def test_launch_cli_rejects_bad_gpu_count():
    with pytest.raises(SystemExit, match="1, 2, 4, 8"):
        launch_main(["--family", "vit", "--num-gpus", "3", "--visible-gpus", "8"])


def test_pack_zip_without_crops(tmp_path):
    dest = tmp_path / "nas.zip"
    pack_main(
        [
            "--profile",
            "efficientnet_families",
            "--output",
            str(dest),
            "--skip-bundle",
        ]
    )
    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert "CLOUD_RUN.txt" in names
    assert "deploy_manifest.json" in names
    assert any(name.endswith("nas/gpus.py") for name in names)
    assert any(name.endswith("efficientnet3d_nas_search_space_192.json") for name in names)
    assert not any("data/processed/" in name for name in names)


def test_pack_collect_skips_processed_by_default():
    files = collect_pack_files(include_crops=False, crop_policy="centered", bundle_files=[])
    assert any(path.name == "gpus.py" for path in files)
    assert not any("processed" in path.parts and path.suffix == ".png" for path in files)


def test_family_grids_still_match_expected_counts():
    _, _, conv = expand_family("convnext")
    _, _, eff = expand_family("efficientnet")
    assert len(conv) == len(eff) == 144
