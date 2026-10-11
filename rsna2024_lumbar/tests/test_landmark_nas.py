"""Landmark NAS grid, GPU shards, and cloud zip (no torch / no DICOM dump)."""

from __future__ import annotations

import json
import zipfile

import pytest

from rsna2024_lumbar.landmarks.cli import main as landmarks_main
from rsna2024_lumbar.landmarks.launch import build_plan, format_plan, main as launch_main
from rsna2024_lumbar.landmarks.pack import collect_pack_files, main as pack_main
from rsna2024_lumbar.landmarks.search import expand_trials
from rsna2024_lumbar.landmarks.unzip import extract_zip
from rsna2024_lumbar.nas.gpus import covered_trial_ids
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def test_default_grid_is_32_trials():
    trials = expand_trials("spinal_canal_stenosis")
    assert len(trials) == 32
    assert trials[0].trial_id == 1
    assert trials[-1].trial_id == 32
    slugs = {t.slug for t in trials}
    assert len(slugs) == 32
    assert any(t.backbone == "convnext_base" and t.image_size == 512 for t in trials)
    assert any(t.backbone == "efficientnet_v2_l" for t in trials)


def test_same_grid_size_for_every_condition():
    counts = {c: len(expand_trials(c)) for c in (
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    )}
    assert set(counts.values()) == {32}


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
def test_shards_cover_all_32_trials(num_gpus):
    plan = build_plan("left_subarticular_stenosis", num_gpus, visible=8)
    assert plan.n_trials == 32
    assert plan.num_gpus == num_gpus
    ids = covered_trial_ids(plan.shards)
    assert ids == list(range(1, 33))
    assert all(shard.n_trials >= 1 for shard in plan.shards)


def test_eight_gpus_get_four_trials_each():
    plan = build_plan("spinal_canal_stenosis", 8, visible=8)
    assert [s.n_trials for s in plan.shards] == [4] * 8
    assert plan.shards[0].start_trial == 1
    assert plan.shards[-1].end_trial == 32
    text = format_plan(plan)
    assert "GPUs:       8" in text
    assert "GPU 0: trials 1-4" in text


def test_launch_cli_lists_grid(capsys):
    launch_main(["--list"])
    out = capsys.readouterr().out
    assert "32 trials per condition" in out
    assert "160 if all 5" in out


def test_launch_cli_dry_run(capsys):
    launch_main(["--num-gpus", "4", "--visible-gpus", "8"])
    out = capsys.readouterr().out
    assert "Trials:     32" in out
    assert "GPUs:       4" in out


def test_landmarks_cli_dispatches_nas_list(capsys):
    landmarks_main(["nas", "--list"])
    out = capsys.readouterr().out
    assert "32 trials per condition" in out


def test_pack_zip_without_images(tmp_path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    for name in LABEL_CSVS:
        (data_root / name).write_text("study_id\n", encoding="utf-8")
    dest = tmp_path / "lumbar_landmarks_cloud.zip"
    pack_main(["--output", str(dest), "--data-root", str(data_root)])
    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
    assert "CLOUD_RUN.txt" in names
    assert "landmarks_deploy_manifest.json" in names
    assert "unzip_lumbar_landmarks_cloud.py" in names
    assert any(name.endswith("landmarks/search.py") for name in names)
    assert any(name.endswith("landmarks/configs/landmark_nas_search_space.json") for name in names)
    assert any(name.endswith("nas/gpus.py") for name in names)
    assert any(name.endswith("train_label_coordinates.csv") for name in names)
    assert not any("train_images/" in name for name in names)
    assert not any("patient_cubes" in name for name in names)


def test_collect_skips_dicoms_unless_requested(tmp_path):
    data_root = tmp_path / "raw"
    (data_root / "train_images" / "1" / "2").mkdir(parents=True)
    (data_root / "train_images" / "1" / "2" / "1.dcm").write_bytes(b"DICM")
    for name in LABEL_CSVS:
        (data_root / name).write_text("study_id\n", encoding="utf-8")
    files = collect_pack_files(data_root=data_root, include_images=False)
    assert not any(path.suffix == ".dcm" for path in files)
    files_on = collect_pack_files(data_root=data_root, include_images=True)
    assert any(path.suffix == ".dcm" for path in files_on)


def test_unzip_stdlib_roundtrip(tmp_path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    for name in LABEL_CSVS:
        (data_root / name).write_text("study_id\n", encoding="utf-8")
    dest = tmp_path / "pack.zip"
    pack_main(["--output", str(dest), "--data-root", str(data_root)])
    out = tmp_path / "extracted"
    repo = extract_zip(dest, out)
    assert (repo / "pyproject.toml").is_file()
    manifest = json.loads((repo / "landmarks_deploy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["include_images"] is False
    assert manifest["trials_per_condition"] == 32
    assert (repo / "unzip_lumbar_landmarks_cloud.py").is_file()
