"""Landmark NAS grid, GPU shards, and cloud zip (no torch / no DICOM dump)."""

from __future__ import annotations

import json
import zipfile

import pytest

from rsna2024_lumbar.landmarks.cli import main as landmarks_main
from rsna2024_lumbar.landmarks.deploy import parse_args as parse_deploy_args
from rsna2024_lumbar.landmarks.launch import build_plan, format_plan, main as launch_main
from rsna2024_lumbar.landmarks.pack import collect_pack_files, main as pack_main
from rsna2024_lumbar.landmarks.models import empty_coordinate_metrics
from rsna2024_lumbar.landmarks.search import expand_trials
from rsna2024_lumbar.landmarks.train import write_split_metrics
from rsna2024_lumbar.landmarks.unzip import extract_zip
from rsna2024_lumbar.nas.gpus import covered_trial_ids
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS


def test_default_grid_is_120_trials_at_512():
    trials = expand_trials("spinal_canal_stenosis")
    assert len(trials) == 120
    assert trials[0].trial_id == 1
    assert trials[-1].trial_id == 120
    assert {t.image_size for t in trials} == {512}
    assert {t.batch_size for t in trials} == {8}
    slugs = {t.slug for t in trials}
    assert len(slugs) == 120
    assert any(t.backbone == "convnext_base" for t in trials)
    assert any(t.backbone == "efficientnet_v2_l" for t in trials)


def test_batch_size_override_is_not_a_search_axis():
    trials = expand_trials("spinal_canal_stenosis", batch_size=4)
    assert len(trials) == 120
    assert {t.batch_size for t in trials} == {4}
    assert all("_bs4_" in t.slug for t in trials)


def test_local_family_is_two_models_times_lr_and_wd():
    trials = expand_trials("spinal_canal_stenosis", family="local")
    assert len(trials) == 24
    assert {t.backbone for t in trials} == {"convnext_tiny", "efficientnet_v2_s"}
    assert {t.image_size for t in trials} == {512}
    assert {t.lr for t in trials} == {1e-4, 3e-4, 6e-4, 1e-3}
    assert {t.weight_decay for t in trials} == {0.003, 0.01, 0.05}


def test_convnext_family_searches_lr_and_weight_decay():
    trials = expand_trials("spinal_canal_stenosis", family="convnext")
    assert len(trials) == 60
    assert {t.backbone for t in trials} == {"convnext_tiny", "convnext_base"}
    assert {t.image_size for t in trials} == {512}
    assert {t.lr for t in trials} == {1e-5, 3e-5, 1e-4, 3e-4, 6e-4, 1e-3}
    assert {t.weight_decay for t in trials} == {0.001, 0.003, 0.01, 0.03, 0.05}


def test_efficientnet_family_has_s_and_l():
    trials = expand_trials("left_subarticular_stenosis", family="efficientnet")
    assert len(trials) == 60
    assert {t.backbone for t in trials} == {"efficientnet_v2_s", "efficientnet_v2_l"}
    assert {t.image_size for t in trials} == {512}


def test_same_grid_size_for_every_condition():
    counts = {c: len(expand_trials(c)) for c in (
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    )}
    assert set(counts.values()) == {120}


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
def test_shards_cover_all_full_trials(num_gpus):
    plan = build_plan("left_subarticular_stenosis", num_gpus, visible=8)
    assert plan.n_trials == 120
    assert plan.num_gpus == num_gpus
    ids = covered_trial_ids(plan.shards)
    assert ids == list(range(1, 121))
    assert all(shard.n_trials >= 1 for shard in plan.shards)


def test_eight_gpus_split_120_trials():
    plan = build_plan("spinal_canal_stenosis", 8, visible=8)
    assert [s.n_trials for s in plan.shards] == [15] * 8
    assert plan.shards[0].start_trial == 1
    assert plan.shards[-1].end_trial == 120
    text = format_plan(plan)
    assert "GPUs:       8" in text
    assert "GPU 0: trials 1-15" in text


def test_launch_cli_lists_grid(capsys):
    launch_main(["--list"])
    out = capsys.readouterr().out
    assert "120 trials per condition" in out
    assert "600 if all 5" in out
    assert "4×1×6×5 = 120" in out
    assert "batch      8" in out


def test_launch_cli_batch_size(capsys):
    launch_main(["--list", "--batch-size", "4"])
    out = capsys.readouterr().out
    assert "120 trials per condition" in out
    assert "batch      4" in out
    assert "_bs4_" in out


def test_launch_cli_dry_run(capsys):
    launch_main(["--num-gpus", "4", "--visible-gpus", "8"])
    out = capsys.readouterr().out
    assert "Trials:     120" in out
    assert "GPUs:       4" in out


def test_landmarks_cli_dispatches_nas_list(capsys):
    landmarks_main(["nas", "--list"])
    out = capsys.readouterr().out
    assert "120 trials per condition" in out


def test_launch_cli_lists_convnext_family(capsys):
    launch_main(["--list", "--family", "convnext"])
    out = capsys.readouterr().out
    assert "family=convnext  60 trials per condition" in out
    assert "2×1×6×5 = 60" in out


def test_deploy_forwards_family_and_gpu_count():
    args = parse_deploy_args(["--family", "local", "--num-gpus", "1"])
    assert args.family == "local"
    assert args.num_gpus == 1


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
    assert any(name.endswith("landmarks/configs/landmark_nas_convnext.json") for name in names)
    assert any(name.endswith("landmarks/configs/landmark_nas_efficientnet.json") for name in names)
    assert any(name.endswith("landmarks/configs/landmark_nas_local.json") for name in names)
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
    assert manifest["trials_per_condition"] == 120
    assert (repo / "unzip_lumbar_landmarks_cloud.py").is_file()


def test_write_split_metrics_has_condition_and_level_rows(tmp_path):
    splits = {name: empty_coordinate_metrics() for name in ("train", "val", "test")}
    written = write_split_metrics(tmp_path, "spinal_canal_stenosis", "convnext_tiny", splits)
    assert written["n_rows"] == 18
    blob = json.loads(written["json"].read_text(encoding="utf-8"))
    assert blob["condition"] == "spinal_canal_stenosis"
    assert set(blob["splits"]["test"]["levels"]) == {"l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"}
    assert blob["splits"]["test"]["levels"]["l3_l4"]["r2"] is None
    text = written["csv"].read_text(encoding="utf-8")
    assert "level_display" in text
    assert "L3/L4" in text
    assert "overall" in text
    assert "n_visible" in text
    assert "mae_x_px" in text
    assert "pck10" in text
