"""Invariants so the same code path works for 3 studies or ~2k."""

from rsna2024_lumbar.preprocessing.catalog import require_raw_layout
from rsna2024_lumbar.preprocessing.constants import LABEL_CSVS
from rsna2024_lumbar.preprocessing.export import export_crops


def test_raw_layout_is_the_kaggle_layout(mini_raw):
    require_raw_layout(mini_raw)
    for name in LABEL_CSVS:
        assert (mini_raw / name).is_file()
    assert (mini_raw / "train_images").is_dir()


def test_export_is_linear_in_jobs(mini_raw, tmp_path):
    one = export_crops(
        mini_raw,
        tmp_path / "one",
        crop_policy="centered",
        conditions=["spinal_canal_stenosis"],
        max_studies=1,
    )
    all_ = export_crops(
        mini_raw,
        tmp_path / "all",
        crop_policy="centered",
        conditions=["spinal_canal_stenosis"],
    )
    assert one["n_jobs"] == 5
    assert all_["n_jobs"] == 10
    assert all_["exported"] == 2 * one["exported"]
