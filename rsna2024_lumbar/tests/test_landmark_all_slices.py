from pathlib import Path

import numpy as np

from rsna2024_lumbar.landmarks.cli import parse_args
from rsna2024_lumbar.landmarks.dataset import (
    LandmarkSample,
    list_condition_dicoms,
    list_dense_slice_samples,
    split_samples,
)
from rsna2024_lumbar.landmarks.densify import densify_coordinates
from rsna2024_lumbar.tests.test_landmark_densify import _axial_stack


def _sample(study_id: int, instance: int) -> LandmarkSample:
    return LandmarkSample(
        study_id=study_id,
        series_id=1,
        instance_number=instance,
        path=Path("x.dcm"),
        xy01=np.zeros((5, 2), dtype=np.float32),
        visible=np.ones(5, dtype=np.float32),
        width=64,
        height=64,
    )


def test_list_dense_slice_samples_one_per_dicom(tmp_path: Path):
    raw = _axial_stack(tmp_path / "raw")
    out = tmp_path / "dense"
    densify_coordinates(raw, out)
    samples = list_dense_slice_samples(
        raw,
        "left_subarticular_stenosis",
        out / "train_label_coordinates_dense.csv",
    )
    assert {s.instance_number for s in samples} == {1, 2, 3}
    for sample in samples:
        assert float(sample.visible.sum()) == 1.0
        assert abs(float(sample.xy01[2, 0]) - 0.5) < 1e-5


def test_list_condition_dicoms_walks_matching_series(tmp_path: Path):
    raw = _axial_stack(tmp_path / "raw")
    found = list_condition_dicoms(raw, "left_subarticular_stenosis")
    assert {(inst, path.name) for _study, _series, inst, path in found} == {
        (1, "1.dcm"),
        (2, "2.dcm"),
        (3, "3.dcm"),
    }
    assert list_condition_dicoms(raw, "spinal_canal_stenosis") == []


def test_split_samples_keeps_every_slice_of_a_study():
    samples = [_sample(1, i) for i in range(3)] + [_sample(2, i) for i in range(2)] + [_sample(3, 0)]
    parts = split_samples(samples, seed=0)
    assert sum(len(items) for items in parts.values()) == 6
    homes: dict[int, set[str]] = {}
    for name, items in parts.items():
        for sample in items:
            homes.setdefault(sample.study_id, set()).add(name)
    assert all(len(names) == 1 for names in homes.values())
    study1 = next(name for name, items in parts.items() if any(s.study_id == 1 for s in items))
    assert sum(1 for sample in parts[study1] if sample.study_id == 1) == 3


def test_cli_all_slices_and_per_slice():
    train = parse_args(
        [
            "train",
            "--condition",
            "spinal_canal_stenosis",
            "--all-slices",
            "--dense-csv",
            "dense.csv",
        ]
    )
    assert train.all_slices
    assert train.dense_csv == Path("dense.csv")
    infer = parse_args(["infer", "--checkpoint", "a.pt", "--output-csv", "b.csv", "--per-slice"])
    assert infer.per_slice
