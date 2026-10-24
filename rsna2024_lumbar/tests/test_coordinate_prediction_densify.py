import csv
from pathlib import Path

import numpy as np

from rsna2024_lumbar.coordinate_prediction.cli import parse_args
from rsna2024_lumbar.coordinate_prediction.densify import densify_anchor, densify_coordinates
from rsna2024_lumbar.coordinate_prediction.split import split_study_ids
from rsna2024_lumbar.preprocessing.constants import label_column
from rsna2024_lumbar.tests.helpers import write_gray_dicom


AX_IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _axial_stack(root: Path) -> Path:
    """Three axials 4 mm apart; the middle slice is labeled at (32, 32)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    study_id, series_id = 7001, 8001
    size = 64
    xy = (32.0, 32.0)
    img = np.full((size, size), 100, dtype=np.uint16)
    img[31:34, 31:34] = 4000
    labels = {label_column("left_subarticular_stenosis", "l3_l4"): "Normal/Mild"}
    for other in (
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "right_subarticular_stenosis",
    ):
        for level in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"):
            labels.setdefault(label_column(other, level), "")
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study_id", *labels])
        w.writeheader()
        w.writerow({"study_id": study_id, **labels})
    with (root / "train_label_coordinates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["study_id", "series_id", "instance_number", "condition", "level", "x", "y"],
        )
        w.writeheader()
        w.writerow(
            {
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": 2,
                "condition": "Left Subarticular Stenosis",
                "level": "L3/L4",
                "x": xy[0],
                "y": xy[1],
            }
        )
    with (root / "train_series_descriptions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study_id", "series_id", "series_description"])
        w.writeheader()
        w.writerow({"study_id": study_id, "series_id": series_id, "series_description": "Axial T2"})
    for inst, z in ((1, -4.0), (2, 0.0), (3, 4.0)):
        write_gray_dicom(
            root / "train_images" / str(study_id) / str(series_id) / f"{inst}.dcm",
            img,
            instance_number=inst,
            ipp=(-32.0, -32.0, z),
            iop=AX_IOP,
            pixel_spacing=(1.0, 1.0),
        )
    return root


def test_split_study_ids_are_disjoint():
    parts = split_study_ids(list(range(20)), seed=0)
    assert len(parts["train"]) + len(parts["val"]) + len(parts["holdout"]) == 20
    assert set(parts["train"]).isdisjoint(parts["val"])
    assert set(parts["train"]).isdisjoint(parts["holdout"])


def test_axial_neighbours_keep_the_same_xy(tmp_path: Path):
    raw = _axial_stack(tmp_path / "raw")
    rows = densify_anchor(raw / "train_images", 7001, 8001, 2, 32.0, 32.0, "Left Subarticular Stenosis", "L3/L4")
    assert len(rows) == 3
    by_inst = {r["instance_number"]: r for r in rows}
    assert by_inst[2]["source"] == "labeled"
    assert by_inst[1]["source"] == "projected"
    assert by_inst[3]["source"] == "projected"
    assert abs(by_inst[1]["x"] - 32.0) < 1e-6
    assert abs(by_inst[3]["y"] - 32.0) < 1e-6
    assert abs(by_inst[1]["offset_mm"]) == 4.0


def test_densify_writes_csv(tmp_path: Path):
    raw = _axial_stack(tmp_path / "raw")
    out = tmp_path / "dense"
    result = densify_coordinates(raw, out)
    assert result["n_rows"] == 3
    assert result["n_projected"] == 2
    assert (out / "train_label_coordinates_dense.csv").is_file()


def test_cli_densify_defaults():
    args = parse_args(["densify"])
    assert args.command == "densify"
    assert args.max_studies is None
