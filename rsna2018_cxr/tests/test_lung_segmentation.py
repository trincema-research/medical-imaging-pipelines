"""Manifest, agreement, shape completion, and label-status tests. No GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rsna2018_cxr.segmentation.active_learning.ranking import rank_review_queue
from rsna2018_cxr.segmentation.cli import main
from rsna2018_cxr.segmentation.constants import MANIFEST_COLUMNS, NATIVE_HW
from rsna2018_cxr.segmentation.data.images import transform_record
from rsna2018_cxr.segmentation.data.severity import build_manifest, count_by_class, count_mismatches
from rsna2018_cxr.segmentation.data.splits import assert_no_patient_leakage, assign_split
from rsna2018_cxr.segmentation.masks.agreement import dice, iou, mean_and_min_dice
from rsna2018_cxr.segmentation.masks.anatomy import anatomy_qc, thresholds_from_percentiles
from rsna2018_cxr.segmentation.masks.consensus import majority_vote, staple
from rsna2018_cxr.segmentation.masks.postprocess import split_lungs
from rsna2018_cxr.segmentation.masks.silver import decide_label
from rsna2018_cxr.segmentation.training.evaluate import summarize_by_severity


def _png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=40).save(path)


def _two_lungs(height: int = 64, width: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = np.zeros((height, width), dtype=bool)
    merged[16:48, 8:28] = True
    merged[16:48, 36:56] = True
    left, right = split_lungs(merged, radiological=True)
    return merged, left, right


def test_manifest_columns_match_the_models_that_ran():
    assert "hybridgnet_mask_path" not in MANIFEST_COLUMNS
    assert "medsam_mask_path" not in MANIFEST_COLUMNS
    assert "xrv_mask_path" not in MANIFEST_COLUMNS
    assert "chexmask_available" not in MANIFEST_COLUMNS
    assert "unet_mask_path" in MANIFEST_COLUMNS
    assert "classical_mask_path" in MANIFEST_COLUMNS
    assert NATIVE_HW == (1024, 1024)


def test_manifest_counts_and_cli(tmp_path: Path):
    root = tmp_path / "severity"
    _png(root / "normal" / "patient-a.png")
    _png(root / "pneumonia" / "severity_1_low_mild" / "patient-b.jpg")
    frame = build_manifest(root)
    assert count_by_class(frame) == {"Normal": 1, "S1": 1, "S2": 0, "S3": 0}
    assert frame["label_status"].tolist() == ["unlabeled", "unlabeled"]
    assert count_mismatches(frame, {"Normal": 1, "S1": 1, "S2": 0, "S3": 0}) == {}
    mismatches = count_mismatches(frame)
    assert mismatches["Normal"] == (1, 8851)
    manifest = tmp_path / "master.csv"
    reports = tmp_path / "qc"
    assert main(
        [
            "build-manifest",
            "--severity-root",
            str(root),
            "--output",
            str(manifest),
            "--report-dir",
            str(reports),
            "--allow-count-mismatch",
        ]
    ) == 0
    report = (reports / "cohort_counts.md").read_text(encoding="utf-8")
    assert "not clinical severity" in report
    assert "| Normal | 1 |" in report
    assert main(
        [
            "build-manifest",
            "--severity-root",
            str(root),
            "--output",
            str(manifest),
            "--report-dir",
            str(reports),
        ]
    ) == 1


def test_native_resolution_is_not_resampled():
    from rsna2018_cxr.segmentation.data.images import assert_native_resolution

    native = np.zeros((1024, 1024), dtype=np.uint8)
    assert_native_resolution(native)
    with pytest.raises(ValueError):
        assert_native_resolution(np.zeros((256, 256), dtype=np.uint8))
    record = transform_record("patient-a", (1024, 1024))
    assert record["original_shape"] == [1024, 1024]
    assert record["model_input_shape"] == [1024, 1024]
    assert record["resize_method"] is None
    assert record["mask_resize_method"] is None
    with pytest.raises(ValueError):
        transform_record("patient-a", (256, 256))


def test_patient_split_does_not_leak():
    first = assign_split("patient-a", seed=2026)
    assert assign_split("patient-a", seed=2026) == first
    ids = ["patient-a", "patient-b", "patient-a"]
    splits = [assign_split(patient_id) for patient_id in ids]
    assert_no_patient_leakage(ids, splits)
    with pytest.raises(ValueError):
        assert_no_patient_leakage(["patient-a", "patient-a"], ["train", "test"])


def test_left_right_order_and_qc():
    merged, left, right = _two_lungs()
    assert left.sum() > 0 and right.sum() > 0
    assert np.nonzero(left)[1].mean() > np.nonzero(right)[1].mean()
    assert not np.logical_and(left, right).any()
    report = anatomy_qc(merged, left, right)
    assert report["pass"]
    assert report["principal_components"] == 2
    empty = np.zeros_like(merged)
    rejected = anatomy_qc(empty, empty, empty)
    assert not rejected["pass"]
    assert "empty" in rejected["hard"]
    band = thresholds_from_percentiles(np.linspace(0.2, 0.6, 50))
    assert band["soft_low"] < band["soft_high"]
    assert band["hard_low"] <= band["soft_low"]


def test_agreement_consensus_and_silver_is_not_gold():
    merged, _, _ = _two_lungs()
    assert dice(merged, merged) == 1.0
    mean_dice, min_dice = mean_and_min_dice([merged, merged, merged])
    assert mean_dice == 1.0 and min_dice == 1.0
    vote = majority_vote([merged, merged, np.zeros_like(merged)])
    assert np.array_equal(vote, merged)
    fused = staple([merged, merged, merged], iterations=5)
    assert iou(fused, merged) == 1.0
    label, review = decide_label(
        mean_dice=0.99,
        min_dice=0.98,
        area_disagreement=0.01,
        anatomy_pass=True,
        left_right_valid=True,
    )
    assert label == "silver" and review is False
    label, review = decide_label(
        mean_dice=0.5,
        min_dice=0.4,
        area_disagreement=0.2,
        anatomy_pass=True,
        left_right_valid=True,
    )
    assert label == "pseudo" and review is True
    label, _ = decide_label(
        mean_dice=1.0,
        min_dice=1.0,
        area_disagreement=0.0,
        anatomy_pass=False,
        left_right_valid=True,
        hard_reject=True,
    )
    assert label == "rejected"
    assert label != "gold"


def test_review_queue_is_balanced_by_severity():
    frame = pd.DataFrame(
        {
            "patient_id": ["n1", "n2", "s1", "s3"],
            "severity_class": ["Normal", "Normal", "S1", "S3"],
            "manual_review_required": [True, True, True, True],
            "active_learning_score": [0.1, 0.4, 0.2, 0.99],
        }
    )
    queue = rank_review_queue(frame, per_class=1)
    assert set(queue["severity_class"]) == {"Normal", "S1", "S3"}
    normal = queue.loc[queue["severity_class"] == "Normal", "patient_id"].item()
    assert normal == "n2"


def test_severity_delta_dice():
    frame = pd.DataFrame(
        {
            "severity_class": ["Normal", "Normal", "S3", "S3"],
            "dice": [0.90, 1.00, 0.70, 0.80],
        }
    )
    summary = summarize_by_severity(frame)
    normal = summary.loc[summary["severity_class"] == "Normal", "delta_dice_vs_normal"].item()
    severe = summary.loc[summary["severity_class"] == "S3", "delta_dice_vs_normal"].item()
    assert normal == 0.0
    assert severe == pytest.approx(0.20)


def _finished_lung(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, lateral: str) -> None:
    """A hemithorax whose lowest pixels are the lateral costophrenic angle."""
    width = max(1, x1 - x0)
    medial_end = y0 + int(0.70 * (y1 - y0))
    for y in range(y0, y1):
        if y <= medial_end:
            mask[y, x0:x1] = True
            continue
        remain = max(2, int(width * 0.30 * (1.0 - (y - medial_end) / max(1, y1 - medial_end))))
        if lateral == "left":
            mask[y, x0 : x0 + remain] = True
        else:
            mask[y, x1 - remain : x1] = True


def test_shape_completion_keeps_a_full_lung_and_extends_a_flat_cut():
    from rsna2018_cxr.segmentation.masks.shape_completion import LungTemplate, complete_lung_field

    field = np.zeros((32, 32), dtype=np.float32)
    field[:, 2:14] = 1.0
    field[:, 18:30] = 1.0
    template = LungTemplate(field=field, aspect=0.85, samples=8)
    gray = np.full((80, 100), 180, dtype=np.uint8)
    full = np.zeros((80, 100), dtype=bool)
    _finished_lung(full, 8, 34, 12, 78, "left")
    _finished_lung(full, 66, 94, 12, 78, "right")
    kept, reason = complete_lung_field(gray, full, template)
    assert reason == "keep"
    assert np.array_equal(kept, full)

    cut = full.copy()
    cut[48:, :] = False
    extended, reason = complete_lung_field(gray, cut, template)
    assert reason == "extend"
    assert extended[58:72, 16:28].any()
    assert extended[58:72, 72:86].any()
    assert not np.array_equal(extended, cut)


def test_shape_completion_mirrors_a_missing_side():
    from rsna2018_cxr.segmentation.masks.shape_completion import LungTemplate, complete_lung_field

    field = np.zeros((32, 32), dtype=np.float32)
    field[:, 2:14] = 1.0
    field[:, 18:30] = 1.0
    template = LungTemplate(field=field, aspect=0.85, samples=8)
    gray = np.full((80, 100), 180, dtype=np.uint8)
    only = np.zeros((80, 100), dtype=bool)
    _finished_lung(only, 62, 90, 12, 78, "right")
    completed, reason = complete_lung_field(gray, only, template)
    assert reason == "mirror"
    assert completed[:, :40].any()
    assert completed[:, 62:90].any()


def test_iteration1_vgg11_artifacts_are_present():
    root = Path(__file__).resolve().parents[1] / "data" / "iteration-1" / "vgg11"
    checkpoint = root / "unet_vgg11_1024_best.pth"
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size < 100_000_000
    history = (root / "metrics_history.csv").read_text(encoding="utf-8").splitlines()
    header = history[0].split(",")
    epoch_62 = history[62].split(",")
    assert header[0] == "epoch"
    assert epoch_62[0] == "62"
    assert float(epoch_62[header.index("val_dice")]) == pytest.approx(0.9753145575523376)
    splits = (root / "splits.csv").read_text(encoding="utf-8")
    assert splits.count("\ntrain,") == 99
    assert splits.count("\nval,") == 11
    assert splits.count("\ntest,") == 28
    assert len(list((root / "masks").glob("*.png"))) == 668


def test_train_refuses_a_non_native_size():
    with pytest.raises(SystemExit, match="1024"):
        main(["train", "--image-size", "256"])
