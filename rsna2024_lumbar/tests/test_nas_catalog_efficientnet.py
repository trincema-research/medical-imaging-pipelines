from __future__ import annotations

from rsna2024_lumbar.nas.catalog import ARCHIVE_RESULT_TARGETS, MODEL_GROUPS, targets_for_groups


def test_efficientnet_in_catalog():
    assert "efficientnet" in MODEL_GROUPS
    labels = {t.label for t in ARCHIVE_RESULT_TARGETS}
    assert "EfficientNet 2D" in labels
    assert "EfficientNet 3D" in labels


def test_targets_for_efficientnet_only():
    targets = targets_for_groups(["efficientnet"])
    assert len(targets) == 2
    assert {t.layout for t in targets} == {"2d", "3d"}
