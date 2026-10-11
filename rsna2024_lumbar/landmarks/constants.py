"""Landmark coordinate pipelines (1st-place style + DICOM projection)."""

from __future__ import annotations

from rsna2024_lumbar.preprocessing.constants import CONDITIONS, LEVEL_DISPLAY, LUMBAR_LEVELS

DENSE_COORDINATES_DIRNAME = "dense_coordinates"
DENSE_CSV_NAME = "train_label_coordinates_dense.csv"
SPLIT_SEED_DEFAULT = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
IMAGE_SIZE_DEFAULT = 512
IMAGE_SIZE_SMOKE = 64

# Avengers 1st place: ConvNeXt-Base + EfficientNetV2-L, level-separated (x/W, y/H) heads, L1.
BACKBONE_CONVNEXT_BASE = "convnext_base"
BACKBONE_CONVNEXT_TINY = "convnext_tiny"
BACKBONE_EFFNET_V2_L = "efficientnet_v2_l"
BACKBONE_EFFNET_V2_S = "efficientnet_v2_s"
BACKBONES = (
    BACKBONE_CONVNEXT_TINY,
    BACKBONE_CONVNEXT_BASE,
    BACKBONE_EFFNET_V2_S,
    BACKBONE_EFFNET_V2_L,
)

SOURCE_LABELED = "labeled"
SOURCE_PROJECTED = "projected"
SOURCE_COPIED = "copied"
SOURCE_PREDICTED = "predicted"

COORD_NAME_TO_CONDITION = {str(spec["coord_name"]): key for key, spec in CONDITIONS.items()}

__all__ = [
    "BACKBONES",
    "CONDITIONS",
    "COORD_NAME_TO_CONDITION",
    "DENSE_COORDINATES_DIRNAME",
    "DENSE_CSV_NAME",
    "IMAGE_SIZE_DEFAULT",
    "LEVEL_DISPLAY",
    "LUMBAR_LEVELS",
    "SPLIT_RATIOS",
    "SPLIT_SEED_DEFAULT",
]
