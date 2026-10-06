"""Legacy NAS archive layout: ViT, MaxViT, ConvNeXt, EfficientNet × 2D / 3D."""

from __future__ import annotations

from dataclasses import dataclass

from rsna2024_lumbar.nas.families import (
    FAMILY_CONVNEXT,
    FAMILY_CONVNEXT3D,
    FAMILY_EFFICIENTNET,
    FAMILY_EFFICIENTNET3D,
    FAMILY_MAXVIT,
    FAMILY_MAXVIT3D,
    FAMILY_VIT,
    FAMILY_VIT3D,
)

MODEL_GROUP_VIT = "vit"
MODEL_GROUP_MAXVIT = "maxvit"
MODEL_GROUP_CONVNEXT = "convnext"
MODEL_GROUP_EFFICIENTNET = "efficientnet"

MODEL_GROUPS: tuple[str, ...] = (
    MODEL_GROUP_VIT,
    MODEL_GROUP_MAXVIT,
    MODEL_GROUP_CONVNEXT,
    MODEL_GROUP_EFFICIENTNET,
)


@dataclass(frozen=True)
class ArchiveResultTarget:
    """One scannable archive: family key + 2d/3d subfolder under ``NAS results/<Model>/``."""

    family: str
    layout: str
    model_group: str
    archive_folder: str
    label: str


# Families with completed legacy NAS exports (ViT, MaxViT, ConvNeXt).
ARCHIVE_RESULT_TARGETS: tuple[ArchiveResultTarget, ...] = (
    ArchiveResultTarget(
        family=FAMILY_VIT,
        layout="2d",
        model_group=MODEL_GROUP_VIT,
        archive_folder="ViT",
        label="ViT 2D",
    ),
    ArchiveResultTarget(
        family=FAMILY_VIT3D,
        layout="3d",
        model_group=MODEL_GROUP_VIT,
        archive_folder="ViT",
        label="ViT 3D",
    ),
    ArchiveResultTarget(
        family=FAMILY_MAXVIT,
        layout="2d",
        model_group=MODEL_GROUP_MAXVIT,
        archive_folder="MaxViT",
        label="MaxViT 2D",
    ),
    ArchiveResultTarget(
        family=FAMILY_MAXVIT3D,
        layout="3d",
        model_group=MODEL_GROUP_MAXVIT,
        archive_folder="MaxViT",
        label="MaxViT 3D",
    ),
    ArchiveResultTarget(
        family=FAMILY_CONVNEXT,
        layout="2d",
        model_group=MODEL_GROUP_CONVNEXT,
        archive_folder="Convnext",
        label="ConvNeXt 2D",
    ),
    ArchiveResultTarget(
        family=FAMILY_CONVNEXT3D,
        layout="3d",
        model_group=MODEL_GROUP_CONVNEXT,
        archive_folder="Convnext",
        label="ConvNeXt 3D",
    ),
    ArchiveResultTarget(
        family=FAMILY_EFFICIENTNET,
        layout="2d",
        model_group=MODEL_GROUP_EFFICIENTNET,
        archive_folder="EfficientNet",
        label="EfficientNet 2D",
    ),
    ArchiveResultTarget(
        family=FAMILY_EFFICIENTNET3D,
        layout="3d",
        model_group=MODEL_GROUP_EFFICIENTNET,
        archive_folder="EfficientNet",
        label="EfficientNet 3D",
    ),
)


def targets_for_groups(groups: list[str] | None) -> list[ArchiveResultTarget]:
    if not groups:
        return list(ARCHIVE_RESULT_TARGETS)
    wanted = {g.strip().lower() for g in groups}
    unknown = wanted - set(MODEL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown model group(s) {sorted(unknown)}. Choose {MODEL_GROUPS}.")
    return [t for t in ARCHIVE_RESULT_TARGETS if t.model_group in wanted]
