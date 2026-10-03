"""Isolated NAS families: ViT, MaxViT, ConvNeXt, ConvNeXt3D, EfficientNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"

FAMILY_VIT = "vit"
FAMILY_MAXVIT = "maxvit"
FAMILY_CONVNEXT = "convnext"
FAMILY_CONVNEXT3D = "convnext3d"
FAMILY_EFFICIENTNET = "efficientnet"
FAMILIES = (
    FAMILY_VIT,
    FAMILY_MAXVIT,
    FAMILY_CONVNEXT,
    FAMILY_CONVNEXT3D,
    FAMILY_EFFICIENTNET,
)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    model_type: str
    config_file: Path
    output_base: str
    variant_key: str
    input_layout: str = "2d"


FAMILY_SPECS: dict[str, FamilySpec] = {
    FAMILY_VIT: FamilySpec(
        name=FAMILY_VIT,
        model_type="vit",
        config_file=CONFIGS / "vit_nas_search_space_8x8.json",
        output_base="runs/lumbar_nas",
        variant_key="vit_variants",
    ),
    FAMILY_MAXVIT: FamilySpec(
        name=FAMILY_MAXVIT,
        model_type="maxvit",
        config_file=CONFIGS / "maxvit_nas_search_space_8x8.json",
        output_base="runs/lumbar_nas_maxvit",
        variant_key="maxvit_variants",
    ),
    FAMILY_CONVNEXT: FamilySpec(
        name=FAMILY_CONVNEXT,
        model_type="convnext",
        config_file=CONFIGS / "convnext_nas_search_space_pruned.json",
        output_base="runs/lumbar_nas_convnext",
        variant_key="cnn2d_variants",
    ),
    FAMILY_CONVNEXT3D: FamilySpec(
        name=FAMILY_CONVNEXT3D,
        model_type="convnext3d",
        config_file=CONFIGS / "convnext3d_nas_search_space_192.json",
        output_base="runs/lumbar_nas_convnext3d",
        variant_key="cnn3d_variants",
        input_layout="3d_level_stack",
    ),
    FAMILY_EFFICIENTNET: FamilySpec(
        name=FAMILY_EFFICIENTNET,
        model_type="efficientnet",
        config_file=CONFIGS / "efficientnet_nas_search_space.json",
        output_base="runs/lumbar_nas_efficientnet",
        variant_key="effnet2d_variants",
    ),
}


def require_family(name: str) -> FamilySpec:
    key = name.strip().lower()
    if key not in FAMILY_SPECS:
        raise ValueError(f"Unknown NAS family {name!r}. Choose {FAMILIES}.")
    return FAMILY_SPECS[key]
