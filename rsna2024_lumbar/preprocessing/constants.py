"""RSNA 2024 lumbar labels, levels, and named crop policies."""

from __future__ import annotations

from dataclasses import dataclass

LUMBAR_LEVELS: tuple[str, ...] = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")

LEVEL_DISPLAY: dict[str, str] = {
    "l1_l2": "L1/L2",
    "l2_l3": "L2/L3",
    "l3_l4": "L3/L4",
    "l4_l5": "L4/L5",
    "l5_s1": "L5/S1",
}

SEVERITY_CLASSES: tuple[str, ...] = ("Normal/Mild", "Moderate", "Severe")
SEVERITY_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_CLASSES)}

CROP_POLICY_CENTERED = "centered"
CROP_POLICY_EXTEND50 = "extend50"
CROP_POLICIES: tuple[str, ...] = (CROP_POLICY_CENTERED, CROP_POLICY_EXTEND50)

# Native crop is 12.5% of slice W/H, centered on the labeled (x, y).
BASE_CROP_FRACTION = 0.125
EXTEND50_EXTRA_OF_BASE = 0.5

CENTERED_PNG_SIZE = (64, 64)  # W, H
EXTEND50_PNG_SIZE = (96, 64)

# extend50 grows the base square toward the side of interest (image coords).
EXTEND50_SIDE_BY_CONDITION: dict[str, str] = {
    "spinal_canal_stenosis": "left",
    "left_neural_foraminal_narrowing": "left",
    "right_neural_foraminal_narrowing": "left",
    "left_subarticular_stenosis": "right",
    "right_subarticular_stenosis": "left",
}

CONDITIONS: dict[str, dict[str, object]] = {
    "spinal_canal_stenosis": {
        "coord_name": "Spinal Canal Stenosis",
        "series_keywords": ("sagittal t2",),
    },
    "left_neural_foraminal_narrowing": {
        "coord_name": "Left Neural Foraminal Narrowing",
        "series_keywords": ("sagittal t1",),
    },
    "right_neural_foraminal_narrowing": {
        "coord_name": "Right Neural Foraminal Narrowing",
        "series_keywords": ("sagittal t1",),
    },
    "left_subarticular_stenosis": {
        "coord_name": "Left Subarticular Stenosis",
        "series_keywords": ("axial t2",),
    },
    "right_subarticular_stenosis": {
        "coord_name": "Right Subarticular Stenosis",
        "series_keywords": ("axial t2",),
    },
}

LABEL_CSVS = (
    "train.csv",
    "train_label_coordinates.csv",
    "train_series_descriptions.csv",
)


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def slug(self) -> str:
        return f"{self.width}x{self.height}"


def png_size_for_policy(crop_policy: str) -> ImageSize:
    if crop_policy == CROP_POLICY_EXTEND50:
        return ImageSize(*EXTEND50_PNG_SIZE)
    if crop_policy == CROP_POLICY_CENTERED:
        return ImageSize(*CENTERED_PNG_SIZE)
    raise ValueError(f"Unknown crop policy {crop_policy!r}. Choose {CROP_POLICIES}.")


def label_column(condition: str, level: str) -> str:
    return f"{condition}_{level}"


def severity_dir_name(severity: str) -> str:
    return severity.replace("/", "_")
