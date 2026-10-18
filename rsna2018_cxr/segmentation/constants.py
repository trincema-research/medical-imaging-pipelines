"""Shared names for the RSNA 2018 lung-segmentation cohort."""

from __future__ import annotations

# Annotation-derived burden strata from the Zenodo severity split.
# These are not independently validated clinical severity scores.
# Every study in the severity split is this size. Cohort ops stay on that grid.
NATIVE_HW = (1024, 1024)

SEVERITY_CLASSES = ("Normal", "S1", "S2", "S3")

EXPECTED_COUNTS = {
    "Normal": 8851,
    "S1": 1999,
    "S2": 2006,
    "S3": 2007,
}

# Relative to the severity-split root used locally:
#   normal/*.png
#   pneumonia/severity_*/*.(jpg|png)
DEFAULT_CLASS_DIRS = {
    "Normal": "normal",
    "S1": "pneumonia/severity_1_low_mild",
    "S2": "pneumonia/severity_2_moderate",
    "S3": "pneumonia/severity_3_severe",
}

LABEL_STATUSES = ("unlabeled", "pseudo", "silver", "gold", "rejected")

# Automatic stages may write pseudo, silver, or rejected. Gold is human review only.
AUTOMATIC_LABEL_STATUSES = ("pseudo", "silver", "rejected")

MANIFEST_COLUMNS = (
    "patient_id",
    "source_image_id",
    "severity_class",
    "sex",
    "view_position",
    "image_path",
    "original_width",
    "original_height",
    "unet_mask_path",
    "classical_mask_path",
    "pair_dice_unet_classical",
    "mean_pairwise_dice",
    "min_pairwise_dice",
    "consensus_mask_path",
    "consensus_method",
    "label_status",
    "manual_review_required",
    "manual_review_completed",
    "anatomy_qc_pass",
    "uncertainty_score",
    "active_learning_score",
    "split",
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
