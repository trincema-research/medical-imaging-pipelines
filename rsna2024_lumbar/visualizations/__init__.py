"""Publication figures from harvested best-config-run CSVs."""

from pathlib import Path

YEAR_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PATIENT_LEVEL_FIGURES_DIR = FIGURES_DIR / "patient_level"
SPINAL_LEVEL_FIGURES_DIR = FIGURES_DIR / "spinal_level"
ORDINAL_FIGURES_DIR = FIGURES_DIR / "ordinal"
CONDITION_SPECIFIC_FIGURES_DIR = FIGURES_DIR / "condition_specific"
DATASET_DESCRIPTION_FIGURES_DIR = FIGURES_DIR / "dataset_description"

__all__ = [
    "CONDITION_SPECIFIC_FIGURES_DIR",
    "DATASET_DESCRIPTION_FIGURES_DIR",
    "FIGURES_DIR",
    "ORDINAL_FIGURES_DIR",
    "PATIENT_LEVEL_FIGURES_DIR",
    "SPINAL_LEVEL_FIGURES_DIR",
    "YEAR_ROOT",
]
