"""Publication figures from harvested best-config-run CSVs."""

from pathlib import Path

YEAR_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PATIENT_LEVEL_FIGURES_DIR = FIGURES_DIR / "patient_level"

__all__ = [
    "FIGURES_DIR",
    "PATIENT_LEVEL_FIGURES_DIR",
    "YEAR_ROOT",
]
