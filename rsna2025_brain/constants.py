"""Official RSNA 2025 Kaggle label columns (train.csv)."""

from __future__ import annotations

ID_COLUMN = "SeriesInstanceUID"
METADATA_COLUMNS: tuple[str, ...] = (ID_COLUMN, "PatientAge", "PatientSex", "Modality")

LOCATION_COLUMNS: tuple[str, ...] = (
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
)
PRESENCE_COLUMN = "Aneurysm Present"
TARGET_COLUMNS: tuple[str, ...] = (*LOCATION_COLUMNS, PRESENCE_COLUMN)

LABEL_CSVS: tuple[str, ...] = ("train.csv", "train_localizers.csv")
SERIES_DIRNAME = "series"
LOCALIZER_COLUMNS: tuple[str, ...] = (
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "coordinates",
    "location",
)
