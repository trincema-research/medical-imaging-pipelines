"""Turn the three Kaggle CSVs + train_images/ into a list of crop jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rsna2024_lumbar.preprocessing.constants import (
    CONDITIONS,
    LABEL_CSVS,
    LEVEL_DISPLAY,
    LUMBAR_LEVELS,
    SEVERITY_CLASSES,
    label_column,
)
from rsna2024_lumbar.preprocessing.crops import CropSettings, crop_settings_for


@dataclass(frozen=True)
class CropJob:
    condition: str
    study_id: int
    level: str
    severity: str
    dicom_path: Path
    x: float
    y: float
    crop_policy: str
    crop_settings: CropSettings


def require_raw_layout(data_root: Path) -> None:
    missing = [name for name in LABEL_CSVS if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {data_root}. "
            "Copy the Kaggle CSVs here (see rsna2024_lumbar/data/raw/README.md)."
        )
    if not (data_root / "train_images").is_dir():
        raise FileNotFoundError(f"Missing train_images/ under {data_root}.")


def _dicom_path(images_root: Path, study_id: int, series_id: int, instance: int) -> Path:
    return images_root / str(study_id) / str(series_id) / f"{instance}.dcm"


def _jobs_for_study(
    study_id: int,
    labels_row,
    coords_df: pd.DataFrame,
    images_root: Path,
    condition: str,
    crop_policy: str,
) -> list[CropJob] | None:
    """All five levels or None (drop incomplete studies — matches training)."""
    spec = CONDITIONS[condition]
    coord_name = str(spec["coord_name"])
    study_coords = coords_df[
        (coords_df["study_id"] == study_id) & (coords_df["condition"] == coord_name)
    ]
    if study_coords.empty:
        return None

    settings = crop_settings_for(condition, crop_policy)
    jobs: list[CropJob] = []
    for level in LUMBAR_LEVELS:
        raw = labels_row[label_column(condition, level)]
        if raw != raw or str(raw).strip() == "":
            return None
        severity = str(raw).strip()
        if severity not in SEVERITY_CLASSES:
            return None
        level_rows = study_coords[study_coords["level"] == LEVEL_DISPLAY[level]]
        if level_rows.empty:
            return None
        row = level_rows.iloc[0]
        path = _dicom_path(
            images_root,
            study_id,
            int(row.series_id),
            int(row.instance_number),
        )
        if not path.is_file():
            return None
        jobs.append(
            CropJob(
                condition=condition,
                study_id=int(study_id),
                level=level,
                severity=severity,
                dicom_path=path,
                x=float(row.x),
                y=float(row.y),
                crop_policy=crop_policy,
                crop_settings=settings,
            )
        )
    return jobs


def iter_crop_jobs(
    data_root: Path,
    *,
    crop_policy: str,
    conditions: list[str] | None = None,
    max_studies: int | None = None,
) -> list[CropJob]:
    require_raw_layout(data_root)
    labels = pd.read_csv(data_root / "train.csv")
    coords = pd.read_csv(data_root / "train_label_coordinates.csv")
    images_root = data_root / "train_images"
    study_ids = labels["study_id"].astype(int).tolist()
    if max_studies is not None:
        study_ids = study_ids[:max_studies]
    labels_ix = labels.set_index("study_id")
    wanted = conditions or list(CONDITIONS)

    jobs: list[CropJob] = []
    for condition in wanted:
        if condition not in CONDITIONS:
            raise KeyError(f"Unknown condition: {condition}")
        for study_id in study_ids:
            row = labels_ix.loc[study_id]
            batch = _jobs_for_study(
                study_id, row, coords, images_root, condition, crop_policy
            )
            if batch:
                jobs.extend(batch)
    return jobs
