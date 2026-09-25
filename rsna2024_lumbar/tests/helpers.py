"""Build a tiny RSNA-shaped dump (synthetic DICOMs, no patient data)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from rsna2024_lumbar.preprocessing.constants import (
    CONDITIONS,
    LEVEL_DISPLAY,
    LUMBAR_LEVELS,
    label_column,
)

SLICE = 128
DOT_XY = (64.0, 64.0)


def write_gray_dicom(path: Path, pixels: np.ndarray, *, instance_number: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels, dtype=np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.Rows, ds.Columns = int(pixels.shape[0]), int(pixels.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.InstanceNumber = int(instance_number)
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def _dot_slice() -> np.ndarray:
    img = np.full((SLICE, SLICE), 100, dtype=np.uint16)
    x, y = int(DOT_XY[0]), int(DOT_XY[1])
    img[y - 1 : y + 2, x - 1 : x + 2] = 4000
    return img


def _blank_csvs(root: Path, studies: list[dict]) -> None:
    label_cols = [label_column(c, lv) for c in CONDITIONS for lv in LUMBAR_LEVELS]
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study_id", *label_cols])
        w.writeheader()
        for study in studies:
            row = {col: "" for col in label_cols}
            row["study_id"] = study["study_id"]
            row.update(study["labels"])
            w.writerow(row)

    with (root / "train_label_coordinates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["study_id", "series_id", "instance_number", "condition", "level", "x", "y"],
        )
        w.writeheader()
        for study in studies:
            w.writerows(study["coords"])

    with (root / "train_series_descriptions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study_id", "series_id", "series_description"])
        w.writeheader()
        for study in studies:
            w.writerow(
                {
                    "study_id": study["study_id"],
                    "series_id": study["series_id"],
                    "series_description": study["series_description"],
                }
            )


def _scs_study(
    study_id: int,
    series_id: int,
    *,
    labels: dict[str, str],
    skip_last_dicom: bool = False,
) -> dict:
    coords = []
    for i, level in enumerate(LUMBAR_LEVELS, start=1):
        coords.append(
            {
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": i,
                "condition": "Spinal Canal Stenosis",
                "level": LEVEL_DISPLAY[level],
                "x": DOT_XY[0],
                "y": DOT_XY[1],
            }
        )
    return {
        "study_id": study_id,
        "series_id": series_id,
        "series_description": "Sagittal T2",
        "labels": {label_column("spinal_canal_stenosis", lv): labels[lv] for lv in LUMBAR_LEVELS},
        "coords": coords,
        "skip_last_dicom": skip_last_dicom,
    }


def write_mini_dataset(root: Path) -> Path:
    """
    3 studies, SCS only, 80×80 slices with a bright dot at (40, 40):

    - 1001 complete (all Normal/Mild)
    - 1002 incomplete (no L5/S1 DICOM) → exporter must skip
    - 1003 complete (L4/L5 Moderate)
    """
    root = Path(root)
    images = root / "train_images"
    if images.exists():
        for p in images.rglob("*"):
            if p.is_file():
                p.unlink()
    studies = [
        _scs_study(1001, 2001, labels={lv: "Normal/Mild" for lv in LUMBAR_LEVELS}),
        _scs_study(
            1002,
            2002,
            labels={lv: "Normal/Mild" for lv in LUMBAR_LEVELS},
            skip_last_dicom=True,
        ),
        _scs_study(
            1003,
            2003,
            labels={
                **{lv: "Normal/Mild" for lv in LUMBAR_LEVELS},
                "l4_l5": "Moderate",
            },
        ),
    ]
    _blank_csvs(root, studies)
    pixels = _dot_slice()
    for study in studies:
        for i, _level in enumerate(LUMBAR_LEVELS, start=1):
            if study["skip_last_dicom"] and i == len(LUMBAR_LEVELS):
                continue
            dest = images / str(study["study_id"]) / str(study["series_id"]) / f"{i}.dcm"
            write_gray_dicom(dest, pixels, instance_number=i)
    return root


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent / "fixtures"
    write_mini_dataset(dest)
    print(f"Wrote mini dump to {dest}")
