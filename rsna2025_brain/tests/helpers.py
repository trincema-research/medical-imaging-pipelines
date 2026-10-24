"""Synthetic Kaggle-shaped RSNA 2025 dump for download/unzip/validate tests."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from rsna2025_brain.constants import ID_COLUMN, LABEL_CSVS, LOCATION_COLUMNS, PRESENCE_COLUMN, SERIES_DIRNAME

UID_A = "1.2.840.1.test.a"
UID_B = "1.2.840.1.test.b"
SOP_A = "1.2.840.1.test.a.1"
SOP_B = "1.2.840.1.test.b.1"


def _train_header() -> list[str]:
    return [ID_COLUMN, "Modality", "PatientAge", "PatientSex", *LOCATION_COLUMNS, PRESENCE_COLUMN]


def _zero_locations() -> dict[str, str]:
    return {col: "0" for col in LOCATION_COLUMNS}


def write_label_csvs(dest: Path, *, both_series: bool = True) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            ID_COLUMN: UID_A,
            "Modality": "CTA",
            "PatientAge": "50",
            "PatientSex": "F",
            **_zero_locations(),
            PRESENCE_COLUMN: "0",
        }
    ]
    if both_series:
        loc = _zero_locations()
        loc["Left Middle Cerebral Artery"] = "1"
        rows.append(
            {
                ID_COLUMN: UID_B,
                "Modality": "MRA",
                "PatientAge": "60",
                "PatientSex": "M",
                **loc,
                PRESENCE_COLUMN: "1",
            }
        )
    with (dest / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_train_header())
        writer.writeheader()
        writer.writerows(rows)
    with (dest / "train_localizers.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["SeriesInstanceUID", "SOPInstanceUID", "coordinates", "location"]
        )
        writer.writeheader()
        if both_series:
            writer.writerow(
                {
                    "SeriesInstanceUID": UID_B,
                    "SOPInstanceUID": SOP_B,
                    "coordinates": '{"x": 12, "y": 34}',
                    "location": "Left Middle Cerebral Artery",
                }
            )


def write_series(dest: Path, *, uids: tuple[str, ...] = (UID_A, UID_B)) -> None:
    sops = {UID_A: SOP_A, UID_B: SOP_B}
    for uid in uids:
        folder = dest / SERIES_DIRNAME / uid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{sops[uid]}.dcm").write_bytes(b"dicom")


def write_extracted_dump(dest: Path) -> Path:
    dest = Path(dest)
    write_label_csvs(dest)
    write_series(dest)
    return dest


def write_dump_zip(archive: Path) -> Path:
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = archive.parent / "_staging_brain_dump"
    if staging.exists():
        for path in staging.rglob("*"):
            if path.is_file():
                path.unlink()
    write_extracted_dump(staging)
    with zipfile.ZipFile(archive, "w") as zf:
        for name in LABEL_CSVS:
            zf.write(staging / name, name)
        for path in (staging / SERIES_DIRNAME).rglob("*.dcm"):
            zf.write(path, path.relative_to(staging).as_posix())
    return archive
