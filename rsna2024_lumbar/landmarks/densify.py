"""Fill (x, y) on every slice of a labeled series.

Axial intuition: once a condition has a mark, the canal / foramen stays at
roughly the same in-plane pixel as you walk superior–inferior. Copying (x, y)
is the cheap version. The accurate version projects the labeled LPS point onto
each neighbour with ImagePositionPatient / ImageOrientationPatient /
PixelSpacing (same math as the patient cubes). If a DICOM has no geometry,
the pixel is copied.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from rsna2024_lumbar.landmarks.constants import (
    DENSE_COORDINATES_DIRNAME,
    DENSE_CSV_NAME,
    SOURCE_COPIED,
    SOURCE_LABELED,
    SOURCE_PROJECTED,
)
from rsna2024_lumbar.preprocessing.catalog import require_raw_layout
from rsna2024_lumbar.preprocessing.dicom_io import SliceGeometry, geometry_from_dataset
from rsna2024_lumbar.preprocessing.geometry import patient_to_pixel, pixel_to_patient, plane_offset_mm

HERE = Path(__file__).resolve().parents[1]
DEFAULT_RAW = HERE / "data" / "raw"
DEFAULT_OUTPUT = HERE / "data" / "processed" / DENSE_COORDINATES_DIRNAME

DENSE_FIELDS = (
    "study_id",
    "series_id",
    "instance_number",
    "condition",
    "level",
    "x",
    "y",
    "source",
    "offset_mm",
    "in_plane",
    "rows",
    "columns",
    "anchor_instance",
)


def _instance_sort(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def _header_geometry(path: Path) -> SliceGeometry | None:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return geometry_from_dataset(ds)


def _list_series_dicoms(images_root: Path, study_id: int, series_id: int) -> list[Path]:
    folder = images_root / str(int(study_id)) / str(int(series_id))
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.dcm"), key=_instance_sort)


def densify_anchor(
    images_root: Path,
    study_id: int,
    series_id: int,
    instance_number: int,
    x: float,
    y: float,
    condition: str,
    level: str,
) -> list[dict]:
    """Project one labeled mark onto every slice in that series."""
    paths = _list_series_dicoms(images_root, study_id, series_id)
    if not paths:
        return []
    labeled_path = images_root / str(int(study_id)) / str(int(series_id)) / f"{int(instance_number)}.dcm"
    if not labeled_path.is_file():
        labeled_path = next((p for p in paths if _instance_sort(p) == int(instance_number)), None)
    if labeled_path is None or not labeled_path.is_file():
        return []
    labeled_geom = _header_geometry(labeled_path)
    xyz = None
    if labeled_geom is not None:
        xyz = pixel_to_patient(
            np.array(labeled_geom.ipp),
            np.array(labeled_geom.iop),
            labeled_geom.pixel_spacing,
            float(x),
            float(y),
        )
    rows: list[dict] = []
    for path in paths:
        inst = _instance_sort(path)
        geom = _header_geometry(path)
        source = SOURCE_COPIED
        px, py = float(x), float(y)
        offset = 0.0
        height = width = 0
        if geom is not None:
            height, width = geom.rows, geom.columns
            if xyz is not None:
                px, py = patient_to_pixel(np.array(geom.ipp), np.array(geom.iop), geom.pixel_spacing, xyz)
                offset = plane_offset_mm(np.array(geom.ipp), np.array(geom.iop), xyz)
                source = SOURCE_LABELED if inst == int(instance_number) else SOURCE_PROJECTED
            elif inst == int(instance_number):
                source = SOURCE_LABELED
        elif inst == int(instance_number):
            source = SOURCE_LABELED
        in_plane = True
        if width > 0 and height > 0:
            in_plane = (-0.5 <= px < width + 0.5) and (-0.5 <= py < height + 0.5)
        rows.append(
            {
                "study_id": int(study_id),
                "series_id": int(series_id),
                "instance_number": inst,
                "condition": condition,
                "level": level,
                "x": float(px),
                "y": float(py),
                "source": source,
                "offset_mm": float(offset),
                "in_plane": bool(in_plane),
                "rows": int(height),
                "columns": int(width),
                "anchor_instance": int(instance_number),
            }
        )
    return rows


def densify_coordinates(
    data_root: Path,
    output_root: Path,
    *,
    max_studies: int | None = None,
    study_ids: list[int] | None = None,
    progress: bool = False,
) -> dict:
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    require_raw_layout(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    coords = pd.read_csv(data_root / "train_label_coordinates.csv")
    images_root = data_root / "train_images"
    ids = study_ids if study_ids is not None else coords["study_id"].astype(int).drop_duplicates().tolist()
    if max_studies is not None:
        ids = ids[: max(0, int(max_studies))]
    iterator: list[int] | object = ids
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(ids, desc="Densify coordinates", unit="study")
        except ImportError:
            pass

    written: list[dict] = []
    n_labeled = n_projected = n_copied = 0
    for study_id in iterator:
        hit = coords[coords["study_id"] == int(study_id)]
        for row in hit.itertuples(index=False):
            extra = densify_anchor(
                images_root,
                int(row.study_id),
                int(row.series_id),
                int(row.instance_number),
                float(row.x),
                float(row.y),
                str(row.condition),
                str(row.level),
            )
            for item in extra:
                if item["source"] == SOURCE_LABELED:
                    n_labeled += 1
                elif item["source"] == SOURCE_PROJECTED:
                    n_projected += 1
                else:
                    n_copied += 1
            written.extend(extra)

    csv_path = output_root / DENSE_CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(DENSE_FIELDS))
        writer.writeheader()
        writer.writerows(written)
    meta = {
        "n_rows": len(written),
        "n_studies": len(ids),
        "n_labeled": n_labeled,
        "n_projected": n_projected,
        "n_copied": n_copied,
        "csv_path": csv_path.as_posix(),
        "note": "Projected rows inherit the labeled LPS point. Copied rows have no DICOM geometry.",
    }
    meta_path = output_root / "densify_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"output_root": output_root, "csv_path": csv_path, "meta_path": meta_path, **meta}
