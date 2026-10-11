"""2D slice → 5-level (x/W, y/H) samples, Avengers 1st-place style."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from rsna2024_lumbar.landmarks.constants import CONDITIONS, IMAGE_SIZE_DEFAULT, LEVEL_DISPLAY, LUMBAR_LEVELS
from rsna2024_lumbar.landmarks.split import split_study_ids
from rsna2024_lumbar.preprocessing.dicom_io import read_gray_uint8


@dataclass(frozen=True)
class LandmarkSample:
    study_id: int
    series_id: int
    instance_number: int
    path: Path
    xy01: np.ndarray  # (5, 2) normalized
    visible: np.ndarray  # (5,) 0/1
    width: int
    height: int


def _median_instance(rows: pd.DataFrame) -> int:
    return int(np.median(rows["instance_number"].astype(int).to_numpy()))


def list_condition_samples(
    data_root: Path,
    condition: str,
    *,
    study_ids: list[int] | None = None,
    image_size: int = IMAGE_SIZE_DEFAULT,
) -> list[LandmarkSample]:
    """One sample per study: the median labeled instance, all five level heads."""
    del image_size  # used at load time
    spec = CONDITIONS[condition]
    coord_name = str(spec["coord_name"])
    coords = pd.read_csv(Path(data_root) / "train_label_coordinates.csv")
    hit = coords[coords["condition"] == coord_name]
    if study_ids is not None:
        wanted = set(int(v) for v in study_ids)
        hit = hit[hit["study_id"].astype(int).isin(wanted)]
    images = Path(data_root) / "train_images"
    samples: list[LandmarkSample] = []
    for study_id, group in hit.groupby(hit["study_id"].astype(int)):
        if group.empty:
            continue
        instance = _median_instance(group)
        series_id = int(group.iloc[0].series_id)
        # Prefer the series of the median instance when levels share one series.
        same_inst = group[group["instance_number"].astype(int) == instance]
        if not same_inst.empty:
            series_id = int(same_inst.iloc[0].series_id)
        path = images / str(int(study_id)) / str(series_id) / f"{instance}.dcm"
        if not path.is_file():
            continue
        xy01 = np.zeros((len(LUMBAR_LEVELS), 2), dtype=np.float32)
        visible = np.zeros(len(LUMBAR_LEVELS), dtype=np.float32)
        width = height = 0
        for i, level in enumerate(LUMBAR_LEVELS):
            level_rows = group[group["level"] == LEVEL_DISPLAY[level]]
            if level_rows.empty:
                continue
            row = level_rows.iloc[0]
            if int(row.series_id) != series_id or int(row.instance_number) != instance:
                continue
            if width == 0:
                gray = read_gray_uint8(path)
                height, width = gray.shape[:2]
            xy01[i, 0] = float(row.x) / max(width, 1)
            xy01[i, 1] = float(row.y) / max(height, 1)
            visible[i] = 1.0
        if float(visible.sum()) <= 0.0:
            continue
        samples.append(
            LandmarkSample(
                study_id=int(study_id),
                series_id=series_id,
                instance_number=instance,
                path=path,
                xy01=xy01,
                visible=visible,
                width=width,
                height=height,
            )
        )
    return samples


def load_rgb_slice(path: Path, image_size: int) -> np.ndarray:
    """Windowed DICOM → (3, H, W) float32 in [0, 1], resized."""
    gray = read_gray_uint8(path)
    rgb = Image.fromarray(gray).convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def split_samples(
    samples: list[LandmarkSample],
    *,
    seed: int = 42,
) -> dict[str, list[LandmarkSample]]:
    ids = [s.study_id for s in samples]
    parts = split_study_ids(ids, seed=seed)
    by_id = {s.study_id: s for s in samples}
    return {name: [by_id[i] for i in part if i in by_id] for name, part in parts.items()}
