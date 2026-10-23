"""Predict 5-level (x, y) on a 2D slice, then project onto the rest of the series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna2024_lumbar.landmarks.constants import CONDITIONS, LEVEL_DISPLAY, LUMBAR_LEVELS, SOURCE_PREDICTED
from rsna2024_lumbar.landmarks.dataset import list_condition_dicoms, list_condition_samples, load_rgb_slice
from rsna2024_lumbar.landmarks.densify import densify_anchor
from rsna2024_lumbar.landmarks.models import build_regressor
from rsna2024_lumbar.preprocessing.dicom_io import read_gray_uint8


def load_checkpoint(path: Path, device: str | None = None):
    blob = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = build_regressor(blob["backbone"], pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


@torch.no_grad()
def predict_slice(model, path: Path, image_size: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image = torch.from_numpy(load_rgb_slice(path, image_size)).unsqueeze(0)
    xy01, vis = model(image)
    xy = xy01[0].cpu().numpy()
    xy[:, 0] *= float(width)
    xy[:, 1] *= float(height)
    return xy, vis[0].cpu().numpy()


def _prediction_row(
    *,
    study_id: int,
    series_id: int,
    instance_number: int,
    coord_name: str,
    level: str,
    x: float,
    y: float,
    height: int,
    width: int,
) -> dict:
    return {
        "study_id": study_id,
        "series_id": series_id,
        "instance_number": instance_number,
        "condition": coord_name,
        "level": level,
        "x": float(x),
        "y": float(y),
        "source": SOURCE_PREDICTED,
        "offset_mm": 0.0,
        "in_plane": True,
        "rows": height,
        "columns": width,
        "anchor_instance": instance_number,
    }


def predict_condition_csv(
    data_root: Path,
    checkpoint: Path,
    output_csv: Path,
    *,
    study_ids: list[int] | None = None,
    project_series: bool = True,
    per_slice: bool = False,
) -> dict:
    """Write predicted (and optionally series-projected) coordinates for one condition."""
    model, blob = load_checkpoint(checkpoint)
    condition = blob["condition"]
    image_size = int(blob["image_size"])
    images_root = Path(data_root) / "train_images"
    coord_name = str(CONDITIONS[condition]["coord_name"])
    rows: list[dict] = []
    if per_slice:
        targets = list_condition_dicoms(data_root, condition, study_ids=study_ids)
        n_studies = len({study_id for study_id, _series_id, _inst, _path in targets})
        for study_id, series_id, instance_number, path in targets:
            gray = read_gray_uint8(path)
            height, width = gray.shape[:2]
            xy, _vis = predict_slice(model, path, image_size, width, height)
            for i, level in enumerate(LUMBAR_LEVELS):
                rows.append(
                    _prediction_row(
                        study_id=study_id,
                        series_id=series_id,
                        instance_number=instance_number,
                        coord_name=coord_name,
                        level=LEVEL_DISPLAY[level],
                        x=float(xy[i, 0]),
                        y=float(xy[i, 1]),
                        height=height,
                        width=width,
                    )
                )
    else:
        samples = list_condition_samples(data_root, condition, study_ids=study_ids)
        n_studies = len(samples)
        for sample in samples:
            gray = read_gray_uint8(sample.path)
            height, width = gray.shape[:2]
            xy, _vis = predict_slice(model, sample.path, image_size, width, height)
            for i, level in enumerate(LUMBAR_LEVELS):
                if project_series:
                    extra = densify_anchor(
                        images_root,
                        sample.study_id,
                        sample.series_id,
                        sample.instance_number,
                        float(xy[i, 0]),
                        float(xy[i, 1]),
                        coord_name,
                        LEVEL_DISPLAY[level],
                    )
                    for item in extra:
                        item["source"] = (
                            SOURCE_PREDICTED
                            if item["instance_number"] == sample.instance_number
                            else item["source"]
                        )
                        if item["instance_number"] != sample.instance_number and item["source"] != SOURCE_PREDICTED:
                            item["source"] = f"predicted_{item['source']}"
                        rows.append(item)
                else:
                    rows.append(
                        _prediction_row(
                            study_id=sample.study_id,
                            series_id=sample.series_id,
                            instance_number=sample.instance_number,
                            coord_name=coord_name,
                            level=LEVEL_DISPLAY[level],
                            x=float(xy[i, 0]),
                            y=float(xy[i, 1]),
                            height=height,
                            width=width,
                        )
                    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    meta = {
        "n_rows": len(rows),
        "n_studies": n_studies,
        "checkpoint": str(checkpoint),
        "csv": str(output_csv),
        "per_slice": bool(per_slice),
    }
    output_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
