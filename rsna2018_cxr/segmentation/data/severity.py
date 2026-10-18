"""Build one manifest row per study in the severity split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna2018_cxr.segmentation.constants import (
    DEFAULT_CLASS_DIRS,
    EXPECTED_COUNTS,
    MANIFEST_COLUMNS,
    SEVERITY_CLASSES,
)
from rsna2018_cxr.segmentation.data.images import image_hw, is_image_path
from rsna2018_cxr.segmentation.data.splits import assign_split


def iter_class_images(severity_root: Path, class_dirs: dict[str, str] | None = None):
    """Yield (severity_class, path) for every image under the class folders."""
    mapping = class_dirs or DEFAULT_CLASS_DIRS
    for severity in SEVERITY_CLASSES:
        relative = mapping[severity]
        folder = severity_root / relative
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and is_image_path(path):
                yield severity, path


def build_manifest(
    severity_root: Path,
    *,
    class_dirs: dict[str, str] | None = None,
    seed: int = 2026,
    read_shapes: bool = True,
    metadata_csv: Path | None = None,
) -> pd.DataFrame:
    """One row per study. Image shape is read unless ``read_shapes`` is false."""
    rows: list[dict] = []
    for severity, path in iter_class_images(severity_root, class_dirs):
        patient_id = path.stem
        width = height = None
        if read_shapes:
            height, width = image_hw(path)
        rows.append(
            {
                "patient_id": patient_id,
                "source_image_id": "",
                "severity_class": severity,
                "sex": "",
                "view_position": "",
                "image_path": str(path),
                "original_width": width,
                "original_height": height,
                "unet_mask_path": "",
                "classical_mask_path": "",
                "pair_dice_unet_classical": None,
                "mean_pairwise_dice": None,
                "min_pairwise_dice": None,
                "consensus_mask_path": "",
                "consensus_method": "",
                "label_status": "unlabeled",
                "manual_review_required": False,
                "manual_review_completed": False,
                "anatomy_qc_pass": None,
                "uncertainty_score": None,
                "active_learning_score": None,
                "split": assign_split(patient_id, seed=seed),
            }
        )
    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if metadata_csv is not None and Path(metadata_csv).is_file():
        frame = _merge_metadata(frame, Path(metadata_csv))
    return frame


def _merge_metadata(frame: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    extra = pd.read_csv(metadata_csv)
    columns = {c.lower(): c for c in extra.columns}
    id_col = columns.get("patient_id") or columns.get("patientid")
    if id_col is None:
        return frame
    keep = [id_col]
    rename = {id_col: "patient_id"}
    for target in ("sex", "view_position", "source_image_id"):
        source = columns.get(target)
        if source:
            keep.append(source)
            rename[source] = target
    extra = extra[keep].rename(columns=rename)
    extra["patient_id"] = extra["patient_id"].astype(str)
    frame = frame.drop(columns=[c for c in rename.values() if c != "patient_id"], errors="ignore")
    return frame.merge(extra, on="patient_id", how="left")


def count_by_class(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["severity_class"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in SEVERITY_CLASSES}


def count_mismatches(frame: pd.DataFrame, expected: dict[str, int] | None = None) -> dict[str, tuple[int, int]]:
    """Return {class: (actual, expected)} for classes that do not match."""
    target = expected or EXPECTED_COUNTS
    actual = count_by_class(frame)
    return {
        name: (actual[name], int(target[name]))
        for name in target
        if actual.get(name, 0) != int(target[name])
    }
