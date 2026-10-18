"""Run candidate masks, consensus, and silver rules on the severity cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from rsna2018_cxr.segmentation.active_learning.ranking import active_learning_score, rank_review_queue
from rsna2018_cxr.segmentation.data.images import read_grayscale, transform_record
from rsna2018_cxr.segmentation.masks.agreement import area_difference, mean_and_min_dice, pairwise
from rsna2018_cxr.segmentation.masks.anatomy import anatomy_qc
from rsna2018_cxr.segmentation.masks.consensus import fuse
from rsna2018_cxr.segmentation.masks.postprocess import split_lungs
from rsna2018_cxr.segmentation.masks.silver import SilverRule, decide_label
from rsna2018_cxr.segmentation.models.classical import segment_classical
from rsna2018_cxr.segmentation.models.existing_unet import ExistingUNetBackend
from rsna2018_cxr.segmentation.models.types import LungPrediction


_CREATED_DIRS: set[Path] = set()


def _save_mask(path: Path, mask: np.ndarray) -> None:
    parent = path.parent
    if parent not in _CREATED_DIRS:
        parent.mkdir(parents=True, exist_ok=True)
        _CREATED_DIRS.add(parent)
    array = (np.asarray(mask) > 0).astype(np.uint8) * 255
    try:
        import cv2

        if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"failed to write {path}")
    except ImportError:
        Image.fromarray(array).save(path)


def _contour(mask: np.ndarray) -> np.ndarray:
    from rsna2018_cxr.segmentation.masks.agreement import boundary

    return boundary(mask)


def _overlay(gray: np.ndarray, consensus: np.ndarray, other: np.ndarray | None) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    agreed = _contour(consensus)
    rgb[agreed] = (0, 255, 0)
    if other is not None:
        only_other = _contour(other) & ~agreed
        rgb[only_other] = (255, 80, 0)
    return rgb


def _save_overlay(path: Path, gray: np.ndarray, consensus: np.ndarray, other: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_overlay(gray, consensus, other)).save(path)


def segment_cohort(
    manifest: pd.DataFrame,
    output_root: Path,
    *,
    unet_checkpoint: Path,
    unet_threshold: float = 0.5,
    batch_size: int = 4,
    fusion: str = "majority",
    use_classical: bool = True,
    inspection_per_class: int = 10,
    limit: int | None = None,
    silver: SilverRule | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Write original-resolution masks and update manifest label fields.

    Saved masks are ``pseudo``, ``silver``, or ``rejected``. None are marked gold.
    """
    rule = silver or SilverRule()
    backend = ExistingUNetBackend(
        unet_checkpoint,
        threshold=unet_threshold,
        device=device,
    )
    frame = manifest.copy().reset_index(drop=True)
    if limit is not None:
        frame = frame.iloc[: int(limit)].copy()
    output_root = Path(output_root)
    overlay_counts = {name: 0 for name in frame["severity_class"].unique()}
    provenance_path = output_root / "provenance.jsonl"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    rows = frame.to_dict(orient="records")
    with provenance_path.open("w", encoding="utf-8") as provenance:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            images = [read_grayscale(Path(row["image_path"])) for row in chunk]
            unet_preds = backend.predict_batch(images)
            for row, gray, unet_pred in zip(chunk, images, unet_preds):
                _update_row(
                    row,
                    gray,
                    unet_pred,
                    output_root=output_root,
                    fusion=fusion,
                    use_classical=use_classical,
                    rule=rule,
                    inspection_per_class=inspection_per_class,
                    overlay_counts=overlay_counts,
                    provenance=provenance,
                )
            done = min(start + batch_size, len(rows))
            print(f"segmented {done}/{len(rows)}", flush=True)
    updated = pd.DataFrame(rows)
    review = updated[updated["manual_review_required"].fillna(False).astype(bool)]
    if not review.empty and review["active_learning_score"].notna().any():
        queue = rank_review_queue(updated, per_class=max(1, inspection_per_class))
        queue_path = output_root / "review_queue.csv"
        queue.to_csv(queue_path, index=False)
    return updated


def _update_row(
    row: dict,
    gray: np.ndarray,
    unet_pred: LungPrediction,
    *,
    output_root: Path,
    fusion: str,
    use_classical: bool,
    rule: SilverRule,
    inspection_per_class: int,
    overlay_counts: dict,
    provenance,
) -> None:
    severity = row["severity_class"]
    patient_id = row["patient_id"]
    mask_dir = output_root / "masks" / severity
    unet_path = mask_dir / f"{patient_id}.png"
    consensus_path = mask_dir / f"{patient_id}_consensus.png"
    left_path = output_root / "left" / severity / f"{patient_id}.png"
    right_path = output_root / "right" / severity / f"{patient_id}.png"
    _save_mask(unet_path, unet_pred.merged_lungs)

    candidates = [unet_pred.merged_lungs]
    names = ["unet"]
    probabilities = [unet_pred.probability_map] if unet_pred.probability_map is not None else None
    inspect = overlay_counts.get(severity, 0) < inspection_per_class
    classical_pred = None
    if use_classical and inspect:
        classical_pred = segment_classical(gray)
        classical_path = mask_dir / f"{patient_id}_classical.png"
        _save_mask(classical_path, classical_pred.merged_lungs)
        row["classical_mask_path"] = str(classical_path)
        candidates.append(classical_pred.merged_lungs)
        names.append("classical")
        pair = pairwise(unet_pred.merged_lungs, classical_pred.merged_lungs)
        row["pair_dice_unet_classical"] = pair["dice"]

    if len(candidates) == 1:
        consensus, method = candidates[0], "unet_only"
        consensus_path = unet_path
    elif fusion == "probability" and probabilities and len(probabilities) == len(candidates):
        consensus, method = fuse(candidates, method="probability", probabilities=probabilities)
    elif len(candidates) == 2 and fusion == "majority":
        consensus = np.logical_and(candidates[0], candidates[1])
        method = "intersection"
    else:
        consensus, method = fuse(candidates, method=fusion)
    left, right = unet_pred.left_lung, unet_pred.right_lung
    if len(candidates) > 1:
        left, right = split_lungs(consensus, radiological=True)
    _save_mask(consensus_path, consensus) if consensus_path != unet_path else None
    _save_mask(left_path, left)
    _save_mask(right_path, right)

    mean_dice, min_dice = mean_and_min_dice(candidates)
    disagreement = area_difference(candidates[0], candidates[1]) if len(candidates) > 1 else 0.0
    qc = anatomy_qc(consensus, left, right)
    left_right_valid = "left_right_order" not in qc["warnings"] and "missing_side" not in qc["warnings"]
    label, review = decide_label(
        mean_dice=mean_dice if mean_dice == mean_dice else 0.0,
        min_dice=min_dice if min_dice == min_dice else 0.0,
        area_disagreement=disagreement,
        anatomy_pass=bool(qc["pass"]),
        left_right_valid=left_right_valid,
        rule=rule,
        hard_reject=not qc["pass"],
    )
    if len(candidates) < 2 and label == "silver":
        label, review = "pseudo", True
    score = active_learning_score(
        mean_dice if mean_dice == mean_dice else float("nan"),
        len(qc["warnings"]),
    )
    row.update(
        {
            "original_height": int(gray.shape[0]),
            "original_width": int(gray.shape[1]),
            "unet_mask_path": str(unet_path),
            "consensus_mask_path": str(consensus_path),
            "consensus_method": method,
            "mean_pairwise_dice": None if mean_dice != mean_dice else mean_dice,
            "min_pairwise_dice": None if min_dice != min_dice else min_dice,
            "label_status": label,
            "manual_review_required": review,
            "manual_review_completed": False,
            "anatomy_qc_pass": bool(qc["pass"]),
            "uncertainty_score": score,
            "active_learning_score": score,
        }
    )
    if overlay_counts.get(severity, 0) < inspection_per_class:
        overlay_path = output_root / "inspection" / severity / f"{patient_id}.png"
        other = classical_pred.merged_lungs if classical_pred is not None else None
        _save_overlay(overlay_path, gray, unet_pred.merged_lungs, other)
        overlay_counts[severity] = overlay_counts.get(severity, 0) + 1
        row["inspection_overlay"] = str(overlay_path)
    record = {
        "patient_id": patient_id,
        "severity": severity,
        "label_status": label,
        "human_review": False,
        "candidate_models": {
            "unet": unet_pred.metadata,
            "classical": None if classical_pred is None else {"model": classical_pred.model_name},
        },
        "agreement": {"mean_pairwise_dice": row["mean_pairwise_dice"], "min_pairwise_dice": row["min_pairwise_dice"]},
        "anatomy_qc": {"pass": qc["pass"], "hard": qc["hard"], "warnings": qc["warnings"]},
        "fusion": method,
        "transform": transform_record(patient_id, gray.shape[:2]),
        "models": names,
    }
    provenance.write(json.dumps(record) + "\n")
