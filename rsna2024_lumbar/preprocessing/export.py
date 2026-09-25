"""Write cropped RGB PNGs + manifest.csv. Safe to resume with skip_existing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from rsna2024_lumbar.preprocessing.catalog import CropJob, iter_crop_jobs
from rsna2024_lumbar.preprocessing.constants import (
    CONDITIONS,
    SEVERITY_CLASSES,
    SEVERITY_TO_IDX,
    png_size_for_policy,
    severity_dir_name,
)
from rsna2024_lumbar.preprocessing.crops import crop_settings_for, describe_settings
from rsna2024_lumbar.preprocessing.dicom_io import crop_dicom


def png_relpath(job: CropJob) -> Path:
    return (
        Path(job.condition)
        / severity_dir_name(job.severity)
        / f"study_{job.study_id}_{job.level}.png"
    )


def export_crops(
    data_root: Path,
    output_root: Path,
    *,
    crop_policy: str,
    conditions: list[str] | None = None,
    max_studies: int | None = None,
    skip_existing: bool = False,
    progress: bool = False,
) -> dict:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_size = png_size_for_policy(crop_policy)
    jobs = iter_crop_jobs(
        data_root,
        crop_policy=crop_policy,
        conditions=conditions,
        max_studies=max_studies,
    )
    if not jobs:
        raise SystemExit(
            f"No complete studies under {data_root}. "
            "Need train.csv + coordinates + train_images/<study>/<series>/<n>.dcm."
        )

    iterator: list[CropJob] | object = jobs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(jobs, desc="Exporting PNGs", unit="crop")
        except ImportError:
            pass

    rows: list[dict] = []
    exported = skipped = 0
    for job in iterator:
        rel = png_relpath(job)
        dest = output_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "condition": job.condition,
            "study_id": job.study_id,
            "level": job.level,
            "severity": job.severity,
            "severity_idx": SEVERITY_TO_IDX[job.severity],
            "dicom_path": str(job.dicom_path),
            "x": job.x,
            "y": job.y,
            "crop_policy": job.crop_policy,
            "png_path": rel.as_posix(),
            "image_size": image_size.slug(),
        }
        if skip_existing and dest.is_file():
            skipped += 1
            rows.append({**record, "status": "skipped_existing"})
            continue
        img = crop_dicom(job.dicom_path, job.x, job.y, job.crop_settings)
        img = img.resize((image_size.width, image_size.height), resample=Image.LANCZOS)
        img.save(dest, format="PNG")
        exported += 1
        rows.append({**record, "status": "exported"})

    fieldnames = [
        "condition",
        "study_id",
        "level",
        "severity",
        "severity_idx",
        "dicom_path",
        "x",
        "y",
        "crop_policy",
        "png_path",
        "image_size",
        "status",
    ]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    wanted = conditions or list(CONDITIONS)
    meta = {
        "crop_policy": crop_policy,
        "image_size": image_size.slug(),
        "n_exported": exported,
        "n_skipped_existing": skipped,
        "n_jobs": len(jobs),
        "conditions": wanted,
        "crop_policy_by_condition": {
            key: describe_settings(crop_settings_for(key, crop_policy)) for key in wanted
        },
    }
    meta_path = output_root / "export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "output_root": output_root,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "exported": exported,
        "skipped": skipped,
        "n_jobs": len(jobs),
        "counts": _counts(rows),
    }


def _counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        cond = out.setdefault(row["condition"], {s: 0 for s in SEVERITY_CLASSES})
        cond[row["severity"]] += 1
    return out


def print_export_summary(result: dict) -> None:
    print(f"Output: {result['output_root']}")
    print(
        f"Exported: {result['exported']:,} | skipped: {result['skipped']:,} | "
        f"jobs: {result['n_jobs']:,}"
    )
    print(f"Manifest: {result['manifest_path']}")
    for condition, sev in result["counts"].items():
        print(f"  {condition}: " + ", ".join(f"{severity_dir_name(k)}={v}" for k, v in sev.items()))
