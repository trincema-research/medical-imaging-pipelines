# Completing lumbar (x, y) on every slice

`train_label_coordinates.csv` has one row per `(study_id, condition, level)` (48,692 marks). Neighbouring DICOMs in the same series have no click. This note is the Methods wording for filling those rows, then learning the same task the way the RSNA 2024 1st-place team did.

Implementation: [`landmarks/`](../landmarks/).

---

## 1. Axial copy, done with DICOM geometry

On an axial T2 series the spinal canal (and each foramen) stays at almost the same in-plane pixel as you move superior–inferior. If L3/L4 left-subarticular is marked at `(x, y)` on instance 19, instances 17–21 are the same canal a few millimetres away. Copying `(x, y)` is a good first guess.

The accurate version does not copy the integer pixel. It maps the mark to LPS

\[
\mathbf{p} = \mathbf{s} + x\,\Delta_c\,\mathbf{r} + y\,\Delta_r\,\mathbf{c}
\]

and projects \(\mathbf{p}\) onto every other slice (`patient_to_pixel`). Within one axial series that is usually a shift of <1 px. On a scoliotic or oblique stack it is the difference between a usable crop and a miss. If `ImagePositionPatient` is missing, the pipeline falls back to a raw copy.

Sagittal neighbours can use the same projection. In-plane drift is larger left–right than up–down, so treat far para-sagittal projections as low-confidence.

`densify` writes those rows to `data/processed/dense_coordinates/train_label_coordinates_dense.csv` (`source=labeled|projected|copied`, plus `offset_mm`). Do not commit the CSV.

---

## 2. What 1st place actually trained

[Avengers](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution) stage 1 created `test_label_coordinates.csv`:

1. Pick one 2D slice per study (median labeled instance of L1–S1).
2. Resize to **512×512**, three channels (the gray slice stacked).
3. **ConvNeXt-Base** and **EfficientNetV2-L** (ImageNet, then a public coordinate pretrain).
4. **Level-separated heads**: five `(x', y') = (x/W, y/H)` outputs, **L1** loss.
5. Mean ensemble. Separate nets for SCS / NFN / subarticular.

Regression is therefore ordinary 2D image regression: the network sees the pixel array of **one slice**, not a 3D volume. After that prediction, this repo projects the five points onto the rest of the series (section 1), which is the axial “copy up and down” step on top of the learned point.

2nd place reported 99.1% of disc points within 5 px. 3rd place used CenterNet heatmaps instead of direct `(x, y)`.

---

## 3. Five pipelines

One model per condition, five level heads. Study-level **70 / 15 / 15** split (`split_study_ids`). Do not train 25 level×condition nets.

| pipeline | series | backbone defaults |
|---|---|---|
| spinal canal stenosis | Sag T2 | `convnext_base` / `efficientnet_v2_l` |
| left / right foraminal | Sag T1 | same |
| left / right subarticular | Ax T2 | same |

Smoke backbone: `convnext_tiny` at 64 px.

---

## 4. Commands

```bash
python -m rsna2024_lumbar.landmarks densify --progress
python -m rsna2024_lumbar.landmarks train --condition spinal_canal_stenosis --backbone convnext_base --pretrained
python -m rsna2024_lumbar.landmarks train --condition spinal_canal_stenosis --all-slices --dense-csv rsna2024_lumbar/data/processed/dense_coordinates/train_label_coordinates_dense.csv
python -m rsna2024_lumbar.landmarks infer --checkpoint path/to.pt --output-csv path/to/pred.csv
python -m rsna2024_lumbar.landmarks infer --checkpoint path/to.pt --output-csv path/to/pred.csv --per-slice
```

Holdout studies and the unlabeled Kaggle test set are the later target for `infer`. Geometry-filled train slices are the cheap dense supervision for a second training round (`source=projected`, or `train --all-slices`). All-image layout (tensor, not a 262k-column table): [landmark_all_slice_pipeline.md](landmark_all_slice_pipeline.md).

---

## 5. Cloud zip and GPU NAS

Landmark training reads **DICOMs** (`train_images/`), not the PNG crop cache. Pack the scripts + label CSVs, and optionally the full image dump:

```bash
python -m rsna2024_lumbar.landmarks pack                         # code + labels
python -m rsna2024_lumbar.landmarks pack --include-images        # + train_images/ (ZIP_STORED for .dcm)
python unzip_lumbar_landmarks_cloud.py --zip lumbar_landmarks_cloud.zip --dest ./landmarks_cloud
python -m rsna2024_lumbar.landmarks deploy --num-gpus 8 --all-conditions --skip-completed
```

Search grid (`landmarks/configs/landmark_nas_search_space.json`): ConvNeXt-Tiny / Base + EfficientNetV2-S / L × lr `{1e-5, 1e-4, 3e-4, 1e-3}` × weight decay `{0.001, 0.01, 0.05}` at fixed **512×512** = **48 trials × 5 conditions**. Image size is not searched. Batch size is a CLI flag (`--batch-size`, default 8), not a search axis. Shard with the same formula as severity NAS: \(k=\lceil N/G\rceil\), one process per GPU, `CUDA_VISIBLE_DEVICES=g`. Eight GPUs → 6 trials each. `--all-conditions --spawn --wait` reuses the GPU set once per condition (sequential-by-condition, not 5× concurrent).

Do not commit `lumbar_landmarks_cloud.zip`, DICOMs, or `data/processed/dense_coordinates/nas/`.
