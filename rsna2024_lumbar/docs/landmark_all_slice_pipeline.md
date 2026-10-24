# Predicting (x, y) on every RSNA 2024 slice

This is the design for **lumbar landmark regression on all images**, not only the one Kaggle click per `(study, condition, level)`. It answers whether we should flatten each 512×512 MRI into 262,144 table columns. We should not.

Code: [`landmarks/`](../landmarks/). Commands: [`landmarks/README.md`](../landmarks/README.md). Geometry fill: [landmark_coordinates.md](landmark_coordinates.md).

---

## 1. Do not flatten the image into a giant table

A row of 262,144 pixel columns plus 5 \(x\) and 5 \(y\) targets is a spreadsheet view of a CNN problem. It is not what the competition used, and it is not used in keypoint papers (CenterNet, HRNet, U-Net heatmaps, YOLO).

| Flattened 512×512 table | 2D encoder (this repo / Avengers) |
|---|---|
| 262k independent features; spatial neighbours are lost | Tensor `(B, 3, 512, 512)`; convolution keeps nearby pixels together |
| One “model” cannot share filters across the spine | ImageNet ConvNeXt / EfficientNetV2 |
| RAM: one batch of 8 is ~16 million cells before the rest of the table | One 512×512 RGB float32 slice is 3 MB |
| Targets as 10 extra columns in the same table | Targets as a `(5, 2)` tensor of \((x/W,\, y/H)\) |

Resize to 512×512 **yes**. Store the result as a **tensor**, not as 262,144 CSV columns. The five lumbar levels are **five heads**, not five extra tables.

The CSV is the **label** (and a little metadata). It is not concatenated onto the pixels as extra features.

---

## 2. What “fusion” actually is here

Three things are fused. None of them is “pixels + CSV in one wide matrix.”

1. **Image.** One DICOM, windowed to uint8, resized to 512×512, gray repeated to 3 channels (ImageNet).
2. **Coordinates.** `train_label_coordinates.csv` is supervision: \((x, y)\) for each level that was clicked. After `densify`, neighbouring slices get a **projected** \((x, y)\) from DICOM LPS (`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`). That is the geometric fusion.
3. **Series metadata.** `train_series_descriptions.csv` (Sag T2 / Sag T1 / Ax T2) picks which of the five condition pipelines sees the slice. Instance number is used to pick or order slices, not as a 262k-wide covariate.

Optional later fusion (not required for v1): a small instance-number head (Avengers stage 1a), or 2.5D extra channels from \(\pm 1\) neighbours. Still tensors, still not a pixel table.

---

## 3. What the competition trained

| Place | Landmark method | Output |
|---|---|---|
| 1st Avengers | 2D ConvNeXt-Base + EfficientNetV2-L, **level-separated heads**, L1 on \((x/W,\, y/H)\). One 512×512 slice (median labeled instance). Then project / copy along the series. | Direct regression |
| 2nd | YOLOX boxes + 2D keypoint regression; coordinates **propagated across slices** that share a keypoint | Boxes + \((x, y)\) |
| 3rd | CenterNet-style **heatmaps** | Peak of a Gaussian map |
| 5th | Heatmap detection, Gaussian-expanded labels, 2D U-Net + sequence model | Heatmap |
| 7th | 2D and 3D keypoint regression (timm / timm-3d) | \((x,y)\) or \((x,y,z)\) |

Nobody trained a tabular model on flattened pixels. Direct L1 regression (what we already implement) and heatmaps are the two standard families. Heatmaps are a later option if PCK@5 stalls; they do not change the data layout.

---

## 4. Target layout (the 5+5 idea, done correctly)

Per **condition**, one forward pass:

\[
\hat{\mathbf{p}} \in [0,1]^{5\times 2},
\qquad
\hat p_{k} = (x_k/W,\, y_k/H),\quad k \in \{\mathrm{L1/L2},\ldots,\mathrm{L5/S1}\}.
\]

A visibility bit \(v_k\in\{0,1\}\) masks the L1 term when that level was not marked (or not visible) on that slice. That is 10 numbers + 5 bits, not 10 CSV columns next to 262k pixels.

Five **condition** nets (SCS, LFN, RFN, LSS, RSS), not 25 level×condition nets. Same as Avengers.

---

## 5. Two ways to cover every image

Kaggle labels **one** instance per `(study, condition, level)`. “All images” means the other DICOMs in that series (and, at test time, the hidden series).

| Stage | Who | What |
|---|---|---|
| A. Geometry densify | no GPU | Project each labeled LPS point onto every other instance (`source=labeled\|projected\|copied`). Writes `train_label_coordinates_dense.csv`. |
| B. 2D regressor on one slice | GPU | Avengers recipe: median labeled instance, five heads, L1. `list_condition_samples`. |
| C. 2D regressor on **every** slice | GPU | Same net, but one sample per DICOM using the dense CSV. `list_dense_slice_samples` + `train --all-slices`. |
| D. Infer on every image | GPU | `--per-slice`: run the net on each DICOM. Default infer still predicts the median slice and **projects** (stage A) to the rest. |

A then B is the cheap 1st-place clone. A then C is the all-image supervised path. D is how test studies (no CSV clicks) get a mark on every slice.

Study-level 70/15/15 (`split_study_ids`) applies in all cases. Never put two slices from the same study on both sides of the split.

---

## 6. Pipeline (v1)

```text
train_images/*.dcm
train_label_coordinates.csv
train_series_descriptions.csv
        │
        ▼
 densify  ──►  dense CSV  (every slice in a labeled series)
        │
        ├── train (default)     one 512×512 slice / study / condition
        └── train --all-slices  one 512×512 slice / instance / condition
        │
        ▼
 ConvNeXt / EfficientNetV2  →  (5, 2) sigmoid xy + 5 vis logits
        │
        ▼
 infer              predict median slice, project along series
 infer --per-slice  predict every DICOM
        │
        ▼
 pred_*.csv  →  severity crops / patient cubes
```

Loss (already in `models.landmark_loss`): masked L1 on \(\hat{\mathbf{p}}\) plus a small BCE on visibility.

Metrics: MAE in pixels, PCK@5 (fraction of visible points within 5 px). 2nd place reported 99.1% PCK@5 on disc points.

---

## 7. Commands

```bash
python -m rsna2024_lumbar.landmarks densify --progress

# B — one slice per study (Avengers)
python -m rsna2024_lumbar.landmarks train --condition spinal_canal_stenosis --backbone convnext_base --pretrained

# C — every densified slice
python -m rsna2024_lumbar.landmarks train --condition spinal_canal_stenosis --all-slices --dense-csv rsna2024_lumbar/data/processed/dense_coordinates/train_label_coordinates_dense.csv

# D — every image at test / holdout
python -m rsna2024_lumbar.landmarks infer --checkpoint path/to.pt --output-csv pred.csv --per-slice
```

Each run writes condition-level and per-lumbar-level metrics (R² / Rx / Ry, MAE, MAE-x / MAE-y, PCK@5 / PCK@10, L1, visible counts) for train / val / test:

- `<condition>_<backbone>_metrics.csv` — one row per `(split, overall|L1/L2…L5/S1)`
- `<condition>_<backbone>_metrics.json` — same numbers nested under `splits.*.levels`
- NAS `trial.json` copies the best-checkpoint `splits` so later we can rank weak condition×level cells and decide whether a specialist net is worth it.

Do not commit dense CSVs, DICOMs, or checkpoints.

NAS families (same 2D heads, ImageNet init, **512×512 fixed**): `full` = 4 models × 6 lrs × 5 weight decays = **120** trials. `convnext` / `efficientnet` are 60 each. `local` is the 24-trial 1-GPU scout.

```bash
python -m rsna2024_lumbar.landmarks nas --family full --list
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 1 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 2 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 4 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 8 --spawn --wait --skip-completed
```
