# Landmark (x, y) completion

Complete `train_label_coordinates.csv` for **every slice** in a labeled series, then (optionally) learn the same points with the RSNA 2024 **1st-place** 2D regressors.

Kaggle only marks one `(study, condition, level)`. Axial neighbours can reuse that in-plane point: the canal does not jump left–right as you walk superior–inferior. `densify` does that the accurate way — project the labeled LPS point with `ImagePositionPatient` / `ImageOrientationPatient` / `PixelSpacing`. If geometry is missing, it copies the pixel.

1st place ([Avengers write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)) then trained **2D** ConvNeXt-Base and EfficientNetV2-L on a 512×512 RGB slice (gray repeated 3×) with **level-separated heads** and **L1** on `(x/width, y/height)`. Same recipe here. Five condition pipelines, not 25 level nets.

```bash
pip install -e ".[landmarks,dev]"

# geometry fill (no GPU) — do this first
python -m rsna2024_lumbar.landmarks densify --max-studies 2 --progress

# 2D regression (one condition; 70/15/15 by study_id)
python -m rsna2024_lumbar.landmarks train \
  --condition spinal_canal_stenosis --backbone convnext_tiny \
  --image-size 64 --epochs 2 --max-studies 4

# predict + project onto the rest of each series
python -m rsna2024_lumbar.landmarks infer \
  --checkpoint rsna2024_lumbar/data/processed/dense_coordinates/runs/spinal_canal_stenosis_convnext_tiny.pt \
  --output-csv rsna2024_lumbar/data/processed/dense_coordinates/pred_scs.csv

# every densified slice (not a 262k-column pixel table)
python -m rsna2024_lumbar.landmarks train \
  --condition spinal_canal_stenosis --all-slices \
  --dense-csv rsna2024_lumbar/data/processed/dense_coordinates/train_label_coordinates_dense.csv

# run the net on every DICOM instead of median + project
python -m rsna2024_lumbar.landmarks infer \
  --checkpoint rsna2024_lumbar/data/processed/dense_coordinates/runs/spinal_canal_stenosis_convnext_tiny.pt \
  --output-csv rsna2024_lumbar/data/processed/dense_coordinates/pred_scs.csv \
  --per-slice
```

Resize to 512×512 as a **tensor** `(B, 3, 512, 512)` with five level heads. Do not flatten pixels into a wide CSV. Design: [docs/landmark_all_slice_pipeline.md](../docs/landmark_all_slice_pipeline.md).

Dense CSVs stay under `data/processed/dense_coordinates/` (gitignored). Do not commit them.

Backbones: `convnext_tiny` (smoke), `convnext_base` and `efficientnet_v2_l` (1st place), `efficientnet_v2_s`.

## Cloud pack / NAS

**512×512 is fixed** (not a search axis). `full` = 4 models × 6 lrs × 5 weight decays = **120** trials per condition. `convnext` / `efficientnet` are 60 each. `local` is the 24-trial 1-GPU scout. Batch size is not searched — default 8, override with `--batch-size`. Same 1/2/4/8 GPU shards as severity NAS (`k = ceil(N/G)`, `CUDA_VISIBLE_DEVICES`). `--all-conditions` is sequential-by-condition.

```bash
# zip code + labels (small). Add --include-images to pack train_images/ (~30 GB DICOMs).
python -m rsna2024_lumbar.landmarks pack --output lumbar_landmarks_cloud.zip
python -m rsna2024_lumbar.landmarks pack --include-images --output lumbar_landmarks_cloud.zip

# on the GPU box (stdlib unzip, no pip yet)
python unzip_lumbar_landmarks_cloud.py --zip lumbar_landmarks_cloud.zip --dest ./landmarks_cloud
cd landmarks_cloud
pip install -e ".[landmarks]"

# print every combination (no training)
python -m rsna2024_lumbar.landmarks nas --family full --list

# shard the same `full` grid (120 trials / condition) across 1, 2, 4, or 8 GPUs
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 1 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 2 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 4 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family full --num-gpus 8 --spawn --wait --skip-completed

# other families, or all five conditions (sequential: one condition, then the next)
python -m rsna2024_lumbar.landmarks nas --family local --num-gpus 1 --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family convnext --num-gpus 8 --all-conditions --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks nas --family efficientnet --num-gpus 8 --all-conditions --spawn --wait --skip-completed
python -m rsna2024_lumbar.landmarks deploy --family full --num-gpus 8 --all-conditions --skip-completed
```

`--num-gpus` must be 1, 2, 4, or 8 (`0` = auto). Shards are `k = ceil(N/G)`: `full` is 120 / 60 / 30 / 15 trials per GPU. Swap `--family full` for `convnext` (60), `efficientnet` (60), or `local` (24).

Every trial writes `<condition>_<backbone>_metrics.csv` / `.json` and copies train/val/test splits into `trial.json`. Rows are `overall` plus each lumbar level (R², Rx, Ry, MAE, MAE-x, MAE-y, PCK@5, PCK@10, L1, `n_visible`). Use those later to pick condition×level pairs that need a specialist model.

Do not commit the zip, DICOMs, or NAS histories. Depth: [docs/landmark_coordinates.md](../docs/landmark_coordinates.md) · [docs/landmark_all_slice_pipeline.md](../docs/landmark_all_slice_pipeline.md).
