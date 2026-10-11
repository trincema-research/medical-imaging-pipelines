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
```

Dense CSVs stay under `data/processed/dense_coordinates/` (gitignored). Do not commit them.

Backbones: `convnext_tiny` (smoke), `convnext_base` and `efficientnet_v2_l` (1st place), `efficientnet_v2_s`.

Depth: [docs/landmark_coordinates.md](../docs/landmark_coordinates.md).
