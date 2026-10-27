# Coordinate prediction (x, y)

Complete `train_label_coordinates.csv` for **every slice** in a labeled series, then (optionally) learn the same points with the RSNA 2024 **1st-place** 2D regressors.

NAS and `train` read the **raw Kaggle DICOMs** (`train_images/` + the three label CSVs). They do **not** use the PNG crop cache. Download the official zip, unpack it into `data/raw/`, then run NAS. This package used to live at `landmarks/`; `pip install -e ".[landmarks]"` and the `rsna2024-landmarks*` scripts still work as aliases.

1st place ([Avengers write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)) trained **2D** ConvNeXt-Base and EfficientNetV2-L on a 512×512 RGB slice (gray repeated 3×) with **level-separated heads** and **L1** on `(x/width, y/height)`. Same recipe here. Five condition pipelines, not 25 level nets.

## From Kaggle zip to NAS (raw dump)

From the repo root (`medical-imaging-pipelines/`). Accept the [competition rules](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules) first.

```bash
# 1. token (gitignored). Kaggle → Settings → API → Generate New Token
copy .env.example .env

# 2. install download + training extras
pip install -e ".[kaggle,coordinate_prediction]"

# 3. official zip → rsna2024_lumbar/data/raw/  (unzips; ~30 GB train_images/)
python -m rsna2024_lumbar.data.download
python -m rsna2024_lumbar.data.unzip       # only if a .zip is still in data/raw/
python -m rsna2024_lumbar.data.validate

# 4. print the grid (no GPU). 4 models × 6 lrs × 5 wds = 120 per condition
python -m rsna2024_lumbar.coordinate_prediction nas --family full --list

# 5. all 5 conditions × all 120 cells. Only --num-gpus and --batch-size change.
python -m rsna2024_lumbar.coordinate_prediction nas \
  --family full \
  --all-conditions \
  --data-root rsna2024_lumbar/data/raw \
  --num-gpus 8 \
  --batch-size 8 \
  --spawn --wait
```

`--data-root` must look like the Kaggle extract:

```
< data-root >/
  train.csv
  train_label_coordinates.csv
  train_series_descriptions.csv
  train_images/<study_id>/<series_id>/<instance_number>.dcm
```

Default `--data-root` is `rsna2024_lumbar/data/raw` (what `data.download` fills). If the dump lives somewhere else, point at that folder:

```bash
python -m rsna2024_lumbar.coordinate_prediction nas \
  --family full --all-conditions \
  --data-root /path/to/rsna-2024-lumbar-spine-degenerative-classification \
  --num-gpus 8 --batch-size 8 \
  --spawn --wait
```

`--num-gpus` is **1, 2, 4, or 8** (`0` = auto). `--batch-size` is the same for every trial (start at 8; use 4 if ConvNeXt-Base / EfficientNetV2-L OOM). Everything else is fixed: **512×512**, ImageNet init, 20 epochs, AdamW, lr `{1e-5, 3e-5, 1e-4, 3e-4, 6e-4, 1e-3}`, weight decay `{0.001, 0.003, 0.01, 0.03, 0.05}`.

`--all-conditions` runs the five conditions **one after another**, each using the full GPU pool (`full` = 120 trials → 15 per GPU on 8 cards). Resume is **on by default**: re-run the same command after a crash or kill. After every epoch the worker appends one row to `training_metrics.csv` and `training_history.csv` (same pattern as severity `train_vit_lumbar` / `best_config_runs`). A trial is complete when that CSV reaches the planned epoch count, or when a legacy `*_history.json` exists. Those are skipped. An epoch that dies mid-step cannot be recovered. Check the grid on disk (no GPU):

```bash
python -m rsna2024_lumbar.coordinate_prediction nas --family full --status
python -m rsna2024_lumbar.coordinate_prediction nas --family full --all-conditions --status
```

`--no-resume` starts those trials over. `--skip-completed` is kept as an alias. Dry-run the shard plan (no training): omit `--spawn`. `--list` prints the planned slugs without looking at outputs.

To start the **full** grid at a given trial (1-based, inclusive), pass `--start-trial`. Remaining cells are resharded across `--num-gpus` (trial 47 of 120 on 8 GPUs → 47–120, not the original 1–15 shards). `--end-trial` is optional. With `--all-conditions` the same window applies to **each** condition independently; if only one condition was interrupted, drop `--all-conditions` and pass `--condition`:

```bash
python -m rsna2024_lumbar.coordinate_prediction nas \
  --family full --condition spinal_canal_stenosis \
  --start-trial 47 \
  --data-root rsna2024_lumbar/data/raw \
  --num-gpus 8 --batch-size 8 \
  --spawn --wait
```

Outputs (gitignored): `rsna2024_lumbar/data/processed/dense_coordinates/nas/full/<condition>/<slug>/`

- `<condition>_<backbone>_metrics.csv` / `.json` — train/val/test, overall **and each lumbar level**
- `trial.json` — hyperparameters + those splits
- `<condition>_<backbone>.pt` — best val checkpoint
- `training_metrics.csv` / `training_history.csv` — one row per finished epoch (written immediately)
- `<condition>_<backbone>_history.json` — full JSON dump after the last epoch
- `<condition>_<backbone>_last.pt` — last finished epoch weights (used if the trial is retried)

Do not commit the zip, DICOMs, or NAS histories. Download details: [data/README.md](../data/README.md).

## Single-run train / densify / infer

Kaggle only marks one `(study, condition, level)`. Axial neighbours can reuse that in-plane point. `densify` projects the labeled LPS point with `ImagePositionPatient` / `ImageOrientationPatient` / `PixelSpacing`. NAS does **not** need densify (it uses the median labeled slice). Use densify for `--all-slices` training.

```bash
pip install -e ".[coordinate_prediction,dev]"

python -m rsna2024_lumbar.coordinate_prediction densify --data-root rsna2024_lumbar/data/raw --progress

python -m rsna2024_lumbar.coordinate_prediction train \
  --data-root rsna2024_lumbar/data/raw \
  --condition spinal_canal_stenosis --backbone convnext_tiny \
  --image-size 64 --epochs 2 --max-studies 4

python -m rsna2024_lumbar.coordinate_prediction infer \
  --data-root rsna2024_lumbar/data/raw \
  --checkpoint rsna2024_lumbar/data/processed/dense_coordinates/runs/spinal_canal_stenosis_convnext_tiny.pt \
  --output-csv rsna2024_lumbar/data/processed/dense_coordinates/pred_scs.csv

python -m rsna2024_lumbar.coordinate_prediction train \
  --data-root rsna2024_lumbar/data/raw \
  --condition spinal_canal_stenosis --all-slices \
  --dense-csv rsna2024_lumbar/data/processed/dense_coordinates/train_label_coordinates_dense.csv

python -m rsna2024_lumbar.coordinate_prediction infer \
  --data-root rsna2024_lumbar/data/raw \
  --checkpoint rsna2024_lumbar/data/processed/dense_coordinates/runs/spinal_canal_stenosis_convnext_tiny.pt \
  --output-csv rsna2024_lumbar/data/processed/dense_coordinates/pred_scs.csv \
  --per-slice
```

Resize to 512×512 as a **tensor** `(B, 3, 512, 512)` with five level heads. Do not flatten pixels into a wide CSV. Design: [docs/landmark_all_slice_pipeline.md](../docs/landmark_all_slice_pipeline.md).

Dense CSVs stay under `data/processed/dense_coordinates/` (gitignored).

Backbones: `convnext_tiny` (smoke), `convnext_base` and `efficientnet_v2_l` (1st place), `efficientnet_v2_s`.

## Cloud pack (code + optional dump)

If the GPU box should download Kaggle itself, copy the repo, run the download commands above, then NAS. If you already have `data/raw/` and want one zip:

```bash
python -m rsna2024_lumbar.coordinate_prediction pack --output lumbar_coordinate_prediction_cloud.zip
python -m rsna2024_lumbar.coordinate_prediction pack --include-images --output lumbar_coordinate_prediction_cloud.zip

python unzip_lumbar_coordinate_prediction_cloud.py --zip lumbar_coordinate_prediction_cloud.zip --dest ./coordinate_prediction_cloud
cd coordinate_prediction_cloud
pip install -e ".[kaggle,coordinate_prediction]"
python -m rsna2024_lumbar.coordinate_prediction nas \
  --family full --all-conditions \
  --data-root rsna2024_lumbar/data/raw \
  --num-gpus 8 --batch-size 8 \
  --spawn --wait
```

`deploy` is install + the same NAS launch:

```bash
python -m rsna2024_lumbar.coordinate_prediction deploy \
  --family full --all-conditions \
  --data-root rsna2024_lumbar/data/raw \
  --num-gpus 8 --batch-size 8
```

Families: `full` (120), `convnext` / `efficientnet` (60), `local` (24). Depth: [docs/landmark_coordinates.md](../docs/landmark_coordinates.md) · [docs/landmark_all_slice_pipeline.md](../docs/landmark_all_slice_pipeline.md).
