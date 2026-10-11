# ▸ RSNA 2024 lumbar

```
⚙  preprocessing/     crop DICOMs → PNG cache
⚙  landmarks/         densify (x, y) on every slice · 2D ConvNeXt / EfficientNetV2
…  models/            later
…  training/          later
⚙  nas/               NAS grids · GPU launch · cloud pack/deploy (8 families)
✔  best_config_runs/     retrain NAS best configs · severity metrics · cloud zip
✔  visualizations/    publication figures from harvested CSVs
⬇  data/              raw/ · processed/ · nas_compact/ · best_config_runs/results/
☰  traces/            download.log · unzip.log · validate.log · pipeline.log
✔  tests/             pytest + synthetic mini-dump
```

From repo root (`pip install -e ".[dev]"` first). Download + `.env`: [data/README.md](data/README.md).

## ⬇ Download (Kaggle → data/raw/)

```bash
copy .env.example .env    # set KAGGLE_API_TOKEN (gitignored)
pip install -e ".[kaggle]"
python -m rsna2024_lumbar.data.download    # zip + unzip; non-zero if it failed
python -m rsna2024_lumbar.data.unzip       # only if a .zip is still in data/raw/
python -m rsna2024_lumbar.data.validate
```

## ⚙ Preprocess

```bash
python -m rsna2024_lumbar.preprocessing --help

# smoke (1 study from your dump)
python -m rsna2024_lumbar.preprocessing \
  --data-root rsna2024_lumbar/data/raw \
  --crop-policy centered --max-studies 1 --progress

# full dump → cache (resume-safe)
python -m rsna2024_lumbar.preprocessing \
  --data-root rsna2024_lumbar/data/raw \
  --output-root rsna2024_lumbar/data/processed/centered \
  --crop-policy centered --skip-existing --progress
```

`--data-root` is any Kaggle-shaped folder (`train.csv` + `train_images/`). Default output is `data/processed/<policy>/`. Policies: `centered` (64×64), `extend50` (96×64). Incomplete studies (missing a level/DICOM) are skipped.

## ⚙ Landmark coordinates (all slices)

Kaggle marks one `(x, y)` per `(study, condition, level)`. `densify` projects that LPS point onto every other slice in the series (axial “copy up and down”). Optional 2D regression matches the RSNA 2024 1st-place split: ConvNeXt-Base / EfficientNetV2-L, level-separated `(x/W, y/H)` heads, L1, 70/15/15 by `study_id`.

```bash
pip install -e ".[landmarks,dev]"
python -m rsna2024_lumbar.landmarks densify --max-studies 2 --progress
python -m rsna2024_lumbar.landmarks train --condition spinal_canal_stenosis --backbone convnext_tiny --image-size 64 --epochs 2 --max-studies 4

# cloud: zip scripts (+ optional train_images/), unzip on the box, 1/2/4/8-GPU NAS
python -m rsna2024_lumbar.landmarks pack --include-images
python -m rsna2024_lumbar.landmarks nas --num-gpus 8 --all-conditions --spawn --wait
```

Dense CSVs stay in `data/processed/dense_coordinates/` (gitignored). Do not commit the cloud zip. Depth: [landmarks/README.md](landmarks/README.md) · [docs/landmark_coordinates.md](docs/landmark_coordinates.md).

## ⚙ NAS

Six isolated families. Grid expansion, 1/2/4/8 GPU shards, and cloud deploy. **per condition** = one task; **all 5** = `--all-conditions` (sequential when `--spawn --wait`).

**Pack + run on cloud (EfficientNet 2D, all five conditions):**

```bash
# On a machine with crops under data/processed/centered/ and the legacy lumbar training repo:
python -m rsna2024_lumbar.nas.pack --profile efficientnet_deploy --legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification

# On the GPU instance (unzip, then from repo root):
python -m rsna2024_lumbar.nas.deploy --family efficientnet --num-gpus 8 --all-conditions \
  --crop-policy centered --epochs 50 --early-stop-patience 5
```

**Best trial from archived NAS results** (legacy ``runs/NAS results/`` on disk):

```bash
python -m rsna2024_lumbar.nas.best --family convnext --layout 2d --shared --attach-grid-index

# Batch CSV for ViT / MaxViT / ConvNeXt (2D + 3D) from ``runs/NAS results/``
python -m rsna2024_lumbar.nas.summarize
```

Dry-run grid:

```bash
python -m rsna2024_lumbar.nas --family vit --list
python -m rsna2024_lumbar.nas --family maxvit --list
python -m rsna2024_lumbar.nas --family convnext --list
python -m rsna2024_lumbar.nas --family convnext3d --list
python -m rsna2024_lumbar.nas --family efficientnet --list
python -m rsna2024_lumbar.nas --family efficientnet3d --list
```

| family | per condition | all 5 | output |
|---|---:|---:|---|
| `vit` | 2304 | 11520 | `runs/lumbar_nas` |
| `maxvit` | 1152 | 5760 | `runs/lumbar_nas_maxvit` |
| `convnext` | 144 | 720 | `runs/lumbar_nas_convnext` |
| `convnext3d` | 384 | 1920 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | 144 | 720 | `runs/lumbar_nas_efficientnet` |
| `efficientnet3d` | 384 | 1920 | `runs/lumbar_nas_efficientnet3d` |

`--crop-policy centered|extend50` sets PNG size. Shard across **1 / 2 / 4 / 8** GPUs (`--num-gpus 0` = auto). Training uses bundled `vit_nas_lumbar.py` (staged by `nas.pack`). Depth: [nas/README.md](nas/README.md).

## ✔ Best-config runs (after NAS)

Confirm NAS winners: fixed retrain protocol, multiple split-seed repeats, CSV metrics for papers/PRs.

```bash
pip install -e ".[best_config_runs,dev]"

# All eight nas_best_* layouts (default 5 repeats)
python -m rsna2024_lumbar.best_config_runs run-all --repeats 5 --epochs 50 \
  --data-root rsna2024_lumbar/data/raw \
  --crops-root rsna2024_lumbar/data/processed/centered

# NAS OA only (no GPU): nas-snapshot-all → data/best_config_runs/results/
python -m rsna2024_lumbar.best_config_runs nas-snapshot-all
```

Cloud (single GPU): pack with `best_config_runs.pack_cloud`, or copy **`lumbar_best_config_runs.py`** next to the zip and run `python lumbar_best_config_runs.py all --repeats 5`.

Depth: [best_config_runs/README.md](best_config_runs/README.md) · results: [data/best_config_runs/README.md](data/best_config_runs/README.md).

## ✔ Visualizations

Publication figures from harvested CSVs (not from manuscript numbers). Same level as `data/`.

```bash
pip install -e ".[visualizations,dev]"
python -m rsna2024_lumbar.visualizations patient-level
python -m rsna2024_lumbar.visualizations spinal-level
python -m rsna2024_lumbar.visualizations ordinal
python -m rsna2024_lumbar.visualizations condition-specific
```

Writes PNG/PDF under [visualizations/figures/](visualizations/figures/). Depth and figure links: [visualizations/README.md](visualizations/README.md).

## ✔ Tests

```bash
pytest rsna2024_lumbar/tests
pytest rsna2024_lumbar/tests/test_export.py -q
```

CI reads **committed** `tests/fixtures/` (same CSV/DICOM layout, no patient data). We assert crop geometry, incomplete studies drop, PNG/manifest write, `--skip-existing` resume, job count vs `--max-studies`, NAS trial counts per condition (2304 / 1152 / 144 / 384 / 144 / 384; ×5 if `--all-conditions`), isolated NAS output bases, and reject bad `--crop-policy` / `--condition` / `--max-studies` / `--family`.

| | input | result |
|---|---|---|
| ✔ | 1001 complete SCS | 5 PNGs exported |
| ✕ | 1002 missing last DICOM | study dropped |
| ✔ | 1003 complete, L4/L5 Moderate | 5 PNGs, Moderate folder used |

Depth: [tests/README.md](tests/README.md).
