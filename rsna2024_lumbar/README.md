# ▸ RSNA 2024 lumbar

```
⚙  preprocessing/     crop DICOMs → PNG cache
…  models/            later
…  training/          later
⚙  nas/               ViT · MaxViT · ConvNeXt · ConvNeXt3D · EfficientNet grids (dry-run)
⬇  data/              download.py · unzip.py · validate.py · raw/ · processed/
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

## ⚙ NAS (dry-run)

Five isolated families. Expands the grids only — no GPU / no training yet. Counts are **per condition**.

```bash
python -m rsna2024_lumbar.nas --family vit --list
python -m rsna2024_lumbar.nas --family maxvit --list
python -m rsna2024_lumbar.nas --family convnext --list
python -m rsna2024_lumbar.nas --family convnext3d --list
python -m rsna2024_lumbar.nas --family efficientnet --list
```

| family | trials | output |
|---|---:|---|
| `vit` | 2304 | `runs/lumbar_nas` |
| `maxvit` | 1152 | `runs/lumbar_nas_maxvit` |
| `convnext` | 144 | `runs/lumbar_nas_convnext` |
| `convnext3d` | 384 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | 24 | `runs/lumbar_nas_efficientnet` |

`--crop-policy centered|extend50` sets PNG size. Depth: [nas/README.md](nas/README.md).

## ✔ Tests

```bash
pytest rsna2024_lumbar/tests
pytest rsna2024_lumbar/tests/test_export.py -q
```

CI reads **committed** `tests/fixtures/` (same CSV/DICOM layout, no patient data). We assert crop geometry, incomplete studies drop, PNG/manifest write, `--skip-existing` resume, job count vs `--max-studies`, NAS trial counts (2304 / 1152 / 144 / 384 / 24), isolated NAS output bases, and reject bad `--crop-policy` / `--condition` / `--max-studies` / `--family`.

| | input | result |
|---|---|---|
| ✔ | 1001 complete SCS | 5 PNGs exported |
| ✕ | 1002 missing last DICOM | study dropped |
| ✔ | 1003 complete, L4/L5 Moderate | 5 PNGs, Moderate folder used |

Depth: [tests/README.md](tests/README.md).
