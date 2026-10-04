# ▸ RSNA 2024 lumbar

```
⚙  preprocessing/     crop DICOMs → PNG cache
…  models/            later
…  training/          later
⚙  nas/               NAS grids · GPU launch · cloud pack/deploy (EfficientNet/ConvNeXt/…)
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
