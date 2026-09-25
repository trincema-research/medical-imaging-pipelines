# ▸ RSNA 2024 lumbar

```
⚙  preprocessing/     crop DICOMs → PNG cache
…  models/            later
…  training/          later
…  nas/               later
⬇  data/raw/          local Kaggle dump (gitignored)
⬆  data/processed/    crop PNG cache (gitignored)
✔  tests/             pytest + synthetic mini-dump
```

From repo root (`pip install -e ".[dev]"` first).

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

## ✔ Tests

```bash
pytest rsna2024_lumbar/tests
pytest rsna2024_lumbar/tests/test_export.py -q
```

CI reads **committed** `tests/fixtures/` (same CSV/DICOM layout, no patient data). We assert crop geometry, incomplete studies drop, PNG/manifest write, `--skip-existing` resume, job count vs `--max-studies`, and reject bad `--crop-policy` / `--condition` / `--max-studies`.

| | input | result |
|---|---|---|
| ✔ | 1001 complete SCS | 5 PNGs exported |
| ✕ | 1002 missing last DICOM | study dropped |
| ✔ | 1003 complete, L4/L5 Moderate | 5 PNGs, Moderate folder used |

Depth: [tests/README.md](tests/README.md).
