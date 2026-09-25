# [24] RSNA 2024 lumbar

```
preprocessing/     crop DICOMs → PNG cache
models/            (later)
training/          (later)
nas/               (later)
data/raw/          local Kaggle dump (gitignored)
data/processed/    crop PNG cache (gitignored)
tests/             pytest + synthetic mini-dump
```

From repo root (`pip install -e ".[dev]"` first).

## Preprocess

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

## Tests

```bash
pytest rsna2024_lumbar/tests
pytest rsna2024_lumbar/tests/test_export.py -q
```

CI does **not** use `data/raw/`. Each run builds a 3-study synthetic dump (same CSV/DICOM layout, no patient data). We assert crop geometry, incomplete studies drop, PNG/manifest write, `--skip-existing` resume, and that job count scales with `--max-studies`.

| input | result we check |
|---|---|
| 1001 complete SCS | 5 PNGs exported |
| 1002 missing last DICOM | study dropped |
| 1003 complete, L4/L5 Moderate | 5 PNGs, Moderate folder used |

Details: [tests/README.md](tests/README.md).
