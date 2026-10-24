# ▸ RSNA 2018 CXR — severity specialists

Binary ViT specialists for the RSNA 2018 3-level burden split (`normal` vs S1 / S2 / S3). The study code lived in `F:\severity_train` (NAS + retrain) on top of the ViT trainer from `F:\datasets_pneumonia`.

```
⚙  severity/                 train · NAS · retrain · weight export/download
⬇  data/severity_models/     3 S1 specialist `.pt` files (Git LFS) + split CSV
✔  tests/
```

The three checkpoints are **S1 repeats** (normal vs `severity_1_low_mild`), not S2/S3. Inference-only copies (~346 MB, no optimizer) live in `data/severity_models/` via Git LFS. The original ~1 GB training dumps stay local.

```bash
pip install -e ".[severity,dev]"
python -m rsna2018_cxr.severity download-weights
python -m rsna2018_cxr.severity nas --help
python -m rsna2018_cxr.severity retrain --help
python -m rsna2018_cxr.severity train --help
pytest rsna2018_cxr/tests
```

Retrain one specialist from the published config + val/test CSV:

```bash
python -m rsna2018_cxr.severity retrain \
  --config-json rsna2018_cxr/severity/configs/retrain_s1.json \
  --split-csv rsna2018_cxr/data/severity_models/val_test_images.csv \
  --data-root F:/severity_train \
  --severity severity_1 \
  --output-dir rsna2018_cxr/data/processed/severity_retrain_s1
```

`--severity severity_2` / `severity_3` use the same trainer. Those weights were not in the study `output_retrain/` folder.
