# medical-imaging-pipelines

Shared training / preprocessing code for RSNA challenges. Each year is isolated; reusable pieces live in `common/`.

```
[lib]  common/              preprocessing · models · metrics · utils
[18]   rsna2018_cxr/        chest X-ray pneumonia
[24]   rsna2024_lumbar/     lumbar stenosis  ← preprocessing first
[25]   rsna2025_brain/      aneurysm detection
[26]   rsna2026_knee/       meniscus / cartilage
[doc]  docs/
[bench] benchmarks/
```

## RSNA 2024 — start here

Tiny CI fixtures in-repo. Full Kaggle dump stays **local** in `rsna2024_lumbar/data/raw/` (gitignored). Crops write to `data/processed/`.

```bash
pip install -e ".[dev]"
pytest
python -m rsna2024_lumbar.preprocessing --help
```

Full export (after you copy `train.csv` + `train_images/` into `data/raw/`):

```bash
python -m rsna2024_lumbar.preprocessing \
  --data-root rsna2024_lumbar/data/raw \
  --output-root rsna2024_lumbar/data/processed/centered \
  --crop-policy centered --progress
```

## Data rules

| Path | Git | What |
|---|---|---|
| `*/tests/fixtures/` | yes | synthetic mini-dataset (CI) |
| `*/data/raw/` | no | official challenge download |
| `*/data/processed/` | no | cropped cache the trainer will load |

Remote tests never need the big dump. Point `--data-root` at `data/raw` locally (or any existing checkout).
