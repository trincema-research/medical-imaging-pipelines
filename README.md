# medical-imaging-pipelines

Shared training / preprocessing code for RSNA challenges. Each year is isolated; reusable pieces live in `common/`.

```
▣  common/              shared preprocessing · models · metrics · utils
▸  rsna2018_cxr/        chest X-ray pneumonia
▸  rsna2024_lumbar/     lumbar stenosis   ← preprocess first
▸  rsna2025_brain/      aneurysm detection
▸  rsna2026_knee/       meniscus / cartilage
☰  docs/
▤  benchmarks/
```

`▣` shared · `▸` challenge · `☰` docs · `▤` benches · `⚙` preprocess · `✔` tests · `⬇` raw · `⬆` crops

## ▸ RSNA 2024 — start here

Tiny CI dump in tests. Label CSVs are committed under `rsna2024_lumbar/data/raw/`; `train_images/` stays **local** (gitignored). Crops write to `data/processed/`.

```bash
pip install -e ".[dev,kaggle]"
# or: pip install -r requirements-dev.txt
# copy .env.example → .env and set KAGGLE_API_TOKEN (gitignored)
pytest                                          # ✔
python -m rsna2024_lumbar.data.download         # ⬇ zip + unzip
python -m rsna2024_lumbar.data.unzip            # ⬇ if a .zip is still in raw/
python -m rsna2024_lumbar.data.validate         # ✔
python -m rsna2024_lumbar.preprocessing --help  # ⚙
```

Full export (copy `train.csv` + `train_images/` into `data/raw/`):

```bash
python -m rsna2024_lumbar.preprocessing \
  --data-root rsna2024_lumbar/data/raw \
  --output-root rsna2024_lumbar/data/processed/centered \
  --crop-policy centered --progress
```

## Data rules

| | Path | Git | What |
|---|---|---|---|
| ✔ | `*/tests/` | yes | synthetic mini-dump (CI) |
| ⬇ | `*/data/raw/*.csv` | yes | Kaggle label tables |
| ⬇ | `*/data/raw/train_images/` (2 studies) | yes | smoke DICOMs only |
| ⬇ | rest of `train_images/` + `.zip` | no | full official dump |
| ⬆ | `*/data/processed/centered/` | yes | PNG crop cache (~240MB) |

Remote tests never need the big dump. Point `--data-root` at `data/raw` locally (or any existing checkout).
