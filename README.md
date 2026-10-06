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
python -m rsna2024_lumbar.nas --family vit --list  # ⚙ NAS dry-run
python -m rsna2024_lumbar.nas.launch --family efficientnet --num-gpus 8 --spawn --wait  # GPU shards
python -m rsna2024_lumbar.nas.pack --profile efficientnet_deploy --legacy-root /path/to/lumbar  # deploy zip
python -m rsna2024_lumbar.nas.deploy --family efficientnet --num-gpus 8 --all-conditions  # cloud one-shot
python -m rsna2024_lumbar.best_config_runs run-all --repeats 5 --epochs 50  # post-NAS retrain + CSV metrics
```

**Cloud NAS:** pack includes code, label CSVs, optional `data/processed/centered/` PNGs, and a copied `vit_nas_lumbar.py` bundle. Unzip on the GPU box and run `nas.deploy` (see [rsna2024_lumbar/nas/README.md](rsna2024_lumbar/nas/README.md)).

**Best-config runs (after NAS):** retrain `nas_compact/best_configs/` winners; results under `data/best_config_runs/results/`. Cloud helper: [lumbar_best_config_runs.py](lumbar_best_config_runs.py) · [rsna2024_lumbar/best_config_runs/README.md](rsna2024_lumbar/best_config_runs/README.md).

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
| ⬆ | `*/data/processed/` | no | full PNG cache (local or inside deploy zip) |

Remote tests never need the big dump. Point `--data-root` at `data/raw` locally (or any existing checkout).
