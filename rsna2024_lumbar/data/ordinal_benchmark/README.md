# Ordinal benchmark runs

Refit training from NAS best configs logs **ordinal severity metrics** alongside standard accuracy:

| Metric | Key suffix | Meaning |
|--------|------------|---------|
| OA | `*_oa_overall` | Exact grade match (0/1/2) |
| O-MAE | `*_omae_overall` | Mean \|true − pred\| |
| QWK | `*_qwk_overall` | Quadratic weighted kappa |
| SER | `*_ser_overall` | Fraction with grade gap ≥ 2 |

## Run locally (GPU)

From the repo root, with raw data and processed PNG crops on disk:

```bash
pip install -e ".[ordinal,dev]"

python -m rsna2024_lumbar.ordinal refit \
  --best-config rsna2024_lumbar/data/nas_compact/best_configs/nas_best_vit_2d.json \
  --condition spinal_canal_stenosis \
  --epochs 50 \
  --data-root /path/to/rsna2024/data/raw \
  --crops-root /path/to/rsna2024/data/processed/centered
```

Outputs land under `rsna2024_lumbar/data/ordinal_benchmark/<family>_<layout>/<condition>/`:

- `training_metrics.csv` — full epoch history (includes ordinal columns when `RSNA2024_ORDINAL_METRICS=1`, set automatically by the refit CLI)
- `ordinal_epoch_metrics.csv` — ordinal columns only
- `ordinal_refit_summary_<config>.csv` — one row per condition at best val accuracy epoch
- `refit_manifest.json` — NAS trial id and hyperparameters used

Set `RSNA2024_DATA_ROOT` / `RSNA2024_CROPS_ROOT` instead of CLI paths when convenient.
