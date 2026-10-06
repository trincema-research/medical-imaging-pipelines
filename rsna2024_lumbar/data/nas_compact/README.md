# NAS compact export (git-friendly)

Legacy full NAS runs live under ``runs/NAS results/`` on disk (large: checkpoints, logs, every trial).

This folder holds a **compact mirror** of per-epoch histories and slim trial configs for all **eight** search targets (ViT, MaxViT, ConvNeXt, EfficientNet × 2D/3D):

```
nas_compact/
  ViT/2d/<condition>/trial_XXXX/training_history.csv
  ViT/2d/<condition>/trial_XXXX/trial_config.json
  EfficientNet/2d/ ...                          # same layout
  best_configs/nas_best_*.csv                   # best trial per condition
  best_configs/nas_best_*.json                  # nested hyperparameters + metrics
  best_configs/nas_best_all.csv                 # combined (8 layouts × 5 conditions)
  manifest.json
```

Per-epoch metrics are in **``training_history.csv``** (no hyperparameter columns).  
**``trial_config.json``** copies NAS hyperparameter fields from ``result.json`` plus summary metrics at the best epoch when available.

## Build (from repo root)

```bash
# One family or estimate size first
python -m rsna2024_lumbar.nas.export_compact --archive-root "/path/to/NAS results" --models efficientnet --estimate-only

# Full history mirror for git (large; EfficientNet adds ~2k+ trials per layout)
python -m rsna2024_lumbar.nas.export_compact --mode all --yes-all

# Refresh best-config CSVs/JSON under best_configs/
python -m rsna2024_lumbar.nas.summarize --csv-dir rsna2024_lumbar/data/nas_compact/best_configs
```

Set ``LUMBAR_NAS_RESULTS_ROOT`` or pass ``--archive-root`` to your ``NAS results`` directory.

## Committed in this repo (typical)

| Path | Contents |
|------|----------|
| ``ViT/``, ``MaxViT/``, ``Convnext/``, ``EfficientNet/`` (2d + 3d) | ``training_history.csv`` + ``trial_config.json`` per trial |
| ``best_configs/nas_best_*.json`` | One file per model layout (eight stems) |
| ``best_configs/nas_best_all.csv`` | 40 rows (8 layouts × 5 conditions) |
| ``manifest.json`` | Export metadata |

**Next stage:** [perf_pipeline](../perf_pipeline/README.md) reads ``best_configs/`` and writes retrain metrics under ``data/perf_pipeline/results/``.
