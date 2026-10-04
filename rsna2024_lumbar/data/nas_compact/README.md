# NAS compact export (git-friendly)

Legacy full NAS runs live under ``runs/NAS results/`` on disk (large: checkpoints, logs, every trial).

This folder holds a **compact mirror** for ViT, MaxViT, and ConvNeXt (2D + 3D):

```
nas_compact/
  ViT/2d/<condition>/trial_XXXX/training_history.csv
  ViT/2d/<condition>/trial_XXXX/trial_config.json
  best_configs/nas_best_*.csv    # ranked best trials + full metrics/hyperparams (default: val_acc)
  best_configs/nas_best_*.json   # same rows, nested hyperparameters + metrics
  manifest.json
  ViT/ ...                       # --mode all only (local; large)
```

Per-epoch metrics are in **``training_history.csv``** (no hyperparameter columns).  
**``trial_config.json``** copies the NAS hyperparameter fields from ``result.json`` plus val acc/loss at the best epoch when available.

## Build (from repo root)

```bash
# Recommended for git: 5 best trials per condition × 6 model/layout targets (~30 folders)
python -m rsna2024_lumbar.nas.export_compact --mode best

# Size check before a full export (~800 MB of CSV histories)
python -m rsna2024_lumbar.nas.export_compact --estimate-only

# Full history mirror (not for git unless you use Git LFS)
python -m rsna2024_lumbar.nas.export_compact --mode all --yes-all
```

Set ``LUMBAR_NAS_RESULTS_ROOT`` to your ``NAS results`` directory, or pass ``--archive-root``.

## Committed in this repo

| Path | Contents |
|------|----------|
| ``ViT/``, ``MaxViT/``, ``Convnext/`` (2d + 3d) | Full export: ``training_history.csv`` + ``trial_config.json`` per trial (~804 MB) |
| ``best_configs/*.csv`` | Best trial per condition (``val_acc`` default), 30 rows combined |
| ``manifest.json`` | Export metadata |

Regenerate: ``python -m rsna2024_lumbar.nas.export_compact --mode all --yes-all`` then ``python -m rsna2024_lumbar.nas.summarize --csv-dir rsna2024_lumbar/data/nas_compact/best_configs``.
