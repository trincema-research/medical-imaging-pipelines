# ⬇ severity_models/

Committed: `val_test_images.csv`, `dataset_split.json`, `training_history.csv`, `manifest.json`.

Not committed: `*.pt` (346 MB inference files, or the original 1 GB dumps). Download:

```bash
python -m rsna2018_cxr.severity download-weights
```

Release tag: `rsna2018-severity-s1-specialists`.
