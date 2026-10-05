# Performance pipeline results

Stage **after** `data/` and `nas/`: retrain **NAS best** hyperparameters and record
**train / val / test** metrics (accuracy, F1, and severity: OA, O-MAE, QWK, SER).

## Run locally

```bash
pip install -e ".[perf_pipeline,dev]"

# One model layout: 5 conditions × 5 split seeds (default)
python -m rsna2024_lumbar.perf_pipeline run \
  --best-config rsna2024_lumbar/data/nas_compact/best_configs/nas_best_efficientnet_2d.json \
  --epochs 50 \
  --data-root /path/to/data/raw \
  --crops-root /path/to/data/processed/centered

# All per-model best configs
python -m rsna2024_lumbar.perf_pipeline run-all --epochs 50 --data-root ... --crops-root ...
```

## Output layout

```
data/perf_pipeline/results/<nas_best_* stem>/
  pipeline_results.csv          # one row per condition × repeat (best val epoch)
  pipeline_run_summary.json
  <condition>/repeat_01/        # training_metrics.csv, best_epoch_results.csv, ...
  ...
```

`pipeline_results.csv` columns include `train_*`, `val_*`, and `test_*` for
`acc`, `accuracy_overall`, `f1_macro_overall`, and severity `oa/omae/qwk/ser` overall.

NAS-only OA (no retrain): `python -m rsna2024_lumbar.perf_pipeline nas-snapshot --best-config ...`
