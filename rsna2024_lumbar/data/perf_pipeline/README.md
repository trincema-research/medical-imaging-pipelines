# Moved → `best_config_runs`

This path is **legacy**. Committed CSVs and docs live under:

- **[`data/best_config_runs/`](../best_config_runs/README.md)** — results, snapshots, article CSVs  
- **[`rsna2024_lumbar/best_config_runs/`](../../best_config_runs/README.md)** — Python package (CLI, cloud pack)

```bash
python -m rsna2024_lumbar.best_config_runs run-all --repeats 5 --epochs 50 \
  --data-root rsna2024_lumbar/data/raw \
  --crops-root rsna2024_lumbar/data/processed/centered
```

Local logs or old run trees under `data/perf_pipeline/` are gitignored; move or delete them locally if you still have this folder.
