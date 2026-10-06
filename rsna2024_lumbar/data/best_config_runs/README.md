# Best-config runs — results

## What is this stage?

| | |
|--|--|
| **Package** | `rsna2024_lumbar/best_config_runs/` — [module README](../../best_config_runs/README.md) |
| **Name** | **best_config_runs** (renamed from `perf_pipeline`; same layout and CSVs) |
| **Comes after** | `data/` (inputs) → `nas/` (search + `nas_compact/best_configs`) |
| **Main purpose** | **Confirm** NAS winners: retrain with a fixed protocol, default **5 repeats** per condition (split seeds), and write **train / val / test** metrics—including **OA, O-MAE, QWK, SER**—to CSV under this folder. |

NAS tells you what worked in search; best_config_runs produces **auditable, comparable** numbers for reporting and papers.

> **Legacy path:** `data/perf_pipeline/` → [../perf_pipeline/README.md](../perf_pipeline/README.md).

## Committed results in git

| File | Description |
|------|-------------|
| `nas_snapshots_all_models.csv` | OA at NAS best val epoch (all **8** layouts including **EfficientNet 2D/3D**; no GPU). |
| `nas_best_*/nas_snapshot_*.csv` | Same, one file per `nas_best_*.json`. |
| `nas_best_*/pipeline_results.csv` | One row per **condition × repeat** at best val epoch (retrain). |
| `pipeline_results_all_models.csv` | Combined retrain rows across layouts. |
| `*/pipeline_run_summary.json` | Repeat count, config path, harvest metadata. |

Local **`repeat_*`** run directories are **gitignored**; only the summary CSVs above are tracked.

Rebuild summaries from a downloaded results tree:

```bash
python -m rsna2024_lumbar.best_config_runs.harvest_results --results-root /path/to/results
```

Regenerate NAS snapshots:

```bash
python -m rsna2024_lumbar.best_config_runs nas-snapshot-all
```

## Run locally

```bash
pip install -e ".[best_config_runs,dev]"

python -m rsna2024_lumbar.best_config_runs run \
  --best-config rsna2024_lumbar/data/nas_compact/best_configs/nas_best_efficientnet_2d.json \
  --epochs 50 --early-stop-patience 5 \
  --data-root rsna2024_lumbar/data/raw \
  --crops-root rsna2024_lumbar/data/processed/centered

python -m rsna2024_lumbar.best_config_runs run-all --repeats 5 --epochs 50 --skip-completed \
  --data-root rsna2024_lumbar/data/raw \
  --crops-root rsna2024_lumbar/data/processed/centered

python -m rsna2024_lumbar.best_config_runs status --repeats 5 --epochs 50
```

Use **`--only nas_best_vit_2d …`** to shard across two GPUs; at most **two** training terminals at once.

## Cloud (single GPU)

On the VM, place **`lumbar_best_config_runs_cloud.zip`** and repo-root **`lumbar_best_config_runs.py`** in the same directory:

```bash
python lumbar_best_config_runs.py unzip
python lumbar_best_config_runs.py run --repeats 5 --skip-completed
```

Pack locally:

```bash
python -m rsna2024_lumbar.best_config_runs.pack_cloud \
  --output lumbar_best_config_runs_cloud.zip \
  --repeats 5 --epochs 50 --early-stop-patience 5 \
  --legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification
```

See [../../best_config_runs/README.md](../../best_config_runs/README.md) and `BEST_CONFIG_RUNS_CLOUD_RUN.txt` in the zip.

## Output layout

```
data/best_config_runs/results/<nas_best_* stem>/
  pipeline_results.csv
  pipeline_run_summary.json
  nas_snapshot_<stem>.csv
  <condition>/repeat_01/        # gitignored
    training_metrics.csv
    ...
```

`pipeline_results.csv` includes `train_*`, `val_*`, and `test_*` for acc, `accuracy_overall`, `f1_macro_overall`, and severity `oa/omae/qwk/ser` overall when the metrics hook is enabled.
