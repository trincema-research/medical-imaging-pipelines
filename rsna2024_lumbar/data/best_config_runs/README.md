# Best-config runs — results

## What is this stage?

| | |
|--|--|
| **Package** | `rsna2024_lumbar/best_config_runs/` — [module README](../../best_config_runs/README.md) |
| **Name** | **best_config_runs** |
| **Comes after** | `data/` (inputs) → `nas/` (search + `nas_compact/best_configs`) |
| **Main purpose** | **Confirm** NAS winners: retrain with a fixed protocol, default **5 repeats** per condition (split seeds), and write **train / val / test** metrics to CSV under this folder. |

NAS tells you what worked in search; best_config_runs produces **auditable, comparable** numbers for reporting and papers.

> **Legacy path:** `data/perf_pipeline/` → see [../perf_pipeline/README.md](../perf_pipeline/README.md). Use this folder (`data/best_config_runs/`) for all new results.

## Committed CSVs in git (article-friendly)

| File | Description |
|------|-------------|
| `nas_snapshots_all_models.csv` | **OA only** at NAS best val epoch (all 8 layouts; from compact histories, no GPU). |
| `nas_best_*/nas_snapshot_*.csv` | Same, one file per layout. |
| `nas_best_*/pipeline_results.csv` | One row per **condition × repeat** at best val epoch (retrain). |
| `pipeline_results_all_models.csv` | Combined retrain rows across layouts. |
| `article_summary_by_condition.csv` | Per model × condition: **mean ± std** over repeats at **best val epoch** (`*_pm` columns). |
| `article_summary_metrics_long.csv` | Same aggregates in long form (`metric`, `mean`, `std`, `mean_pm_std`). |
| `*/pipeline_run_summary.json` | Repeat count, config path, harvest metadata. |

Under each **`nas_best_*/<condition>/repeat_XX/`**, git tracks **`training_history.json`** and **`run_config.json`** only (audit / reproducibility). Other repeat artifacts (`training_metrics.csv`, confusion matrices, checkpoints) stay **local/gitignored**.

### Metric columns

- **Always (when retrain finished):** `train/val/test` `acc`, `accuracy_overall`, `f1_macro_overall`, `best_epoch`.
- **Severity (OA, O-MAE, QWK, SER) + precision/recall:** logged **each epoch** when training is launched via `best_config_runs` (`--log-pipeline-metrics` on `train_vit_lumbar.py`). **`harvest_results`** reads those columns as-is; optional `--legacy-severity-backfill` only fills missing **OA** from accuracy and **val** O-MAE/QWK/SER from val confusion matrices (old cloud bundles without the hook).

Rebuild summaries from a downloaded results tree (no full retrain):

```bash
python -m rsna2024_lumbar.best_config_runs.harvest_results --results-root /path/to/results
```

Article tables (after `pipeline_results.csv` exist):

```bash
python -m rsna2024_lumbar.best_config_runs article-summary --output-base rsna2024_lumbar/data/best_config_runs/results
```

Regenerate NAS OA snapshots:

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

Use **`--num-gpus 4`** (or `0` for auto) on `run-all` to shard pending jobs across GPUs; optional **`--only`** limits models. Checkpoints are **off** unless **`--save-checkpoints`**.

## Cloud (8 GPUs)

### Simple (recommended)

On an 8-GPU VM, place **`lumbar_best_config_runs_cloud.zip`** and repo-root **`lumbar_best_config_runs.py`** in the same directory:

```bash
python lumbar_best_config_runs.py unzip
python lumbar_best_config_runs.py run --repeats 5 --num-gpus 8 --skip-completed
# one step: python lumbar_best_config_runs.py all --repeats 5 --num-gpus 8
```

Use `--early-stop-patience 0` for full 50 epochs without NAS-style early stop.

### Advanced (pack / module deploy)

Pack on a machine with labels, best configs, and PNG crops:

```bash
python -m rsna2024_lumbar.best_config_runs.pack_cloud \
  --output lumbar_best_config_runs_cloud.zip \
  --repeats 5 --epochs 50 --early-stop-patience 5 --num-gpus 8 \
  --skip-bundle \
  --legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification
```

Unpack and train:

```bash
python -m rsna2024_lumbar.best_config_runs.deploy_cloud \
  --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud --repeats 5 --skip-completed
```

Entry points: `rsna2024-best-config-runs-pack`, `-unpack`, `-deploy` (legacy: `rsna2024-perf-pipeline-*`). See `BEST_CONFIG_RUNS_CLOUD_RUN.txt` inside the zip.

## Output layout (local disk)

```
data/best_config_runs/results/<nas_best_* stem>/
  pipeline_results.csv
  pipeline_run_summary.json
  nas_snapshot_<stem>.csv          # optional NAS OA
  <condition>/repeat_01/           # gitignored
    training_metrics.csv
    best_epoch_results.csv
    ...
```

Full CLI map: [../../best_config_runs/README.md](../../best_config_runs/README.md).
