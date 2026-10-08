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
| `condition_specific_runs.csv` | Long table: architecture × representation × condition × seed × split. |
| `condition_specific_by_condition.csv` | Five-run mean ± std per condition (do not mix incomplete groups). |
| `condition_specific_pipeline_summary.csv` | Condition-macro mean, best/worst, range, std. |
| `condition_specific_paper_table.csv` | Paper-ready mean ± std for val/test headline metrics. |
| `condition_specific_radar_*.csv` | Wide radar tables (SCS, LFN, RFN, LSS, RSS). |
| `condition_specific_summary.json` | Run counts, incomplete groups, OA==accuracy, ranking notes. |
| `spinal_level_runs.csv` | One row per run × split × spinal level (L1/L2 ... L5/S1). |
| `spinal_level_means.csv` | Five-run mean ± std per level. |
| `spinal_level_condition_summary.csv` | Macro-level mean, best/worst level, range, std per condition. |
| `spinal_level_pipeline_summary.csv` | Condition-macro spinal profile per architecture × representation. |
| `spinal_level_paper_table.csv` | Paper-ready val/test spinal tables. |
| `spinal_level_radar_*.csv` / `spinal_level_heatmap_*.csv` | Plot-ready wide tables. |
| `spinal_level_summary.json` | Run counts, incomplete groups, class-support warnings. |
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
python -m rsna2024_lumbar.best_config_runs condition-specific --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs spinal-level --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs validate-results --output-base rsna2024_lumbar/data/best_config_runs/results
```

Condition-specific and spinal-level tables are post-hoc on `pipeline_results` (same pattern as article-summary). Spinal-level analysis stratifies the same 200 runs by L1/L2 ... L5/S1; macro-level mean is the equal-weight mean of the five levels (not the pooled overall). Train with `--condition-specific-seeds` (42–46) for the Beyond-Accuracy protocol; existing harvested repeats keep their recorded `split_seed`. Incomplete 5-run groups are flagged and are not treated as complete.

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

There is no separate condition-specific or spinal-level trainer. Pack the usual best_config_runs zip, unpack on the VM, then train with **`--condition-specific-seeds`** (split seeds 42–46). After `run-all`, `condition_specific_*.csv` and `spinal_level_*.csv` are written automatically.

### 1) Pack (local machine)

Needs labels, NAS best configs, PNG crops, and the training bundle (`train_vit_lumbar.py`). Copy **`lumbar_best_config_runs.py`** next to the zip when you upload.

```bash
python -m rsna2024_lumbar.best_config_runs.pack_cloud \
  --output lumbar_best_config_runs_cloud.zip \
  --repeats 5 --epochs 50 --early-stop-patience 5 --num-gpus 8 \
  --skip-bundle
```

`--skip-bundle` uses the bundle already in this repo. Drop it and pass `--legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification` to restage `train_vit_lumbar.py`. `--no-include-crops` omits the PNG cache (you must supply crops on the VM).

Same pack via entry point: `rsna2024-best-config-runs-pack`.

### 2) Unpack (cloud VM)

Put **`lumbar_best_config_runs_cloud.zip`** and **`lumbar_best_config_runs.py`** in the same folder (for example `/workspace`):

```bash
python lumbar_best_config_runs.py unzip
# extracts to ./perf_cloud
```

If `./perf_cloud` already exists: use `run` (next step), or `unzip --force`, or `--dest ./other_dir`.

Module unpack (same extract):

```bash
python -m rsna2024_lumbar.best_config_runs.unpack_cloud \
  --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud
```

### 3) Execute on 8 GPUs

If you already unzipped:

```bash
python lumbar_best_config_runs.py run --repeats 5 --num-gpus 8 --condition-specific-seeds
```

`--skip-completed` is on by default (resume). One-shot unzip+train only when the dest is empty:

```bash
python lumbar_best_config_runs.py all --repeats 5 --num-gpus 8 --condition-specific-seeds
```

From the extracted repo root (`perf_cloud`, the folder with `pyproject.toml`):

```bash
python -m rsna2024_lumbar.best_config_runs.deploy_cloud \
  --repo-root . \
  --repeats 5 --epochs 50 --early-stop-patience 5 \
  --num-gpus 8 --condition-specific-seeds --skip-completed
```

Or unpack + train in one module call:

```bash
python -m rsna2024_lumbar.best_config_runs.deploy_cloud \
  --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud \
  --repeats 5 --epochs 50 --early-stop-patience 5 \
  --num-gpus 8 --condition-specific-seeds --skip-completed
```

`--early-stop-patience 0` runs all 50 epochs. Entry points: `rsna2024-best-config-runs-unpack`, `rsna2024-best-config-runs-deploy`. See `BEST_CONFIG_RUNS_CLOUD_RUN.txt` inside the zip.

### 4) Rebuild summary tables (no GPU)

After training, from the extracted repo root:

```bash
python -m rsna2024_lumbar.best_config_runs.harvest_results \
  --results-root rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs condition-specific \
  --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs spinal-level \
  --output-base rsna2024_lumbar/data/best_config_runs/results
```

`harvest_results` also writes the condition-specific and spinal-level CSVs when `pipeline_results` is present.

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
