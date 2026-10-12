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
| `patient_level_runs.csv` | One row per architecture × representation × repeat × split × patient_id. |
| `patient_level_repeat_summary.csv` | Equal-weight patient macros and error-profile rates per repeat. |
| `patient_level_summary.csv` | Mean ± std across 5 repeats (paper source). |
| `patient_level_paper_table.csv` | Paper-ready val/test patient table. |
| `patient_error_profile_*.csv` / `patient_*_distribution_*.csv` | Plot-ready stacked-bar and boxplot tables. |
| `patient_level_summary.json` | Incomplete groups, leakage warnings, PLOA not implemented. |
| `*/pipeline_run_summary.json` | Repeat count, config path, harvest metadata. |

Under each **`nas_best_*/<condition>/repeat_XX/`**, git tracks **`training_history.json`** and **`run_config.json`** only (audit / reproducibility). Other repeat artifacts (`training_metrics.csv`, `predictions_*.csv`, confusion matrices, checkpoints) stay **local/gitignored**.

### Protocol trees vs the already-integrated grid

The top-level `results/` tree is the earlier 5-repeat grid (split seeds **42 / 142 / 242 / 342 / 442**). Condition-specific and spinal-level tables there were already merged (PRs #15 and #16). The downloaded `results_ordinal_severity` dump matched those histories, so it was not re-imported.

Three later retrains used the Beyond-Accuracy protocol seeds **42–46**. They are distinct GPU runs (no shared history hashes with the top-level tree) and live next to it:

| Folder | Source dump | Tables in git |
|--------|-------------|----------------|
| `protocol_condition_specific/` | `results_condition-specific` | history + `run_config` + condition/spinal summaries |
| `protocol_spinal_level/` | `results_spinal_level` | same |
| `protocol_patient_level/` | `results_patient_level` | those plus `patient_level_*.csv` (needs local `predictions_*.csv` to rebuild) |

Each protocol tree has **198 / 200** finished repeats. The same two runs are missing in all three: `nas_best_convnext3d_3d/right_neural_foraminal_narrowing/repeat_03` and `nas_best_efficientnet3d_3d/right_subarticular_stenosis/repeat_05`. Harvest `--results-root` against the protocol folder so `run_dir` resolves; `harvest_results` also reads `run_config.json` when `pipeline_run_manifest.json` was not copied.

### Metric columns

- **Always (when retrain finished):** `train/val/test` `acc`, `accuracy_overall`, `f1_macro_overall`, `best_epoch`.
- **Severity (OA, O-MAE, QWK, SER) + precision/recall:** logged **each epoch** when training is launched via `best_config_runs` (`--log-pipeline-metrics` on `train_vit_lumbar.py`). **`harvest_results`** copies overall **and per-level** (L1/L2 ... L5/S1) columns into `pipeline_results.csv`; optional `--legacy-severity-backfill` only fills missing **OA** from accuracy and **val** O-MAE/QWK/SER from val confusion matrices (old cloud bundles without the hook).
- **Spinal-level tables** (`spinal_level_*.csv`) are post-hoc on those per-level columns (same 200 runs as condition-specific). Harvest / `run-all` write them automatically.
- **Patient-level tables** (`patient_level_*.csv`) join `predictions_{train,val,test}.csv` (written at the best val epoch) by `patient_id` across the five conditions. Patient-macro exact-rate / O-MAE weight each patient equally; do not substitute pooled accuracy. Error profiles are `exact` / `adjacent_only` / `severe`. PLOA is not computed.
- **Kaggle metric** (`kaggle_metric_scores.csv`): official sample-weighted log loss on joined five-condition softmax (formula: [docs/kaggle_metric.md](../../docs/kaggle_metric.md)). Not the NAS cross-entropy. Local scores use GT-centered crops and the 70/15/15 split.

Rebuild summaries from a downloaded results tree (no full retrain):

```bash
python -m rsna2024_lumbar.best_config_runs.harvest_results --results-root /path/to/results
```

Article tables (after `pipeline_results.csv` exist):

```bash
python -m rsna2024_lumbar.best_config_runs article-summary --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs condition-specific --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs spinal-level --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs patient-level --output-base rsna2024_lumbar/data/best_config_runs/results
python -m rsna2024_lumbar.best_config_runs validate-results --output-base rsna2024_lumbar/data/best_config_runs/results
```

Condition-specific and spinal-level tables are post-hoc on `pipeline_results` (same pattern as article-summary). Spinal-level analysis stratifies the same 200 runs by L1/L2 ... L5/S1; macro-level mean is the equal-weight mean of the five levels (not the pooled overall). Patient-level analysis joins per-prediction dumps across conditions for the same patient. Train with `--condition-specific-seeds` (42–46) for the Beyond-Accuracy protocol; existing harvested repeats keep their recorded `split_seed`. Incomplete 5-run groups are flagged and are not treated as complete.

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

There is no separate condition-specific, spinal-level, or patient-level trainer. Pack the usual best_config_runs zip, unpack on the VM, then train with **`--condition-specific-seeds`** (split seeds 42–46). After `run-all`, `condition_specific_*.csv`, `spinal_level_*.csv`, and `patient_level_*.csv` are written automatically when `predictions_*.csv` exist in the repeat dirs.

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
python -m rsna2024_lumbar.best_config_runs patient-level \
  --output-base rsna2024_lumbar/data/best_config_runs/results
```

`harvest_results` also writes the condition-specific, spinal-level, and patient-level CSVs when `pipeline_results` (and for patient-level, `predictions_*.csv`) is present.

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
