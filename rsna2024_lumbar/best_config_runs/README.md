# Best-config runs (`best_config_runs`)

Stage **3** in the lumbar stack: **`data/`** → **`nas/`** (search + `nas_compact/best_configs`) → **`best_config_runs/`** (retrain NAS winners and publish CSV metrics).

| Doc | Contents |
|-----|----------|
| [../data/best_config_runs/README.md](../data/best_config_runs/README.md) | Results layout, cloud zip, committed CSVs, article tables |
| This file | CLI map and module layout |

## Install

```bash
pip install -e ".[best_config_runs,dev]"
```

Console entry points: `rsna2024-best-config-runs` (aliases: `rsna2024-perf-pipeline`, `rsna2024-ordinal`), `-pack`, `-unpack`, `-deploy`.

After harvest / before merging result CSVs:

```bash
python -m rsna2024_lumbar.best_config_runs validate-results --output-base rsna2024_lumbar/data/best_config_runs/results
```

## CLI (`python -m rsna2024_lumbar.best_config_runs`)

| Command | Purpose |
|---------|---------|
| `run` | One `nas_best_*.json`, all conditions, default 5 repeats |
| `run-all` | All eight per-model best configs |
| `status` | Count finished repeats (resume / second GPU) |
| `list-configs` | Print paths under `best_configs/` |
| `nas-snapshot` / `nas-snapshot-all` | OA from NAS compact histories (no GPU) |
| `article-summary` | Mean ± std article CSV tables |
| `condition-specific` | Beyond-Accuracy condition-macro / best / worst / range / std |
| `validate-results` | Check harvested CSVs and summary tables |

Common flags: `--repeats`, `--epochs`, `--early-stop-patience`, `--skip-completed`, `--only`, `--num-gpus` (1=sequential, 0=auto 2/4/8, or 2/4/8 parallel workers), `--save-checkpoints` (default off), `--data-root`, `--crops-root`. `--condition-specific-seeds` uses split seeds 42–46.

## Module layout

| Module | Role |
|--------|------|
| `severity_metrics.py` | OA, O-MAE, QWK, SER |
| `multi_head.py` | Five-head logits → severity columns |
| `config.py` | Load `nas_best_*.json` |
| `train.py` / `train_entry.py` | Subprocess `train_vit_lumbar.py` with `RSNA2024_BEST_CONFIG_RUNS_METRICS=1` |
| `runner.py` | Repeats × conditions orchestration |
| `results.py` | `pipeline_results.csv` schema |
| `article_summary.py` | Per-condition mean ± std article tables |
| `condition_specific.py` | Condition-macro / best / worst / range / std |
| `nas_snapshot.py` | NAS-history OA only |
| `harvest_results.py` | Rebuild summary CSVs from downloaded `repeat_*/training_metrics.csv` |
| `pack_cloud.py` / `unpack_cloud.py` / `deploy_cloud.py` | Cloud zip (advanced) |
| `cloud_common.py` | Shared pack manifest helpers |

## Cloud (pack / unpack / 8 GPUs)

Copy **`lumbar_best_config_runs.py`** next to `lumbar_best_config_runs_cloud.zip`. Full notes: [../data/best_config_runs/README.md](../data/best_config_runs/README.md#cloud-8-gpus).

Pack:

```bash
python -m rsna2024_lumbar.best_config_runs.pack_cloud \
  --output lumbar_best_config_runs_cloud.zip \
  --repeats 5 --epochs 50 --early-stop-patience 5 --num-gpus 8 \
  --skip-bundle
```

Unpack, then train on 8 GPUs (do not run `all` after a successful `unzip`):

```bash
python lumbar_best_config_runs.py unzip
python lumbar_best_config_runs.py run --repeats 5 --num-gpus 8 --condition-specific-seeds
```

One step (empty dest only):

```bash
python lumbar_best_config_runs.py all --repeats 5 --num-gpus 8 --condition-specific-seeds
```

Module unpack / deploy:

```bash
python -m rsna2024_lumbar.best_config_runs.unpack_cloud \
  --zip lumbar_best_config_runs_cloud.zip --dest ./perf_cloud
python -m rsna2024_lumbar.best_config_runs.deploy_cloud \
  --repo-root ./perf_cloud --repeats 5 --num-gpus 8 \
  --condition-specific-seeds --skip-completed
```

## Metrics note

Severity columns (`*_oa_overall`, `*_omae_overall`, `*_qwk_overall`, `*_ser_overall`) are filled when training runs with the best_config_runs metrics hook. Cloud bundles without an updated `train_vit_lumbar.py` may still produce acc/F1-only rows; use `harvest_results` after fixing the bundle and re-running.
