# Performance pipeline (`perf_pipeline`)

Stage **3** in the lumbar stack: **`data/`** → **`nas/`** (search + `nas_compact/best_configs`) → **`perf_pipeline/`** (retrain NAS winners and publish CSV metrics).

| Doc | Contents |
|-----|----------|
| [../data/perf_pipeline/README.md](../data/perf_pipeline/README.md) | Results layout, cloud zip, committed CSVs, article tables |
| This file | CLI map and module layout |

## Install

```bash
pip install -e ".[perf_pipeline,dev]"
```

Console entry points: `rsna2024-perf-pipeline` (alias `rsna2024-ordinal`), `rsna2024-perf-pipeline-pack`, `-unpack`, `-deploy`.

## CLI (`python -m rsna2024_lumbar.perf_pipeline`)

| Command | Purpose |
|---------|---------|
| `run` | One `nas_best_*.json`, all conditions, default 5 repeats |
| `run-all` | All eight per-model best configs |
| `status` | Count finished repeats (resume / second GPU) |
| `list-configs` | Print paths under `best_configs/` |
| `nas-snapshot` / `nas-snapshot-all` | OA from NAS compact histories (no GPU) |

Common flags: `--repeats`, `--epochs`, `--early-stop-patience`, `--skip-completed`, `--only` (shard models), `--data-root`, `--crops-root`.

## Module layout

| Module | Role |
|--------|------|
| `severity_metrics.py` | OA, O-MAE, QWK, SER |
| `multi_head.py` | Five-head logits → severity columns |
| `config.py` | Load `nas_best_*.json` |
| `train.py` / `train_entry.py` | Subprocess `train_vit_lumbar.py` with `RSNA2024_PERF_PIPELINE_METRICS=1` |
| `runner.py` | Repeats × conditions orchestration |
| `results.py` | `pipeline_results.csv` schema |
| `nas_snapshot.py` | NAS-history OA only |
| `harvest_results.py` | Rebuild summary CSVs from downloaded `repeat_*/training_metrics.csv` |
| `pack_cloud.py` / `unpack_cloud.py` / `deploy_cloud.py` | Cloud zip (advanced) |
| `cloud_common.py` | Shared pack manifest helpers |

## Cloud (simple)

Copy **`lumbar_perf_cloud.py`** (repo root) next to `lumbar_perf_pipeline_cloud.zip`:

```bash
python lumbar_perf_cloud.py unzip
python lumbar_perf_cloud.py run --repeats 5
# or: python lumbar_perf_cloud.py all --repeats 5
```

Pack the zip locally: `python -m rsna2024_lumbar.perf_pipeline.pack_cloud` (see data README).

## Metrics note

Severity columns (`*_oa_overall`, `*_omae_overall`, `*_qwk_overall`, `*_ser_overall`) are filled when training runs with the perf_pipeline metrics hook. Cloud bundles without an updated `train_vit_lumbar.py` may still produce acc/F1-only rows; use `harvest_results` after fixing the bundle and re-running.
