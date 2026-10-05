# Performance pipeline results

## What is this stage?

| | |
|--|--|
| **Package** | `rsna2024_lumbar/perf_pipeline/` |
| **Name** | **Performance pipeline** (`perf_pipeline`) |
| **Comes after** | `data/` (inputs) → `nas/` (search + `nas_compact/best_configs`) |
| **Main purpose** | **Confirm** NAS winners: retrain with a fixed protocol, default **5 repeats** per condition (split seeds), and write **train / val / test** metrics—including **OA, O-MAE, QWK, SER**—to CSV under `data/perf_pipeline/results/`. |

NAS tells you what worked in search; perf_pipeline produces **auditable, comparable** numbers for reporting and PRs.

## Committed results in git

- `nas_snapshots_all_models.csv` — OA at NAS best val epoch (all model layouts, no GPU).
- Per-model `nas_snapshot_<config>.csv` — same, one file per `nas_best_*.json`.
- After local `run` / `run-all`: `pipeline_results.csv` (full severity + accuracy on retrain).

Regenerate NAS snapshots:

```bash
python -m rsna2024_lumbar.perf_pipeline nas-snapshot-all
```

## Run locally

```bash
pip install -e ".[perf_pipeline,dev]"

# One model layout: 5 conditions × 5 split seeds (default)
python -m rsna2024_lumbar.perf_pipeline run \
  --best-config rsna2024_lumbar/data/nas_compact/best_configs/nas_best_efficientnet_2d.json \
  --epochs 50 \
  --data-root /path/to/data/raw \
  --crops-root /path/to/data/processed/centered

# All per-model best configs (3 repeats for PR)
python -m rsna2024_lumbar.perf_pipeline run-all --repeats 3 --epochs 50 --data-root ... --crops-root ...

# Progress (expect 8 models × 5 conditions × 3 repeats = 120)
python -m rsna2024_lumbar.perf_pipeline status --repeats 3 --epochs 50

# Resume / second GPU: skip finished repeats; shard with --only
python -m rsna2024_lumbar.perf_pipeline run-all --repeats 3 --epochs 50 --skip-completed \
  --only nas_best_vit_2d nas_best_maxvit_2d --data-root ... --crops-root ...
```

**Training terminals:** keep at most **two** GPU training jobs at once (one `run-all` or `run` per GPU). Do not restart legacy `ordinal refit` — use `perf_pipeline` only.

## Cloud (single GPU)

Pack on a machine with labels, best configs, and PNG crops:

```bash
python -m rsna2024_lumbar.perf_pipeline.pack_cloud \
  --output lumbar_perf_pipeline_cloud.zip \
  --repeats 5 --epochs 50 --early-stop-patience 5 \
  --legacy-root /path/to/rsna-2024-lumbar-spine-degenerative-classification
```

On the cloud VM (unpack only):

```bash
python -m rsna2024_lumbar.perf_pipeline.unpack_cloud \
  --zip lumbar_perf_pipeline_cloud.zip --dest ./perf_cloud
```

Install + train (defaults: 5 repeats, patience 5, all 8 models):

```bash
cd perf_cloud   # repo root with pyproject.toml
python -m rsna2024_lumbar.perf_pipeline.deploy_cloud --repo-root . --skip-completed
```

One-shot unpack + deploy:

```bash
python -m rsna2024_lumbar.perf_pipeline.deploy_cloud \
  --zip lumbar_perf_pipeline_cloud.zip --dest ./perf_cloud --repeats 5
```

Entry points: `rsna2024-perf-pipeline-pack`, `-unpack`, `-deploy`. See `PERF_PIPELINE_CLOUD_RUN.txt` inside the zip.

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
