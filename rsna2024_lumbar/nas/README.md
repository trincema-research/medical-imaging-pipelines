# NAS — RSNA 2024 lumbar

Six isolated families. Grids expand via ``python -m rsna2024_lumbar.nas``. GPU training uses the bundled legacy ``vit_nas_lumbar.py`` (copied into ``training_bundle/`` at pack time).

**per condition** = one stenosis/narrowing task. **all 5** = ``--all-conditions`` (runs sequentially when ``--spawn --wait``).

| family | config | per condition | all 5 | output base |
|---|---|---:|---:|---|
| `vit` | `configs/vit_nas_search_space_8x8.json` | 2304 | 11520 | `runs/lumbar_nas` |
| `maxvit` | `configs/maxvit_nas_search_space_8x8.json` | 1152 | 5760 | `runs/lumbar_nas_maxvit` |
| `convnext` | `configs/convnext_nas_search_space_pruned.json` | 144 | 720 | `runs/lumbar_nas_convnext` |
| `convnext3d` | `configs/convnext3d_nas_search_space_192.json` | 384 | 1920 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | `configs/efficientnet_nas_search_space_pruned.json` | 144 | 720 | `runs/lumbar_nas_efficientnet` |
| `efficientnet3d` | `configs/efficientnet3d_nas_search_space_192.json` | 384 | 1920 | `runs/lumbar_nas_efficientnet3d` |

## Pack for cloud (code + crops + training scripts)

On a machine with label CSVs, exported PNG crops, and the legacy lumbar repo:

```bash
# Default profile bundles EfficientNet 2D + centered crops + vit_nas_lumbar.py
python -m rsna2024_lumbar.nas.pack \
  --legacy-root F:/datasets_lumbar/rsna-2024-lumbar-spine-degenerative-classification

# Or set LUMBAR_LEGACY_ROOT and omit --legacy-root
```

Crops must live at ``rsna2024_lumbar/data/processed/centered/`` (same layout as ``export_crops``). Labels at ``rsna2024_lumbar/data/raw/*.csv``.

Profiles: ``efficientnet_deploy`` (default), ``convnext_deploy``, ``efficientnet_families``, ``all``, etc.

## One command on cloud

After unzip at the repo root (``pyproject.toml``):

```bash
python -m rsna2024_lumbar.nas.deploy \
  --family efficientnet --num-gpus 8 --all-conditions \
  --crop-policy centered --epochs 50 --early-stop-patience 5
```

``deploy`` runs ``pip install -e .``, installs ``requirements-nas-cloud.txt`` (torch), checks data + bundle, then ``nas.launch --spawn --wait``. With ``--all-conditions``, each condition uses all GPUs before the next starts.

Preflight only: ``python -m rsna2024_lumbar.nas.deploy --family efficientnet --dry-run``

## Parallel GPUs (1 / 2 / 4 / 8)

```bash
python -m rsna2024_lumbar.nas.launch --family efficientnet --num-gpus 8 \
  --condition spinal_canal_stenosis --spawn --wait --amp --progress \
  --epochs 50 --early-stop-patience 5
```

144 trials / 8 GPUs → 18 per GPU. Outputs: ``runs/lumbar_nas_efficientnet/<condition>/trial_*/result.json``.

## Best config from finished NAS runs

Scan legacy archives under ``runs/NAS results/{ViT,MaxViT,Convnext,...}`` or any tree of ``result.json`` files.

```bash
# ConvNeXt 2D — best trial per condition (default: val_acc at best epoch)
python -m rsna2024_lumbar.nas.best --family convnext --layout 2d

# Same run ranked by macro F1 or peak val accuracy
python -m rsna2024_lumbar.nas.best --family convnext --layout 2d --metric val_f1_macro_levels
python -m rsna2024_lumbar.nas.best --family convnext --layout 2d --metric max_val_acc

# ViT 2D + map trials to grid index + one shared HP set across all five conditions
python -m rsna2024_lumbar.nas.best --family vit --layout 2d --attach-grid-index --shared

# Explicit root (portable)
python -m rsna2024_lumbar.nas.best --family maxvit --results-root "/path/to/NAS results/MaxViT/2d"

# Export JSON summary
python -m rsna2024_lumbar.nas.best --family convnext --layout 2d --json-out runs/best_convnext_2d.json

# CSV (auto: runs/nas_best_<family>_<layout>.csv with full config + metrics)
python -m rsna2024_lumbar.nas.best --family vit --layout 2d --csv-out runs/nas_best_vit_2d.csv
```

Set ``LUMBAR_NAS_RESULTS_ROOT`` to the ``NAS results`` folder to omit ``--archive-root``. Metrics: ``val_acc`` (default), ``max_val_acc``, ``final_val_acc``, ``val_f1_macro_levels``, ``test_f1_macro_levels``, ``val_loss``. Module layout: ``results.py`` (load), ``rank.py`` (sort/shared pick), ``best.py`` (CLI).
