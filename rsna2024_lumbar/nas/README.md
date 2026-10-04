# ⚙ NAS — RSNA 2024 lumbar

Six isolated families. This package expands the grids (CPU, no torch). Training / 8-GPU launch stays later.

**per condition** = one stenosis/narrowing task. **all 5** = `--all-conditions` (×5).

| family | config | per condition | all 5 | output base |
|---|---|---:|---:|---|
| `vit` | `configs/vit_nas_search_space_8x8.json` | 2304 | 11520 | `runs/lumbar_nas` |
| `maxvit` | `configs/maxvit_nas_search_space_8x8.json` | 1152 | 5760 | `runs/lumbar_nas_maxvit` |
| `convnext` | `configs/convnext_nas_search_space_pruned.json` | 144 | 720 | `runs/lumbar_nas_convnext` |
| `convnext3d` | `configs/convnext3d_nas_search_space_192.json` | 384 | 1920 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | `configs/efficientnet_nas_search_space_pruned.json` | 144 | 720 | `runs/lumbar_nas_efficientnet` |
| `efficientnet3d` | `configs/efficientnet3d_nas_search_space_192.json` | 384 | 1920 | `runs/lumbar_nas_efficientnet3d` |

ViT / MaxViT use the 8×8 AdamW + SGD grids. ConvNeXt / EfficientNet 2D share the pruned 144-cell HP axes. ConvNeXt3D / EfficientNet3D share the 192-HP × `cnn3d_s`/`cnn3d_b` grid. The old 24-cell JSON files stay as legacy stubs. Image size comes from `--crop-policy` (`centered` 64×64, `extend50` 96×64). `runs/` is gitignored.

```bash
python -m rsna2024_lumbar.nas --family vit --list
python -m rsna2024_lumbar.nas --family maxvit --list
python -m rsna2024_lumbar.nas --family convnext --list
python -m rsna2024_lumbar.nas --family convnext3d --list
python -m rsna2024_lumbar.nas --family efficientnet --list
python -m rsna2024_lumbar.nas --family efficientnet3d --list --crop-policy extend50
```

## Parallel GPUs (1 / 2 / 4 / 8)

`--num-gpus 0` picks the largest of those that fits visible CUDA devices.

```bash
python -m rsna2024_lumbar.nas.launch --family convnext --num-gpus 4 --all-conditions
python -m rsna2024_lumbar.nas.launch --family efficientnet --num-gpus 8 --spawn --wait
python -m rsna2024_lumbar.nas.launch --family efficientnet3d --num-gpus 0 --all-conditions --spawn --wait
```

144 trials / 8 GPUs → 18 each. 384 / 8 → 48 each. `--spawn` starts one process per shard (`CUDA_VISIBLE_DEVICES`). The worker records the shard; the training loop is not in this package yet.

## Cloud zip

```bash
python -m rsna2024_lumbar.nas.pack --profile efficientnet_families
python -m rsna2024_lumbar.nas.pack --profile convnext_families
python -m rsna2024_lumbar.nas.pack --profile all --include-crops
```

Default zip is code + label CSVs (no PNG cache). `--include-crops` adds `data/processed/<policy>/`.
