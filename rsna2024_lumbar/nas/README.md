# ⚙ NAS — RSNA 2024 lumbar

Six isolated families. This package expands the grids (CPU, no torch). Training / 8-GPU launch stays later. Counts are **per condition**.

| family | config | trials | output base |
|---|---|---:|---|
| `vit` | `configs/vit_nas_search_space_8x8.json` | 2304 | `runs/lumbar_nas` |
| `maxvit` | `configs/maxvit_nas_search_space_8x8.json` | 1152 | `runs/lumbar_nas_maxvit` |
| `convnext` | `configs/convnext_nas_search_space_pruned.json` | 144 | `runs/lumbar_nas_convnext` |
| `convnext3d` | `configs/convnext3d_nas_search_space_192.json` | 384 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | `configs/efficientnet_nas_search_space_pruned.json` | 144 | `runs/lumbar_nas_efficientnet` |
| `efficientnet3d` | `configs/efficientnet3d_nas_search_space_192.json` | 384 | `runs/lumbar_nas_efficientnet3d` |

ViT / MaxViT use the 8×8 AdamW + SGD grids. ConvNeXt / EfficientNet 2D share the pruned 144-cell HP axes. ConvNeXt3D / EfficientNet3D share the 192-HP × `cnn3d_s`/`cnn3d_b` grid. The old 24-cell JSON files stay as legacy stubs. Image size comes from `--crop-policy` (`centered` 64×64, `extend50` 96×64). `runs/` is gitignored.

```bash
python -m rsna2024_lumbar.nas --family vit --list
python -m rsna2024_lumbar.nas --family maxvit --list
python -m rsna2024_lumbar.nas --family convnext --list
python -m rsna2024_lumbar.nas --family convnext3d --list
python -m rsna2024_lumbar.nas --family efficientnet --list
python -m rsna2024_lumbar.nas --family efficientnet3d --list --crop-policy extend50
```
