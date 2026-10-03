# ⚙ NAS — RSNA 2024 lumbar

Five isolated families. This package expands the grids (CPU, no torch). Training / 8-GPU launch stays later. Counts are **per condition**.

| family | config | trials | output base |
|---|---|---:|---|
| `vit` | `configs/vit_nas_search_space_8x8.json` | 2304 | `runs/lumbar_nas` |
| `maxvit` | `configs/maxvit_nas_search_space_8x8.json` | 1152 | `runs/lumbar_nas_maxvit` |
| `convnext` | `configs/convnext_nas_search_space_pruned.json` | 144 | `runs/lumbar_nas_convnext` |
| `convnext3d` | `configs/convnext3d_nas_search_space_192.json` | 384 | `runs/lumbar_nas_convnext3d` |
| `efficientnet` | `configs/efficientnet_nas_search_space.json` | 24 | `runs/lumbar_nas_efficientnet` |

ViT / MaxViT use the 8×8 AdamW + SGD grids. ConvNeXt 2D is the pruned 144-cell grid that `lumbar_nas_convnext` actually ran (not the leftover 24-cell file). ConvNeXt3D is 192 hyperparams × `cnn3d_s`/`cnn3d_b`. `configs/convnext_nas_search_space.json` is the legacy 24-cell stub. Image size comes from `--crop-policy` (`centered` 64×64, `extend50` 96×64). `runs/` is gitignored.

```bash
python -m rsna2024_lumbar.nas --family vit --list
python -m rsna2024_lumbar.nas --family maxvit --list
python -m rsna2024_lumbar.nas --family convnext --list
python -m rsna2024_lumbar.nas --family convnext3d --list
python -m rsna2024_lumbar.nas --family efficientnet --list --crop-policy extend50
```
