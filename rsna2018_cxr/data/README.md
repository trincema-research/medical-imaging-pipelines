# ⬇ data

Generated locally. Not committed, except this file. The mask PNGs are the iteration-1 segment of the severity split. A later Zenodo deposit is the place for the finished set, after the lung shapes are refined.

| Path | What |
|---|---|
| `manifests/master.csv` | one row per study |
| `models/compact_unet.pt` | compact U-Net, iteration 1, trained at 256. Not the cohort segmenter |
| `segmented/masks/<class>/*.png` | VGG11 lung field at 1024×1024 |
| `segmented/left/`, `segmented/right/` | patient left / right |
| `segmented/inspection/<class>/` | 10 overlays per class. Green U-Net, orange classical disagreement |
| `segmented/inspection_shape/<class>/` | experimental shape-completion contours. Not applied to the cohort masks |
| `segmented/manifest.csv` | paths, Dice, `label_status` |
| `segmented/provenance.jsonl` | checkpoint, threshold, fusion, QC |
| `segmented/review_queue.csv` | severity-balanced cases to review |

Counts and model settings: `segmentation/docs/iteration-1.md`.
