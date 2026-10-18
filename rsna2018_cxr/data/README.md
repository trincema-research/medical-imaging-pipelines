# ⬇ data

The severity-split masks under `segmented/` are generated locally and are not committed. The VGG11 training run that produced them is committed under `iteration-1/vgg11/`.

| Path | Git | What |
|---|---|---|
| `iteration-1/vgg11/unet_vgg11_1024_best.pth` | yes | weights from epoch 62 |
| `iteration-1/vgg11/metrics_history.csv` | yes | train, val, and test metrics for 100 epochs |
| `iteration-1/vgg11/splits.csv` | yes | 99 / 11 / 28 ids used for that run |
| `iteration-1/vgg11/masks/` | yes | 668 labeled lung masks the local set started with |
| `models/compact_unet.pt` | no | compact U-Net side run. Not the cohort segmenter |
| `segmented/` | no | 14,863 severity masks and inspection overlays |

Source chest films stay at `F:\CXR_segmentation\dataset\images`.
