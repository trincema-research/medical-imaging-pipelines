# ▸ RSNA 2018 CXR — lung segmentation

Severity-stratified lung masks for the RSNA 2018 cohort (Zenodo 10.5281/zenodo.20783246).

Normal 8851 · S1 1999 · S2 2006 · S3 2007. Every study is 1024×1024. A different size is rejected, not resized.

S1/S2/S3 are annotation-derived radiographic burden strata, not clinical severity. Silver masks are automatic consensus, not ground truth. Gold is reserved for human review. Iteration 1 never wrote gold: one model cannot be silver, so the saved masks are `pseudo` or `rejected`.

```
⚙  segmentation/     manifest · VGG11 segment · QC · compact U-Net train
☰  configs/          cohort paths and the thresholds the run used
✔  tests/
📄 segmentation/docs/iteration-1.md
⬇  data/             masks and checkpoints, local only
```

```bash
python -m rsna2018_cxr.segmentation build-manifest --severity-root F:/severity_train
python -m rsna2018_cxr.segmentation segment --checkpoint F:/CXR_segmentation/models/unet_vgg11_1024_best.pth
```

`segment` writes 1024×1024 masks to `data/segmented/`. Those PNGs stay local until a finished set is posted to Zenodo. The method, the VGG11 checkpoint metrics, the label counts, and the unused compact-U-Net training result are in [segmentation/docs/iteration-1.md](segmentation/docs/iteration-1.md).

The cohort segmenter is the external VGG11-BN U-Net. A classical Otsu mask is drawn only on 10 inspection films per class. Shape-completion overlays under `data/segmented/inspection_shape/` are a separate look and were not written back onto the cohort masks.
