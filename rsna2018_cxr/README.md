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
python -m rsna2018_cxr.segmentation segment
```

`segment` loads `data/iteration-1/vgg11/unet_vgg11_1024_best.pth` and writes 1024×1024 masks to `data/segmented/`. Those severity PNGs stay local until a finished set is posted to Zenodo. The weights, the 100-epoch log, the split, and the labeled masks are already in `data/iteration-1/vgg11/`. Method and counts: [segmentation/docs/iteration-1.md](segmentation/docs/iteration-1.md).

A classical Otsu mask is drawn only on 10 inspection films per class. Shape-completion overlays under `data/segmented/inspection_shape/` were not written back onto the cohort masks. Post-DAE, HybridGNet, and MedSAM were not part of this run.
