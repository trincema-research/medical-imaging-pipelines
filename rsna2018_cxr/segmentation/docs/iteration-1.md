# Iteration 1 — severity-split lung masks

This is the run that segmented the Zenodo severity split (10.5281/zenodo.20783246) and wrote the first masks. The segmenter is the local VGG11-BN U-Net. Post-DAE, HybridGNet, MedSAM, and CheXmask were not used.

The weights, the epoch log, the train/val/test ids, and the labeled masks that training started from are in `data/iteration-1/vgg11/`. The 14,863 severity masks and the inspection overlays stay local under `data/segmented/` for a later Zenodo deposit.

S1, S2, and S3 are annotation-derived radiographic burden, not clinical severity. Silver would be automatic agreement between models, not ground truth. Gold is human review only. This run never wrote gold. One model cannot be silver, so every saved mask is `pseudo` or `rejected`.

## Cohort

| Class | Studies | Files |
|---|---:|---|
| Normal | 8851 | `normal/*.png` |
| S1 | 1999 | `pneumonia/severity_1_low_mild/*.jpg` |
| S2 | 2006 | `pneumonia/severity_2_moderate/*.jpg` |
| S3 | 2007 | `pneumonia/severity_3_severe/*.jpg` |
| Total | 14863 | |

Every file is grayscale 1024×1024. A study of any other size is rejected. Nothing in this cohort is resized.

Local root used for the run: `F:\severity_train`.

## Model that wrote the masks

VGG11-BN U-Net, class `UNetVGG11`, loaded by `ExistingUNetBackend`.

| | |
|---|---|
| Checkpoint | `data/iteration-1/vgg11/unet_vgg11_1024_best.pth` |
| Architecture | VGG11-BN encoder, bilinear upsampling, 1-channel first convolution, 1-class logits |
| Loss | BCE with logits. Best epoch is the lowest validation loss |
| Trained at | 1024×1024 |
| Best epoch | 62 of 100 |
| Train Dice / Jaccard | 0.9824 / 0.9655 |
| Val Dice / Jaccard | 0.9753 / 0.9522 |
| Test Dice / Jaccard at that epoch | 0.9738 / 0.9495 |
| Train loss / val loss | 0.0212 / 0.0300 |
| Optimizer | Adam, learning rate 0.0005, batch 4, mixed precision, plateau schedule |
| Epoch log | `data/iteration-1/vgg11/metrics_history.csv` |
| Split | 99 train / 11 val / 28 test, listed in `splits.csv` |
| Labeled masks | `data/iteration-1/vgg11/masks/` (668 PNGs). Three split ids have no mask file; the loader treated those as empty |
| Source films | `F:\CXR_segmentation\dataset\images`. Not committed; about 4.4 GB |

Inference on the severity cohort:

- Input is the file divided by 255. No ImageNet mean.
- Logits go through a sigmoid. The mask is probability ≥ 0.5.
- CUDA uses fp16 autocast. The weights stay in the checkpoint dtype.
- `postprocess_merged` drops tiny components. `split_lungs` separates the two lungs. With `radiological=True`, patient left is image right on a standard PA film.
- The mask is saved at 1024×1024, the same grid as the film.

Retrain with `python -m rsna2018_cxr.segmentation train-vgg`. The committed file is the weights and the scalar metrics from epoch 62. The original 186 MB file also stored the Adam state; that state is not in git because GitHub rejects a file over 100 MB, and inference does not need it.

The compact U-Net in this repository was not used for these 14863 masks.

## Classical check on 40 films

`segment_classical` runs only for the first 10 studies of each class. It is CLAHE, a bilateral filter, and Otsu, then a short morphological open and close. It follows the body silhouette. It is drawn in orange on `data/segmented/inspection/<class>/` where its contour differs from the U-Net. The U-Net contour is green. The green line is a 4-neighbor boundary in NumPy (`masks/agreement.py`), then a Pillow PNG. It is not a plotting library.

Consensus for those 40 films is the intersection of the two masks. The other 14823 films are `unet_only`.

## Label outcome

Anatomy QC uses the defaults in `QcThresholds` (area fraction, overlap, asymmetry). `configs/qc.yaml` records the same numbers. A hard anatomy failure is `rejected`. Everything else is `pseudo`, because a single model cannot be silver.

| Class | Pseudo | Rejected |
|---|---:|---:|
| Normal | 8802 | 49 |
| S1 | 1871 | 128 |
| S2 | 1808 | 198 |
| S3 | 1669 | 338 |

Anatomy pass: 14150. Anatomy fail: 713. Review scores are written with `active_learning_score` (1 − Dice, plus 0.05 per anatomy warning) and a severity-balanced `review_queue.csv`. That queue is a list to look at. It is not a new label.

Saved layout, all local:

- `data/segmented/masks/<class>/<id>.png` merged U-Net
- `data/segmented/left/` and `right/` the patient-side split
- `data/segmented/inspection/<class>/` 10 overlays per class
- `data/segmented/manifest.csv`, `provenance.jsonl`, `review_queue.csv`

## Compact U-Net training result

`CompactUNet` (features 32) was trained in this repo on the same external labeled set, as image and mask pairs. Loss is 0.5 BCE-with-logits + 0.5 soft Dice. The split is `assign_split` on the image stem, seed 2026.

| | |
|---|---|
| Checkpoint | `rsna2018_cxr/data/models/compact_unet.pt` (local, gitignored, about 31 MB) |
| Input size of this run | 256×256 |
| Best epoch | 7 |
| Val Dice | 0.9629 |
| Train / val pairs | 519 / 149 |

This checkpoint did not segment the severity cohort. Later training through `train` is locked to 1024×1024 so a new compact model matches the severity grid. Labeled source films that are not already 1024 are resized only inside that trainer.

## Inspection shapes

`masks/shape_completion.py` places a mean lung, built from complete Normal U-Net masks, under a flat or missing base. `render_completed_inspection` writes green contours to `data/segmented/inspection_shape/`. Those overlays are a look at the idea. They were not copied onto the 14863 masks. The shape is not finished: some severe films still stop in the mid-lung. Notes for a later shape model, starting with Post-DAE, are in `shape-constrained-crops.md`.

## Reproduce

```bash
python -m rsna2018_cxr.segmentation build-manifest --severity-root F:/severity_train
python -m rsna2018_cxr.segmentation segment --checkpoint F:/CXR_segmentation/models/unet_vgg11_1024_best.pth
```

`segment` defaults to that checkpoint, threshold 0.5, and batch 8.
