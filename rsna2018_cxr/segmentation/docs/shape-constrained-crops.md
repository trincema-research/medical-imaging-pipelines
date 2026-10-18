# Shape-constrained lung crops

Design notes for later. Iteration 1 did not use these models. The run that wrote the masks is the local VGG11-BN U-Net, recorded in `iteration-1.md`.

Near term, keep generating masks with the U-Net and improve that loop on selected complete lungs. The first shape model to investigate after that is Post-DAE: a network trained on complete masks only, used to fit a whole lung onto the U-Net output for the chest film being processed.

## Enforced complete lung shape

**Question.** Select only perfect lung shapes, then force the U-Net to best-fit one statistical lung from those shapes and always emit the complete lung. If severe pneumonia removes one lung, fit that side from the detected lung or from other anatomical cues. Prefer a complete shape slightly off target over the fragments from the first pass. Later, an annotation tool can nudge those contours, and the corrected severe masks can teach a stronger shape-aware model.

**Answer.** The U-Net cannot be switched into that behavior. It paints whatever local pattern matches its training masks, so on a white lung it is free to return a fragment. The complete outline has to be applied after it, as a hard constraint on the mask that gets saved.

Build one statistical lung from the perfect Normal, S1, S2, and S3 masks that are accepted. Both lungs are in that model, including the costophrenic angles. At inference the U-Net is only a cue finder: apex, lateral wall, and whichever base is still visible. Those cues place the statistical shape. The file written to disk is that shape, both sides, every time. The raw U-Net pieces are not the mask.

When one lung is missing, the side that still has an apex and a wall fixes the scale and the height. The missing side is the mirror of that fitted lung across the mediastinum, taken from the same statistical shape. If neither side has a usable cue, the shape is placed on the chest using the body outline and the usual apex-to-base proportion. It will sit in the right part of the film and it will be a whole lung. On a rotated portable film it can be shifted or a little too wide.

Normal films that already match the statistical lung stay close to the contours that already look right. Severity films stop showing scraps. Some bases will sit slightly high or low, and a high diaphragm can be pulled toward the average lung.

An annotation pass that nudges those complete contours is the useful label, because the edit records where the average lung was wrong on an opaque film. Two things update from those edits. The statistical shape, so the next fit starts closer on severe chests. The U-Net, so it gets better at finding the remaining apex and chest wall under pneumonia. The U-Net still should not be the source of the outline. Once enough severe corrections exist, its cues get sharper, and the shape model carries the complete lung.

## Combining published methods for high-quality crops

**Question.** Combine the atlas, statistical-shape, Post-DAE, and graph-decoder techniques with current models so the dataset yields very high quality lung crops. Design only.

**Answer.** The reliable design is a shape model in front of the U-Net, with a person only nudging the complete contour. The crop is taken from that contour, so the opaque lung stays inside it.

Every study gets one bilateral lung. It is always a whole shape from the library of perfect masks. The U-Net does not write the file. It only supplies cues: apex, lateral wall, and any base that is still visible. The saved mask is the library shape placed on those cues. A missing side is the same shape mirrored across the mediastinum. If both sides are washed out, the shape is placed on the chest from the body outline and the usual apex-to-base proportion. The crop is the bounding region of that mask, with a small margin, on the original 1024 grid.

Three layers:

1. **Cues.** Keep the current VGG11 U-Net. It is already good where the lung is dark. On a severe film its fragments are evidence of where air remains, not the outline. A promptable segmenter such as MedSAM can sit beside it later as a local editor. It finds a border that is pointed at. It does not know that a white hemithorax is still lung, so it does not own the mask.

2. **Shape.** This is the Candemir / statistical-shape / Post-DAE idea, built from the selected perfect lungs, including any severe ones already trusted. A compact model is enough: the mean contour plus a few principal modes, or a small autoencoder trained only on those masks. Fitting uses the U-Net cues and, when one lung is gone, the other lung. The decoder can only emit a lung from that family, so scraps and holes cannot come out. Post-DAE is the same idea with a denoising autoencoder instead of PCA. HybridGNet is the stronger form once enough corrected severe masks exist: the image encoder reads the film, and a graph decoder emits a fixed set of landmarks, so the contour is a lung by construction.

3. **Adjustment.** The review screen shows the complete contour, not the U-Net pieces. The base or one side is dragged. The edit is projected back onto the shape model, so a correction cannot leave a fragment. Those edits are gold. They update the shape library first. The U-Net is retrained only after a batch of severe corrections exists, and those corrections are up-weighted so the Normal films do not wash them out. The U-Net’s new job is to find the remaining apex and chest wall under pneumonia. The outline still comes from the shape layer.

How a round goes. Select the perfect masks across Normal, S1, S2, and S3. Fit the shape model on those only. Place it on all 14,863 films. Review a fixed severe holdout plus the films where the fit moved a long way from the U-Net cues. Correct those, add them to the library, and refit. Quality on the holdout is whether both lungs are complete and the opaque lung is inside the crop, not whether the contour matches the old aerated fragment.

Normal films stay close to the contours already in the library. Severity films stop returning pieces. Early severe crops will be a complete lung in approximately the right place, high or low by a rib or two when the diaphragm is invisible. Each correction of a white-out film teaches the library that shape.

## References

- Candemir et al., IEEE TMI, 2014. Anatomical atlases with nonrigid registration. https://doi.org/10.1109/TMI.2013.2290491
- Shi et al., IEEE TMI, 2008. Population and patient-specific lung shape statistics. https://doi.org/10.1109/TMI.2007.908130
- Mansoor et al., IEEE TBME, 2020. Deep prediction of statistical lung-shape coefficients. https://doi.org/10.1109/TBME.2019.2933508
- Larrazabal, Martínez, Glocker, and Ferrante, IEEE TMI, 2020. Post-DAE. https://doi.org/10.1109/TMI.2020.3005297
- Gaggion et al., IEEE TMI, 2023. HybridGNet. https://doi.org/10.1109/TMI.2022.3224660
