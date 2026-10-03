# ⬇ raw/ — labels + smoke studies in git

Committed:

- `train.csv`, `train_label_coordinates.csv`, `train_series_descriptions.csv`
- two **complete** studies (only the DICOM instances the cropper needs): `3318343342`, `3065863143`

Not committed: the Kaggle `.zip`, `*.kaggle-partial`, the rest of `train_images/` (~30GB), `test_images/`.

Download does **not** treat the two smoke studies as a full dump. It only skips when `train_images/` has a folder for every `study_id` in `train.csv`.

How to fill the rest: [../README.md](../README.md). Token is the repo-root `.env`.

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study_id>/<series_id>/<instance_number>.dcm
```
