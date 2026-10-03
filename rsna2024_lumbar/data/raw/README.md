# ⬇ raw/ — dump destination

Gitignored (this README stays). ~30GB. CI never reads here (`✔ tests/fixtures/` instead).

How to fill it: [../README.md](../README.md). Download unzips automatically; if a `.zip` is left here run `python -m rsna2024_lumbar.data.unzip`. A `*.kaggle-partial` file means the transfer is not finished — do not unzip yet. Token is the repo-root `.env`, not this folder.

Expected files:

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study_id>/<series_id>/<instance_number>.dcm
```
