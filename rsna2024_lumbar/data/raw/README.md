# data/raw — local only

Drop the Kaggle RSNA 2024 files here (not committed):

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study_id>/<series_id>/<instance_number>.dcm
```

Or point `--data-root` at any existing copy. CI uses `tests/fixtures/` instead.
