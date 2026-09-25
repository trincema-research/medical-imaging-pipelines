# ⬇ data/raw — local only

Gitignored. Fill with:

```bash
python -m rsna2024_lumbar.data.download
```

or drop the Kaggle files here by hand:

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study_id>/<series_id>/<instance_number>.dcm
```

Or point `--data-root` at any existing copy. CI uses `✔ tests/` instead.
