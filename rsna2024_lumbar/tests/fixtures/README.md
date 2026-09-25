# ✔ fixtures/ — committed mini dump (CI)

Same layout as Kaggle `data/raw/`. Checked in so GitHub Actions does not generate data.

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study>/<series>/<instance>.dcm
```

| study | what | expected |
|---|---|---|
| 1001 | complete SCS | kept (5 crops) |
| 1002 | missing last DICOM | skipped |
| 1003 | complete, L4/L5 Moderate | kept (5 crops) |

128×128 synthetic slices, bright dot at (64, 64). No patient data.

Regenerate: `python -m rsna2024_lumbar.tests.helpers`
