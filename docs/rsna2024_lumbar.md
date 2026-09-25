# RSNA 2024 lumbar — preprocess

Input (Kaggle / `data/raw/`):

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study>/<series>/<instance>.dcm
```

Policies:

| name | crop | PNG |
|---|---|---|
| `centered` | 12.5% × 12.5% on (x, y) | 64×64 |
| `extend50` | same base + 50% width (L/R by condition) | 96×64 |

Output (`data/processed/<policy>/`):

```
<condition>/<Normal_Mild|Moderate|Severe>/study_<id>_<level>.png
manifest.csv
export_meta.json
```

Training later loads this folder. Incomplete studies (missing a level/DICOM) are skipped by default.
