# ⬇ raw/ — official Kaggle extract (local)

Download writes here. Do **not** commit the `.zip`, `series/`, or `segmentations/`.

```
train.csv
train_localizers.csv
series/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
segmentations/          # optional subset; not required to start
```

How to fill it:

```bash
copy .env.example .env    # same KAGGLE_API_TOKEN as lumbar
python -m rsna2025_brain.data.download
python -m rsna2025_brain.data.validate
```

Token is the repo-root `.env`. Competition: [rsna-intracranial-aneurysm-detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection).
