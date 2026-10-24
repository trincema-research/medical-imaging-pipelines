# ⬇ data/ — scripts + local dumps

```
download.py      ⬇  Kaggle zip → raw/ (unzips; exits non-zero on failure)
unzip.py         ⬇  extract a .zip already in raw/
validate.py      ✔  check a dump before later preprocess
raw/             ⬇  train.csv + train_localizers.csv (optional in git) · series/ local
processed/       ⬆  later crop cache (local)
```

Scripts stay **here** (not in `raw/`) so `python -m rsna2025_brain.data.<name>` works. DICOMs stay local.

## 🔑 Auth (same token as lumbar)

```bash
# from medical-imaging-pipelines/
copy .env.example .env
```

Set `KAGGLE_API_TOKEN=` from [API settings](https://www.kaggle.com/settings/api) → Generate New Token. The same repo-root `.env` is what lumbar already uses. Accept the [competition rules](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/rules) first. Download loads `.env` automatically.

## Commands

```bash
pip install -e ".[kaggle]"
python -m rsna2025_brain.data.download    # zip + unzip → raw/
python -m rsna2025_brain.data.unzip       # if a .zip is still in raw/
python -m rsna2025_brain.data.validate    # default: raw/
```

`--force` re-downloads. `--keep-zip` extracts but leaves the archive. `--dest` overrides `raw/`. A `*.kaggle-partial` file means the transfer is still running or was cut off — re-run download (do not unzip yet).

Expected `raw/` layout: [raw/README.md](raw/README.md).
