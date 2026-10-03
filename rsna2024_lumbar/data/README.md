# ⬇ data/ — scripts + local dumps

```
download.py      ⬇  Kaggle zip → raw/ (unzips; exits non-zero on failure)
unzip.py         ⬇  extract a .zip already in raw/
validate.py      ✔  check a dump before preprocess
raw/             ⬇  official dump (~30GB, gitignored)
processed/       ⬆  crop PNG cache (gitignored)
```

Scripts stay **here** (not in `raw/`) so git can track them and `python -m rsna2024_lumbar.data.<name>` works. `raw/` and `processed/` only hold large files plus a short README each.

## 🔑 Auth (repo-root `.env`)

```bash
# from medical-imaging-pipelines/
copy .env.example .env
```

Set `KAGGLE_API_TOKEN=` from [API settings](https://www.kaggle.com/settings/api) → Generate New Token. Do not use the old `kaggle.json` username/key. Accept the [competition rules](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules) first. Download loads `.env` automatically.

## Commands

```bash
pip install -e ".[kaggle]"
python -m rsna2024_lumbar.data.download    # zip + unzip → raw/  (fails on CLI error / partial / bad zip)
python -m rsna2024_lumbar.data.unzip       # if a .zip is still in raw/
python -m rsna2024_lumbar.data.validate    # default: raw/
python -m rsna2024_lumbar.data.validate --data-root rsna2024_lumbar/tests/fixtures
```

`--force` re-downloads. `--keep-zip` extracts but leaves the archive. `--dest` overrides `raw/`. A `*.kaggle-partial` file means the transfer is still running or was cut off — re-run download (do not unzip yet).

Every data script prints a trace (`START` · `STEP` · `DATA` · `OK` · `DONE` · `NEXT` · `WARN` · `FAIL`). Tokens are never logged (only `token_set` / `KGAT_` prefix). The same lines append under [../traces/](../traces/README.md): `download.log` / `unzip.log` / `validate.log` plus `pipeline.log` (all components, time order).

Expected `raw/` layout: [raw/README.md](raw/README.md). Crops: [processed/README.md](processed/README.md).
