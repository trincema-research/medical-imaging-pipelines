# ▸ RSNA 2025 brain

Intracranial aneurysm detection. This PR adds the same download → unpack → validate path as lumbar, using the **same** repo-root `KAGGLE_API_TOKEN`.

```
⚙  data/            download · unzip · validate
⬇  data/raw/        Kaggle extract (series/ gitignored)
✔  tests/           download / unzip / validate (synthetic zip)
```

```bash
# from medical-imaging-pipelines/
copy .env.example .env    # KAGGLE_API_TOKEN (same file as lumbar)
pip install -e ".[kaggle,dev]"
python -m rsna2025_brain.data.download
python -m rsna2025_brain.data.unzip       # only if a .zip is still in raw/
python -m rsna2025_brain.data.validate
pytest rsna2025_brain/tests
```

`--dest` overrides `rsna2025_brain/data/raw`. `--force` re-downloads. `--keep-zip` leaves the archive.

Competition: [rsna-intracranial-aneurysm-detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection). Depth: [data/README.md](data/README.md).
