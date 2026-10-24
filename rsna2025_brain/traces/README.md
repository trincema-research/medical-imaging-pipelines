# ☰ traces/ — one folder for every RSNA 2025 run

Gitignored `*.log` files. This README stays.

```
pipeline.log     all components, in time order  ← start here
download.log     python -m rsna2025_brain.data.download
unzip.log        python -m rsna2025_brain.data.unzip
validate.log     python -m rsna2025_brain.data.validate
```

Tokens are never written (only `token_set` / `KGAT_`).
