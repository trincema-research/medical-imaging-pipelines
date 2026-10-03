# ☰ traces/ — one folder for every RSNA 2024 run

Gitignored `*.log` files. This README stays. Open this directory to follow download → unzip → validate (and later preprocess) without hunting through the repo.

```
pipeline.log     all components, in time order  ← start here
download.log     python -m rsna2024_lumbar.data.download
unzip.log        python -m rsna2024_lumbar.data.unzip
validate.log     python -m rsna2024_lumbar.data.validate
```

Each CLI appends a `-------- timestamp  component  script --------` banner, then the same `START` / `STEP` / `DATA` / `OK` / `DONE` / `NEXT` / `WARN` / `FAIL` lines printed on stdout. Tokens are never written (only `token_set` / `KGAT_`).

`pipeline.log` is the full story. The per-component files are the same lines filtered to one script.
