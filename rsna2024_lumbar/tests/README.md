# ✔ Tests — RSNA 2024 preprocess

```bash
# repo root
pip install -e ".[dev]"
pytest rsna2024_lumbar/tests
pytest rsna2024_lumbar/tests/test_catalog.py -k incomplete
```

GitHub Actions runs the same command against **committed** files in `fixtures/` (no Kaggle download, no `data/raw/`).

## ▾ Why a synthetic dump

The real set is huge and not committable. Tests read a **Kaggle-shaped** mini-root committed under `fixtures/` (`conftest.py` → `mini_raw`). Same files the exporter reads:

```
train.csv
train_label_coordinates.csv
train_series_descriptions.csv
train_images/<study_id>/<series_id>/<instance_number>.dcm
```

Slices are 128×128 MR-like synthetics with a bright 3×3 at **(64, 64)** so crop boxes are deterministic. See [fixtures/README.md](fixtures/README.md).

## ▾ Mini studies

| study | series | what | expected |
|---|---|---|---|
| 1001 | 2001 | SCS, all 5 levels, Normal/Mild, DICOMs 1–5 | **kept** — 5 jobs |
| 1002 | 2002 | same labels/coords, **no** `5.dcm` (L5/S1) | **dropped** — incomplete |
| 1003 | 2003 | SCS complete, L4/L5 = Moderate | **kept** — 5 jobs, one under `Moderate/` |

SCS only. Other condition columns in `train.csv` are empty. Catalog requires all five levels + files (same rule as training).

`--max-studies 1` = first row of `train.csv` only → 1001 if complete.

## ▾ What each file covers

| file | checks |
|---|---|
| `test_crop_geometry.py` | `centered` = 16×16 on 128px; `extend50` = 16×24; SCS grows **left**, LSS **right** |
| `test_catalog.py` | 1002 absent; 1001+1003 → 10 jobs; `max_studies=1` → only 1001 |
| `test_export.py` | 10 PNGs + `manifest.csv` + `export_meta.json`; 64×64; Moderate path for 1003 L4/L5; skip-existing exports 0 / skips 10; `png_relpath` contract |
| `test_scale_contract.py` | raw layout = Kaggle names; 1 study → 5 jobs, 2 complete → 10 (linear) |
| `test_params.py` | CLI defaults / `centered`+`extend50`+`--max-studies 1`; reject bad policy, condition, `max-studies 0`, missing data-root |
| `test_download.py` | dest is `data/raw/`; skip if complete; no kaggle CLI → SystemExit (no network) |
| `test_validate.py` | fixtures layout OK; missing root / incomplete dump fail; `--strict` flags 1002's missing DICOM |

## ▾ Pass / fail at a glance

```
centered, SCS, no max_studies
  jobs:      10   (not 15 — 1002 gone)
  PNGs:      10   under output_root/<condition>/<severity>/
  1001:      study_1001_l1_l2.png … l5_s1.png in Normal_Mild/
  1003 l4_l5: study_1003_l4_l5.png in Moderate/
  resume:    skip_existing → exported=0, skipped=10
```

A failure usually means: a broken study was kept, a complete one was dropped, crop side flipped, or the on-disk contract changed (trainer will not find PNGs).

## ⬇ Full dataset (not CI)

Copy the official dump into `../data/raw/` (or pass `--data-root`). Same functions, more rows. `--skip-existing` is the resume switch. `--max-studies N` is the smoke switch before a long export.
