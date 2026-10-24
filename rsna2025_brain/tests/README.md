# ✔ Tests — RSNA 2025 brain data

```bash
pip install -e ".[dev]"
pytest rsna2025_brain/tests
```

CI runs `pytest -m "not local"`. These tests use a **synthetic** Kaggle-shaped zip (`train.csv` + `train_localizers.csv` + `series/<uid>/*.dcm`). They never need the official dump or a token.
