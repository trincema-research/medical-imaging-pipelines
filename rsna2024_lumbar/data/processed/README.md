# ⬆ processed/ — crop cache

Export with `rsna2024_lumbar.preprocessing` into `centered/` (64×64) or `extend50/` (96×64). The full PNG tree stays **local** (gitignored); CI uses `tests/fixtures/` instead.

```bash
python -m rsna2024_lumbar.preprocessing \
  --data-root rsna2024_lumbar/data/raw \
  --output-root rsna2024_lumbar/data/processed/centered \
  --crop-policy centered --skip-existing --progress
```

For cloud NAS, include crops in the deploy zip:

```bash
python -m rsna2024_lumbar.nas.pack --profile efficientnet_deploy --include-crops
```

Trainers and `nas.deploy` read `data/processed/<policy>/` by default.
