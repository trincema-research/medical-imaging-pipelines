# ⬆ processed/ — crop cache

`centered/` (64×64 PNGs + `manifest.csv`) is committed so smoke/training can run without re-exporting. ~240MB.

Fill or refresh:

```bash
python -m rsna2024_lumbar.preprocessing --crop-policy centered --skip-existing --progress
```

`extend50/` stays local if you export it. Trainer loads this folder.
