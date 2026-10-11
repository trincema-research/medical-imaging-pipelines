# Visualizations

Independent figure module, same level as `data/` and `best_config_runs/`. Plots are generated only from harvested CSVs — never from manuscript numbers.

| | |
|--|--|
| **Package** | `rsna2024_lumbar/visualizations/` |
| **Source CSVs** | `data/best_config_runs/results/` (protocol trees when present) |
| **Committed figures** | `figures/` |

## Patient-level (validation)

```bash
pip install -e ".[visualizations,dev]"

python -m rsna2024_lumbar.visualizations patient-level
# same:
python -m rsna2024_lumbar.visualizations.plot_patient_level_results
```

Reads `protocol_patient_level/patient_level_summary.csv` and `patient_level_repeat_summary.csv`, recomputes validation means from the repeat file, and stops if they disagree.

Outputs under `figures/patient_level/`:

| File | Figure |
|------|--------|
| `patient_error_profile_distribution.png` / `.pdf` | R0 / R1 / R2 stacked bars |
| `patient_exact_vs_omae_tradeoff.png` / `.pdf` | Patient Exact vs O-MAE |
| `severe_error_patient_burden.png` / `.pdf` | R2 severe-error rate |
| `patient_level_plot_source_validation.csv` | Audit table used for the plots |
| `*_metadata.json` | Source paths, git commit, run counts |

Override paths with `--summary`, `--repeat-summary`, and `--output-dir`.

## Spinal-level (validation)

```bash
python -m rsna2024_lumbar.visualizations spinal-level
# same:
python -m rsna2024_lumbar.visualizations.plot_spinal_level_results
```

Reads `protocol_spinal_level/spinal_level_means.csv`, averages the five conditions into 40 architecture × representation × level rows, and stops if the schema is incomplete.

Outputs under `figures/spinal_level/`:

| File | Figure |
|------|--------|
| `best_worst_spinal_level_bottleneck.png` / `.pdf` | Best vs worst Macro-F1 by level |
| `spinal_level_profile_plots.png` / `.pdf` | Macro-F1 and QWK profiles |
| `anatomically_grounded_multi_metric_overlay.png` / `.pdf` | Mean F1 / QWK / O-MAE / SER by level |
| `spinal_level_macro_f1_heatmap_2d.png` / `..._25d_to_3d.png` | Shared-scale Macro-F1 heatmaps |
| `spinal_level_plot_source.csv` | 40-row audit table |
| `spinal_level_bottleneck_source.csv` / `spinal_level_anatomical_overlay_source.csv` | Derived tables |
