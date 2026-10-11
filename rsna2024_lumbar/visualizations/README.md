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
