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

Outputs under [`figures/patient_level/`](figures/patient_level/):

| Figure | PNG | PDF |
|--------|-----|-----|
| R0 / R1 / R2 stacked bars | [patient_error_profile_distribution.png](figures/patient_level/patient_error_profile_distribution.png) | [pdf](figures/patient_level/patient_error_profile_distribution.pdf) |
| Patient Exact vs O-MAE | [patient_exact_vs_omae_tradeoff.png](figures/patient_level/patient_exact_vs_omae_tradeoff.png) | [pdf](figures/patient_level/patient_exact_vs_omae_tradeoff.pdf) |
| R2 severe-error rate | [severe_error_patient_burden.png](figures/patient_level/severe_error_patient_burden.png) | [pdf](figures/patient_level/severe_error_patient_burden.pdf) |
| Audit table | [patient_level_plot_source_validation.csv](figures/patient_level/patient_level_plot_source_validation.csv) | |

Override paths with `--summary`, `--repeat-summary`, and `--output-dir`.

## Spinal-level (validation)

```bash
python -m rsna2024_lumbar.visualizations spinal-level
# same:
python -m rsna2024_lumbar.visualizations.plot_spinal_level_results
```

Reads `protocol_spinal_level/spinal_level_means.csv`, averages the five conditions into 40 architecture × representation × level rows, and stops if the schema is incomplete.

Outputs under [`figures/spinal_level/`](figures/spinal_level/):

| Figure | PNG | PDF |
|--------|-----|-----|
| Best vs worst Macro-F1 by level | [best_worst_spinal_level_bottleneck.png](figures/spinal_level/best_worst_spinal_level_bottleneck.png) | [pdf](figures/spinal_level/best_worst_spinal_level_bottleneck.pdf) |
| Macro-F1 and QWK profiles | [spinal_level_profile_plots.png](figures/spinal_level/spinal_level_profile_plots.png) | [pdf](figures/spinal_level/spinal_level_profile_plots.pdf) |
| Mean F1 / QWK / O-MAE / SER by level | [anatomically_grounded_multi_metric_overlay.png](figures/spinal_level/anatomically_grounded_multi_metric_overlay.png) | [pdf](figures/spinal_level/anatomically_grounded_multi_metric_overlay.pdf) |
| Macro-F1 heatmap (2D) | [spinal_level_macro_f1_heatmap_2d.png](figures/spinal_level/spinal_level_macro_f1_heatmap_2d.png) | [pdf](figures/spinal_level/spinal_level_macro_f1_heatmap_2d.pdf) |
| Macro-F1 heatmap (2.5D-to-3D) | [spinal_level_macro_f1_heatmap_25d_to_3d.png](figures/spinal_level/spinal_level_macro_f1_heatmap_25d_to_3d.png) | [pdf](figures/spinal_level/spinal_level_macro_f1_heatmap_25d_to_3d.pdf) |
| Combined heatmaps | [spinal_level_macro_f1_profiles.png](figures/spinal_level/spinal_level_macro_f1_profiles.png) | [pdf](figures/spinal_level/spinal_level_macro_f1_profiles.pdf) |
| 40-row audit table | [spinal_level_plot_source.csv](figures/spinal_level/spinal_level_plot_source.csv) | |
| Bottleneck / overlay tables | [spinal_level_bottleneck_source.csv](figures/spinal_level/spinal_level_bottleneck_source.csv) · [spinal_level_anatomical_overlay_source.csv](figures/spinal_level/spinal_level_anatomical_overlay_source.csv) | |

## Ordinal severity (validation)

```bash
python -m rsna2024_lumbar.visualizations ordinal
# same:
python -m rsna2024_lumbar.visualizations.plot_ordinal_severity_results
```

Reads `condition_specific_by_condition.csv` (or a wide `ordinal_condition_means.csv`), keeps the 40 architecture × representation × condition rows, and stops if the schema is incomplete.

Outputs under [`figures/ordinal/`](figures/ordinal/):

| Figure | PNG | PDF |
|--------|-----|-----|
| QWK vs O-MAE trade-off | [ordinal_qwk_vs_omae_tradeoff.png](figures/ordinal/ordinal_qwk_vs_omae_tradeoff.png) | [pdf](figures/ordinal/ordinal_qwk_vs_omae_tradeoff.pdf) |
| Ordinal metric heatmaps | [ordinal_metric_heatmaps.png](figures/ordinal/ordinal_metric_heatmaps.png) | [pdf](figures/ordinal/ordinal_metric_heatmaps.pdf) |
| 2D vs 2.5D gain / degradation | [ordinal_2d_vs_25d_gain_degradation.png](figures/ordinal/ordinal_2d_vs_25d_gain_degradation.png) | [pdf](figures/ordinal/ordinal_2d_vs_25d_gain_degradation.pdf) |
| Severe ordinal error burden | [severe_ordinal_error_burden.png](figures/ordinal/severe_ordinal_error_burden.png) | [pdf](figures/ordinal/severe_ordinal_error_burden.pdf) |
| Condition-level audit table | [ordinal_plot_source_condition_level.csv](figures/ordinal/ordinal_plot_source_condition_level.csv) | |
| Architecture-mean audit table | [ordinal_plot_source_architecture_means.csv](figures/ordinal/ordinal_plot_source_architecture_means.csv) | |
