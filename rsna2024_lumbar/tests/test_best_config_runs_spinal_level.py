from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.spinal_level import (
    aggregate_level_means,
    build_heatmap_frame,
    build_runs_dataframe,
    canonical_spinal_level,
    summarize_by_condition,
    summarize_pipeline,
    write_spinal_level_outputs,
)
from rsna2024_lumbar.best_config_runs.results import PIPELINE_RESULTS_COLUMNS
from rsna2024_lumbar.preprocessing.constants import LUMBAR_LEVELS


def test_canonical_spinal_level_normalizes_aliases():
    assert canonical_spinal_level("l1_l2") == "L1/L2"
    assert canonical_spinal_level("L1-L2") == "L1/L2"
    assert canonical_spinal_level("L4/L5") == "L4/L5"
    with pytest.raises(ValueError):
        canonical_spinal_level("L6/S1")


def _pipeline_row(**overrides) -> dict:
    row = {c: "" for c in PIPELINE_RESULTS_COLUMNS}
    row.update(
        {
            "model_config": "nas_best_vit_2d",
            "model_label": "ViT 2D",
            "family": "vit",
            "archive_layout": "2d",
            "condition": "spinal_canal_stenosis",
            "repeat_index": 1,
            "split_seed": 42,
        }
    )
    row.update(overrides)
    return row


def _five_level_pipeline(*, n_repeats: int = 5, condition: str = "spinal_canal_stenosis") -> pd.DataFrame:
    """Val per-level accuracy/OA = 0.90, 0.80, 0.70, 0.60, 0.50; O-MAE inverse."""
    accs = [0.90, 0.80, 0.70, 0.60, 0.50]
    rows: list[dict] = []
    for repeat in range(1, n_repeats + 1):
        rec = _pipeline_row(condition=condition, repeat_index=repeat, split_seed=41 + repeat)
        for slug, acc in zip(LUMBAR_LEVELS, accs):
            rec[f"val_accuracy_{slug}"] = acc
            rec[f"val_f1_macro_{slug}"] = acc
            rec[f"val_precision_macro_{slug}"] = acc
            rec[f"val_recall_macro_{slug}"] = acc
            sev = slug.upper()
            rec[f"val_oa_{sev}"] = acc
            rec[f"val_omae_{sev}"] = round(1.0 - acc, 2)
            rec[f"val_qwk_{sev}"] = acc
            rec[f"val_ser_{sev}"] = round(1.0 - acc, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def test_macro_level_mean_best_worst_range_std():
    runs = build_runs_dataframe(_five_level_pipeline())
    means = aggregate_level_means(runs)
    by_cond = summarize_by_condition(means)
    oa = by_cond[(by_cond["metric"] == "oa") & (by_cond["split"] == "validation")].iloc[0]
    assert oa["macro_level_mean"] == pytest.approx(0.70)
    assert oa["best_level"] == "L1/L2"
    assert oa["worst_level"] == "L5/S1"
    assert oa["level_range"] == pytest.approx(0.40)
    assert oa["level_std"] == pytest.approx(0.158113883, abs=1e-6)
    assert oa["status"] == "complete"

    omae = by_cond[(by_cond["metric"] == "omae") & (by_cond["split"] == "validation")].iloc[0]
    assert omae["best_level"] == "L1/L2"
    assert omae["worst_level"] == "L5/S1"


def test_incomplete_repeats_are_flagged():
    runs = build_runs_dataframe(_five_level_pipeline(n_repeats=4))
    means = aggregate_level_means(runs)
    assert (means["status"] == "incomplete").all()
    assert int(means.iloc[0]["n_repeats"]) == 4
    by_cond = summarize_by_condition(means)
    assert (by_cond["status"] == "incomplete").all()


def test_oa_equals_accuracy_and_heatmap(tmp_path: Path):
    slugs = [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    df = pd.concat([_five_level_pipeline(condition=c) for c in slugs], ignore_index=True)
    runs = build_runs_dataframe(df)
    assert bool(runs["oa_equals_accuracy"].dropna().all())
    means = aggregate_level_means(runs)
    pipe = summarize_pipeline(means)
    oa = pipe[(pipe["metric"] == "oa") & (pipe["split"] == "validation")].iloc[0]
    assert oa["macro_level_mean"] == pytest.approx(0.70)
    heat = build_heatmap_frame(pipe, metric="f1_macro", split="validation")
    assert heat.iloc[0]["pipeline"] == "ViT - 2D"
    assert list(heat.columns[1:]) == ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    paths = write_spinal_level_outputs(tmp_path, pipeline_df=df, print_summary=False)
    assert paths["runs"].is_file()
    assert paths["means"].is_file()
    assert paths["condition_summary"].is_file()
    assert paths["pipeline_summary"].is_file()
    assert paths["paper_table"].is_file()
    assert (tmp_path / "spinal_level_radar_validation.csv").is_file()
    assert (tmp_path / "spinal_level_heatmap_macro_f1_validation.csv").is_file()
