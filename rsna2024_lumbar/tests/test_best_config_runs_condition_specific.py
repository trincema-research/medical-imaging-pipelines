from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.condition_specific import (
    aggregate_by_condition,
    architecture_label,
    build_radar_frame,
    build_runs_dataframe,
    representation_label,
    summarize_pipeline,
    write_condition_specific_outputs,
    _best_worst,
    _oa_equals_accuracy,
)
from rsna2024_lumbar.best_config_runs.results import PIPELINE_RESULTS_COLUMNS


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
            "val_accuracy_overall": 0.90,
            "val_oa_overall": 0.90,
            "val_f1_macro_overall": 0.80,
            "val_precision_macro_overall": 0.81,
            "val_recall_macro_overall": 0.79,
            "val_omae_overall": 0.10,
            "val_qwk_overall": 0.70,
            "val_ser_overall": 0.02,
            "test_accuracy_overall": 0.88,
            "test_oa_overall": 0.88,
        }
    )
    row.update(overrides)
    return row


def test_architecture_and_representation_labels():
    assert architecture_label("convnext3d") == "ConvNeXt"
    assert architecture_label("efficientnet") == "EfficientNet"
    assert architecture_label("maxvit3d") == "MaxViT"
    assert representation_label("2d") == "2D"
    assert representation_label("3d_level_stack") == "2.5D-to-3D"
    assert representation_label("3d") == "2.5D-to-3D"


def test_oa_equals_accuracy():
    assert _oa_equals_accuracy(0.91, 0.91) is True
    assert _oa_equals_accuracy(0.91, 0.90) is False
    assert _oa_equals_accuracy("", 0.90) is None


def test_best_worst_higher_and_lower_is_better():
    means = {"SCS": 0.9, "LFN": 0.8, "RFN": 0.5, "LSS": 0.7, "RSS": 0.6}
    assert _best_worst(means, higher_is_better=True) == ("SCS", "RFN")
    omae = {"SCS": 0.10, "LFN": 0.20, "RFN": 0.50, "LSS": 0.30, "RSS": 0.40}
    assert _best_worst(omae, higher_is_better=False) == ("SCS", "RFN")


def _five_condition_pipeline(*, n_repeats: int = 5) -> pd.DataFrame:
    """SCS..RSS val accuracy means 0.90, 0.80, 0.70, 0.60, 0.50; O-MAE inverse."""
    slugs = [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    accs = [0.90, 0.80, 0.70, 0.60, 0.50]
    rows: list[dict] = []
    for slug, acc in zip(slugs, accs):
        for repeat in range(1, n_repeats + 1):
            rows.append(
                _pipeline_row(
                    condition=slug,
                    repeat_index=repeat,
                    split_seed=41 + repeat,
                    val_accuracy_overall=acc,
                    val_oa_overall=acc,
                    val_omae_overall=round(1.0 - acc, 2),
                    val_f1_macro_overall=acc,
                    val_qwk_overall=acc,
                    val_ser_overall=round(1.0 - acc, 2),
                )
            )
    return pd.DataFrame(rows)


def test_condition_macro_mean_best_worst_range_std():
    runs = build_runs_dataframe(_five_condition_pipeline())
    by_cond = aggregate_by_condition(runs)
    summary = summarize_pipeline(by_cond)
    oa = summary[(summary["metric"] == "oa_overall") & (summary["split"] == "validation")].iloc[0]
    assert oa["condition_macro_mean"] == pytest.approx(0.70)
    assert oa["best_condition"] == "SCS"
    assert oa["worst_condition"] == "RSS"
    assert oa["condition_range"] == pytest.approx(0.40)
    assert oa["condition_std"] == pytest.approx(0.158113883, abs=1e-6)
    assert oa["status"] == "complete"

    omae = summary[(summary["metric"] == "omae_overall") & (summary["split"] == "validation")].iloc[0]
    assert omae["best_condition"] == "SCS"
    assert omae["worst_condition"] == "RSS"
    assert omae["condition_macro_mean"] == pytest.approx(0.30)


def test_incomplete_repeats_are_flagged():
    df = _five_condition_pipeline(n_repeats=4)
    runs = build_runs_dataframe(df)
    by_cond = aggregate_by_condition(runs)
    scs = by_cond[
        (by_cond["condition"] == "SCS")
        & (by_cond["metric"] == "oa_overall")
        & (by_cond["split"] == "validation")
    ].iloc[0]
    assert scs["n_runs"] == 4
    assert scs["status"] == "incomplete"
    summary = summarize_pipeline(by_cond)
    oa = summary[(summary["metric"] == "oa_overall") & (summary["split"] == "validation")].iloc[0]
    assert oa["status"] == "incomplete"
    assert oa["condition_macro_mean"] == pytest.approx(0.70)


def test_radar_wide_columns(tmp_path: Path):
    df = _five_condition_pipeline()
    runs = build_runs_dataframe(df)
    by_cond = aggregate_by_condition(runs)
    radar = build_radar_frame(by_cond, metric="oa_overall", split="validation")
    assert list(radar.columns)[:7] == [
        "architecture",
        "representation",
        "SCS",
        "LFN",
        "RFN",
        "LSS",
        "RSS",
    ]
    assert radar.iloc[0]["SCS"] == pytest.approx(0.90)
    assert radar.iloc[0]["condition_macro_mean"] == pytest.approx(0.70)

    paths = write_condition_specific_outputs(tmp_path, pipeline_df=df)
    assert paths["runs"].is_file()
    assert paths["by_condition"].is_file()
    assert paths["pipeline_summary"].is_file()
    assert paths["paper_table"].is_file()
    assert paths["summary_json"].is_file()
    assert (tmp_path / "condition_specific_radar_oa_validation.csv").is_file()
    runs_df = pd.read_csv(paths["runs"])
    assert "accuracy_overall" in runs_df.columns
    assert "oa_overall" in runs_df.columns
    assert set(runs_df["split"].unique()) >= {"validation", "test", "train"}
