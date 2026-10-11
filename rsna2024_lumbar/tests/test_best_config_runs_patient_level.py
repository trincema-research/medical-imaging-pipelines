from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.patient_level import (
    build_patient_runs,
    compute_patient_metrics,
    error_profile_from_max_error,
    leakage_warnings,
    summarize_across_repeats,
    summarize_repeats,
    write_patient_level_outputs,
)


def test_profiles_and_no_ploa():
    exact = compute_patient_metrics([0, 1, 2], [0, 1, 2])
    assert exact["error_profile"] == "exact"
    assert exact["patient_exact_rate"] == 1.0
    assert exact["patient_omae"] == 0.0
    assert "ploa" not in exact

    adjacent = compute_patient_metrics([0, 1, 2], [0, 2, 2])
    assert adjacent["error_profile"] == "adjacent_only"
    assert adjacent["patient_max_error"] == 1
    assert adjacent["has_severe_error"] == 0

    severe = compute_patient_metrics([0, 1], [2, 1])
    assert severe["error_profile"] == "severe"
    assert severe["patient_max_error"] == 2
    assert severe["n_severe_errors"] == 1
    assert error_profile_from_max_error(2) == "severe"
    with pytest.raises(ValueError):
        error_profile_from_max_error(3)


def test_equal_weight_macro_not_pooled():
    """Two patients: 1/1 exact vs 1/4 exact. Macro is 0.625; pooled is 0.4."""
    predictions = pd.DataFrame(
        [
            _pred(patient_id=1, y_true=0, y_pred=0, spinal_level="L1/L2"),
            _pred(patient_id=2, y_true=0, y_pred=0, spinal_level="L1/L2"),
            _pred(patient_id=2, y_true=1, y_pred=2, spinal_level="L2/L3"),
            _pred(patient_id=2, y_true=2, y_pred=1, spinal_level="L3/L4"),
            _pred(patient_id=2, y_true=0, y_pred=1, spinal_level="L4/L5"),
        ]
    )
    runs = build_patient_runs(predictions)
    assert len(runs) == 2
    assert runs["patient_exact_rate"].mean() == pytest.approx(0.625)
    pooled = runs["n_exact"].sum() / runs["n_predictions"].sum()
    assert pooled == pytest.approx(0.4)
    assert runs["patient_exact_rate"].mean() != pytest.approx(pooled)


def _pred(**overrides) -> dict:
    rec = {
        "patient_id": 10,
        "condition": "SCS",
        "spinal_level": "L1/L2",
        "split": "validation",
        "y_true": 0,
        "y_pred": 0,
        "architecture": "ViT",
        "representation": "2D",
        "repeat": 1,
        "seed": 42,
        "trial_id": 1,
        "best_epoch": 3,
    }
    rec.update(overrides)
    return rec


def _five_seed_predictions(*, n_repeats: int = 5, n_patients: int = 4) -> pd.DataFrame:
    rows: list[dict] = []
    for repeat in range(1, n_repeats + 1):
        seed = 41 + repeat
        for pid in range(1, n_patients + 1):
            # Patient 1: all exact. Patient 2: adjacent. Patient 3: severe. Patient 4: mixed exact.
            if pid == 1:
                pairs = [(0, 0), (1, 1)]
            elif pid == 2:
                pairs = [(0, 1), (1, 1)]
            elif pid == 3:
                pairs = [(0, 2), (1, 1)]
            else:
                pairs = [(2, 2), (0, 0)]
            for cond in ("SCS", "LFN"):
                for level, (yt, yp) in zip(("L1/L2", "L2/L3"), pairs):
                    rows.append(
                        _pred(
                            patient_id=pid,
                            condition=cond,
                            spinal_level=level,
                            y_true=yt,
                            y_pred=yp,
                            repeat=repeat,
                            seed=seed,
                        )
                    )
    return pd.DataFrame(rows)


def test_profile_rates_sum_to_one_and_incomplete_flag():
    runs = build_patient_runs(_five_seed_predictions())
    repeats = summarize_repeats(runs)
    row = repeats[(repeats["split"] == "validation") & (repeats["repeat"] == 1)].iloc[0]
    profile_sum = (
        row["patient_exact_profile_rate"]
        + row["patient_adjacent_only_rate"]
        + row["patient_severe_error_rate"]
    )
    assert profile_sum == pytest.approx(1.0)
    p1 = runs[(runs["patient_id"] == 1) & (runs["repeat"] == 1)].iloc[0]
    assert p1["patient_exact_rate"] + p1["patient_error_rate"] == pytest.approx(1.0)

    complete = summarize_across_repeats(repeats)
    assert (complete["status"] == "complete").all()
    assert int(complete.iloc[0]["n_repeats"]) == 5

    incomplete_repeats = summarize_repeats(build_patient_runs(_five_seed_predictions(n_repeats=4)))
    incomplete = summarize_across_repeats(incomplete_repeats)
    assert (incomplete["status"] == "incomplete").all()


def test_leakage_across_splits():
    df = pd.DataFrame(
        [
            _pred(patient_id=1, split="train"),
            _pred(patient_id=1, split="validation"),
        ]
    )
    warns = leakage_warnings(df)
    assert any("train and validation" in w for w in warns)


def test_write_outputs(tmp_path: Path):
    df = _five_seed_predictions()
    paths = write_patient_level_outputs(
        tmp_path,
        pipeline_df=pd.DataFrame(),
        prediction_df=df,
        print_summary=False,
    )
    assert paths["runs"].is_file()
    assert paths["repeat_summary"].is_file()
    assert paths["summary"].is_file()
    assert paths["paper_table"].is_file()
    assert (tmp_path / "patient_error_profile_validation.csv").is_file()
    assert (tmp_path / "patient_exact_rate_distribution_validation.csv").is_file()
    assert (tmp_path / "patient_omae_distribution_test.csv").is_file()
    payload = (tmp_path / "patient_level_summary.json").read_text(encoding="utf-8")
    assert "ploa" not in pd.read_csv(paths["runs"]).columns
    assert '"ploa_implemented": false' in payload
