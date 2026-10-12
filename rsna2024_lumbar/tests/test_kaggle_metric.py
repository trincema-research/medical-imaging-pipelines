from __future__ import annotations

import pandas as pd
import pytest

from rsna2024_lumbar.best_config_runs.kaggle_metric import (
    ANY_SEVERE_SCALAR_OFFICIAL,
    ParticipantVisibleError,
    frames_from_predictions,
    get_condition,
    level_slug,
    prediction_row_id,
    score,
    score_parts,
    score_prediction_table,
)
from rsna2024_lumbar.best_config_runs.kaggle_score import (
    architecture_from_stem,
    representation_from_stem,
)


def _solution_row(row_id: str, cls: int, weight: float) -> dict:
    one_hot = [0.0, 0.0, 0.0]
    one_hot[cls] = 1.0
    return {
        "row_id": row_id,
        "normal_mild": one_hot[0],
        "moderate": one_hot[1],
        "severe": one_hot[2],
        "sample_weight": weight,
    }


def _submission_row(row_id: str, probs: tuple[float, float, float]) -> dict:
    return {
        "row_id": row_id,
        "normal_mild": probs[0],
        "moderate": probs[1],
        "severe": probs[2],
    }


def _minimal_frames(scs_severe_prob: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    solution = pd.DataFrame(
        [
            _solution_row("100_spinal_canal_stenosis_l1_l2", 0, 1.0),
            _solution_row("100_spinal_canal_stenosis_l2_l3", 2, 4.0),
            _solution_row("101_spinal_canal_stenosis_l1_l2", 0, 1.0),
            _solution_row("100_left_neural_foraminal_narrowing_l1_l2", 1, 2.0),
            _solution_row("100_left_subarticular_stenosis_l1_l2", 0, 1.0),
        ]
    )
    submission = pd.DataFrame(
        [
            _submission_row("100_spinal_canal_stenosis_l1_l2", (0.90, 0.07, 0.03)),
            _submission_row("100_spinal_canal_stenosis_l2_l3", (0.05, 0.10, scs_severe_prob)),
            _submission_row("101_spinal_canal_stenosis_l1_l2", (0.88, 0.08, 0.04)),
            _submission_row("100_left_neural_foraminal_narrowing_l1_l2", (0.10, 0.80, 0.10)),
            _submission_row("100_left_subarticular_stenosis_l1_l2", (0.85, 0.10, 0.05)),
        ]
    )
    rest = 1.0 - scs_severe_prob
    submission.loc[1, "normal_mild"] = rest * 0.3
    submission.loc[1, "moderate"] = rest * 0.7
    return solution, submission


def test_get_condition_groups():
    assert get_condition("spinal_canal_stenosis_l1_l2") == "spinal"
    assert get_condition("left_neural_foraminal_narrowing_l4_l5") == "foraminal"
    assert get_condition("right_subarticular_stenosis_l5_s1") == "subarticular"


def test_level_slug_and_row_id():
    assert level_slug("L1/L2") == "l1_l2"
    assert prediction_row_id(4003253, "spinal_canal_stenosis", "L3/L4") == (
        "4003253_spinal_canal_stenosis_l3_l4"
    )


def test_perfect_one_hot_score_is_near_zero():
    solution = pd.DataFrame(
        [
            _solution_row("1_spinal_canal_stenosis_l1_l2", 0, 1.0),
            _solution_row("2_spinal_canal_stenosis_l1_l2", 2, 4.0),
            _solution_row("1_left_neural_foraminal_narrowing_l1_l2", 1, 2.0),
            _solution_row("1_left_subarticular_stenosis_l1_l2", 2, 4.0),
        ]
    )
    submission = pd.DataFrame(
        [
            _submission_row("1_spinal_canal_stenosis_l1_l2", (1.0, 0.0, 0.0)),
            _submission_row("2_spinal_canal_stenosis_l1_l2", (0.0, 0.0, 1.0)),
            _submission_row("1_left_neural_foraminal_narrowing_l1_l2", (0.0, 1.0, 0.0)),
            _submission_row("1_left_subarticular_stenosis_l1_l2", (0.0, 0.0, 1.0)),
        ]
    )
    value = score(solution, submission, "row_id", ANY_SEVERE_SCALAR_OFFICIAL)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_worse_severe_probability_increases_score():
    better, better_sub = _minimal_frames(scs_severe_prob=0.80)
    worse, worse_sub = _minimal_frames(scs_severe_prob=0.10)
    good = score(better, better_sub, "row_id", ANY_SEVERE_SCALAR_OFFICIAL)
    bad = score(worse, worse_sub, "row_id", ANY_SEVERE_SCALAR_OFFICIAL)
    assert bad > good


def test_frames_from_predictions_and_table_score():
    pred = pd.DataFrame(
        [
            {
                "patient_id": 100,
                "condition": "spinal_canal_stenosis",
                "spinal_level": "L1/L2",
                "y_true": 0,
                "p_class_0": 0.9,
                "p_class_1": 0.07,
                "p_class_2": 0.03,
            },
            {
                "patient_id": 100,
                "condition": "spinal_canal_stenosis",
                "spinal_level": "L2/L3",
                "y_true": 2,
                "p_class_0": 0.05,
                "p_class_1": 0.10,
                "p_class_2": 0.85,
            },
            {
                "patient_id": 100,
                "condition": "left_neural_foraminal_narrowing",
                "spinal_level": "L1/L2",
                "y_true": 1,
                "p_class_0": 0.1,
                "p_class_1": 0.8,
                "p_class_2": 0.1,
            },
            {
                "patient_id": 100,
                "condition": "left_subarticular_stenosis",
                "spinal_level": "L1/L2",
                "y_true": 0,
                "p_class_0": 0.85,
                "p_class_1": 0.10,
                "p_class_2": 0.05,
            },
            {
                "patient_id": 101,
                "condition": "spinal_canal_stenosis",
                "spinal_level": "L1/L2",
                "y_true": 0,
                "p_class_0": 0.88,
                "p_class_1": 0.08,
                "p_class_2": 0.04,
            },
        ]
    )
    solution, submission = frames_from_predictions(pred)
    assert list(solution["row_id"]) == [
        "100_spinal_canal_stenosis_l1_l2",
        "100_spinal_canal_stenosis_l2_l3",
        "100_left_neural_foraminal_narrowing_l1_l2",
        "100_left_subarticular_stenosis_l1_l2",
        "101_spinal_canal_stenosis_l1_l2",
    ]
    assert solution.loc[0, "sample_weight"] == 1.0
    assert solution.loc[1, "sample_weight"] == 4.0
    parts = score_prediction_table(pred)
    assert parts["score"] == pytest.approx(
        score(solution, submission, "row_id", ANY_SEVERE_SCALAR_OFFICIAL)
    )
    assert set(parts) >= {
        "score",
        "spinal",
        "foraminal",
        "subarticular",
        "any_severe_spinal",
        "n_rows",
    }


def test_negative_submission_is_rejected():
    solution, submission = _minimal_frames()
    submission.loc[0, "normal_mild"] = -0.1
    with pytest.raises(ParticipantVisibleError):
        score(solution, submission, "row_id", 1.0)


def test_score_parts_matches_overall():
    solution, submission = _minimal_frames(0.7)
    parts = score_parts(solution, submission)
    assert parts["score"] == pytest.approx(score(solution, submission, "row_id", 1.0))


def test_family_labels():
    assert architecture_from_stem("nas_best_convnext3d_3d") == "ConvNeXt"
    assert representation_from_stem("nas_best_convnext3d_3d") == "2.5D-to-3D"
    assert representation_from_stem("nas_best_vit_2d") == "2D"
