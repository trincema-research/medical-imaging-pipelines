"""RSNA 2024 Kaggle evaluation metric (sample-weighted log loss).

The ``score`` function is the competition metric notebook: mean of spinal /
foraminal / subarticular weighted log losses plus ``any_severe_spinal``,
weighted by ``any_severe_scalar`` (official value: 1.0). Lower is better.

This is not the unweighted multi-head CE used during NAS. Local scores still
use ground-truth-centered crops and our 70/15/15 study split — not the hidden
Kaggle test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

TARGET_LEVELS: tuple[str, ...] = ("normal_mild", "moderate", "severe")
SAMPLE_WEIGHTS: tuple[float, ...] = (1.0, 2.0, 4.0)
ANY_SEVERE_SCALAR_OFFICIAL = 1.0
CONDITION_GROUPS: tuple[str, ...] = ("spinal", "foraminal", "subarticular")
ROW_ID_COLUMN = "row_id"


class ParticipantVisibleError(Exception):
    pass


def get_condition(full_location: str) -> str:
    for injury_condition in CONDITION_GROUPS:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f"condition not found in {full_location}")


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    any_severe_scalar: float,
) -> float:
    """Official RSNA 2024 participant-visible metric."""
    target_levels = list(TARGET_LEVELS)
    solution = solution.copy()
    submission = submission.copy()

    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError("All submission values must be numeric")

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError("All submission values must be finite")

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError("All labels must be at least zero")
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError("All predictions must be at least zero")

    solution["study_id"] = solution["row_id"].apply(lambda x: x.split("_")[0])
    solution["location"] = solution["row_id"].apply(lambda x: "_".join(x.split("_")[1:]))
    solution["condition"] = solution["row_id"].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission["study_id"] = solution["study_id"]
    submission["location"] = solution["location"]
    submission["condition"] = solution["condition"]

    condition_losses = []
    condition_weights = []
    for condition in CONDITION_GROUPS:
        condition_indices = solution.loc[solution["condition"] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, "sample_weight"].values,
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    any_severe_spinal_labels = pd.Series(
        solution.loc[solution["condition"] == "spinal"].groupby("study_id")["severe"].max()
    )
    any_severe_spinal_weights = pd.Series(
        solution.loc[solution["condition"] == "spinal"].groupby("study_id")["sample_weight"].max()
    )
    any_severe_spinal_predictions = pd.Series(
        submission.loc[submission["condition"] == "spinal"].groupby("study_id")["severe"].max()
    )
    any_severe_spinal_loss = sklearn.metrics.log_loss(
        y_true=any_severe_spinal_labels,
        y_pred=any_severe_spinal_predictions,
        sample_weight=any_severe_spinal_weights,
        labels=[0, 1],
    )
    condition_losses.append(any_severe_spinal_loss)
    condition_weights.append(any_severe_scalar)
    return float(np.average(condition_losses, weights=condition_weights))


def score_parts(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    row_id_column_name: str = ROW_ID_COLUMN,
    any_severe_scalar: float = ANY_SEVERE_SCALAR_OFFICIAL,
) -> dict[str, float]:
    """Official score plus the four group log losses."""
    target_levels = list(TARGET_LEVELS)
    solution = solution.copy()
    submission = submission.copy()
    overall = score(solution, submission, row_id_column_name, any_severe_scalar)

    solution["study_id"] = solution["row_id"].apply(lambda x: str(x).split("_")[0])
    solution["location"] = solution["row_id"].apply(lambda x: "_".join(str(x).split("_")[1:]))
    solution["condition"] = solution["row_id"].apply(get_condition)
    aligned = submission.set_index(row_id_column_name).loc[solution[row_id_column_name]]
    aligned = aligned.reset_index(drop=True)

    parts: dict[str, float] = {"score": overall}
    for condition in CONDITION_GROUPS:
        idx = solution["condition"] == condition
        parts[condition] = float(
            sklearn.metrics.log_loss(
                y_true=solution.loc[idx, target_levels].values,
                y_pred=aligned.loc[idx, target_levels].values,
                sample_weight=solution.loc[idx, "sample_weight"].values,
            )
        )
    spinal = solution["condition"] == "spinal"
    parts["any_severe_spinal"] = float(
        sklearn.metrics.log_loss(
            y_true=solution.loc[spinal].groupby("study_id")["severe"].max(),
            y_pred=aligned.loc[spinal]
            .assign(study_id=solution.loc[spinal, "study_id"].values)
            .groupby("study_id")["severe"]
            .max(),
            sample_weight=solution.loc[spinal].groupby("study_id")["sample_weight"].max(),
            labels=[0, 1],
        )
    )
    return parts


def level_slug(value: str) -> str:
    text = str(value).strip().lower().replace("/", "_").replace("-", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_")


def prediction_row_id(patient_id: Any, condition: str, spinal_level: str) -> str:
    return f"{int(patient_id)}_{condition}_{level_slug(spinal_level)}"


def _one_hot(class_index: int) -> dict[str, float]:
    row = {name: 0.0 for name in TARGET_LEVELS}
    if 0 <= int(class_index) < len(TARGET_LEVELS):
        row[TARGET_LEVELS[int(class_index)]] = 1.0
    return row


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if not np.isfinite(probs).all():
        raise ParticipantVisibleError("All predictions must be finite")
    total = float(probs.sum())
    if total <= 0:
        return np.full(len(TARGET_LEVELS), 1.0 / len(TARGET_LEVELS))
    return probs / total


def frames_from_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build official ``solution`` / ``submission`` tables from ``predictions_*.csv``."""
    if predictions.empty:
        raise ValueError("predictions is empty")
    required = {"patient_id", "condition", "spinal_level", "y_true"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    solution_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        y_true = int(row.y_true)
        if y_true < 0:
            continue
        condition = str(row.condition)
        row_id = prediction_row_id(row.patient_id, condition, row.spinal_level)
        raw = [
            getattr(row, "p_class_0", np.nan),
            getattr(row, "p_class_1", np.nan),
            getattr(row, "p_class_2", np.nan),
        ]
        if any(v == "" or pd.isna(v) for v in raw):
            continue
        raw_arr = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw_arr).all():
            continue
        probs = _normalize_probs(raw_arr)
        labels = _one_hot(y_true)
        solution_rows.append(
            {
                ROW_ID_COLUMN: row_id,
                **labels,
                "sample_weight": float(SAMPLE_WEIGHTS[y_true]),
            }
        )
        submission_rows.append(
            {
                ROW_ID_COLUMN: row_id,
                TARGET_LEVELS[0]: float(probs[0]),
                TARGET_LEVELS[1]: float(probs[1]),
                TARGET_LEVELS[2]: float(probs[2]),
            }
        )
    solution = pd.DataFrame(solution_rows)
    submission = pd.DataFrame(submission_rows)
    if solution.empty:
        raise ValueError("no labeled prediction rows")
    return solution, submission


def score_prediction_table(
    predictions: pd.DataFrame,
    *,
    any_severe_scalar: float = ANY_SEVERE_SCALAR_OFFICIAL,
) -> dict[str, float]:
    solution, submission = frames_from_predictions(predictions)
    parts = score_parts(
        solution,
        submission,
        row_id_column_name=ROW_ID_COLUMN,
        any_severe_scalar=any_severe_scalar,
    )
    parts["n_rows"] = float(len(solution))
    parts["n_studies"] = float(solution["row_id"].map(lambda x: str(x).split("_")[0]).nunique())
    return parts
