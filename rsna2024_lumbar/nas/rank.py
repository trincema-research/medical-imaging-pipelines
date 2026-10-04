"""Rank NAS trials and pick best configs per condition or shared across conditions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from rsna2024_lumbar.nas.results import TrialRecord
from rsna2024_lumbar.preprocessing.constants import CONDITIONS

# Default: val accuracy at the early-stopping / best epoch (best_epoch_metrics.json).
METRIC_VAL_ACC = "val_acc"
METRIC_MAX_VAL_ACC = "max_val_acc"
METRIC_FINAL_VAL_ACC = "final_val_acc"
METRIC_VAL_F1 = "val_f1_macro_levels"
METRIC_TEST_F1 = "test_f1_macro_levels"
METRIC_VAL_LOSS = "val_loss"

DEFAULT_RANK_METRIC = METRIC_VAL_ACC

RANK_METRICS: tuple[str, ...] = (
    METRIC_VAL_ACC,
    METRIC_MAX_VAL_ACC,
    METRIC_FINAL_VAL_ACC,
    METRIC_VAL_F1,
    METRIC_TEST_F1,
    METRIC_VAL_LOSS,
)

METRIC_HELP: dict[str, str] = {
    METRIC_VAL_ACC: "Validation accuracy at best epoch (best_epoch_metrics.json; falls back to max_val_acc).",
    METRIC_MAX_VAL_ACC: "Peak validation accuracy over all epochs (result.json).",
    METRIC_FINAL_VAL_ACC: "Validation accuracy at the last epoch (result.json).",
    METRIC_VAL_F1: "Macro F1 over levels at best epoch (result.json).",
    METRIC_TEST_F1: "Test macro F1 at best epoch (result.json).",
    METRIC_VAL_LOSS: "Validation loss at best epoch (best_epoch_metrics.json).",
}


def metric_value(record: TrialRecord, metric: str) -> float:
    if metric == METRIC_VAL_ACC:
        if record.val_acc is not None:
            return record.val_acc
        return record.max_val_acc
    if metric == METRIC_VAL_F1:
        return record.val_f1_macro_levels
    if metric == METRIC_VAL_LOSS:
        if record.val_loss is None:
            raise ValueError(f"Missing val_loss for {record.result_path} (no best_epoch_metrics.json).")
        return record.val_loss
    if metric == METRIC_MAX_VAL_ACC:
        return record.max_val_acc
    if metric == METRIC_FINAL_VAL_ACC:
        return record.final_val_acc
    if metric == METRIC_TEST_F1:
        return record.test_f1_macro_levels
    raise ValueError(f"Unknown metric {metric!r}. Choose {RANK_METRICS}.")


def higher_is_better(metric: str) -> bool:
    return metric != METRIC_VAL_LOSS


def rank_key(metric: str) -> Callable[[TrialRecord], float]:
    reverse = higher_is_better(metric)

    def key(record: TrialRecord) -> float:
        value = metric_value(record, metric)
        return value if reverse else -value

    return key


def sort_trials(records: list[TrialRecord], metric: str) -> list[TrialRecord]:
    return sorted(records, key=rank_key(metric), reverse=True)


def best_per_condition(
    records: list[TrialRecord],
    metric: str,
) -> dict[str, TrialRecord]:
    best: dict[str, TrialRecord] = {}
    for condition in CONDITIONS:
        subset = [r for r in records if r.condition == condition]
        if not subset:
            continue
        best[condition] = sort_trials(subset, metric)[0]
    return best


@dataclass(frozen=True)
class SharedConfigPick:
    hyperparam_key: tuple
    mean_score: float
    per_condition: dict[str, TrialRecord]

    @property
    def trial_count(self) -> int:
        return len(self.per_condition)


def best_shared_hyperparams(
    records: list[TrialRecord],
    metric: str,
    *,
    require_all_conditions: bool = True,
) -> SharedConfigPick | None:
    """
    Pick one hyperparameter setting that appears in every condition and maximizes mean metric.

    Useful when you want a single config to reuse across all five tasks.
    """
    by_key: dict[tuple, dict[str, TrialRecord]] = {}
    for record in records:
        by_key.setdefault(record.hyperparam_key(), {})[record.condition] = record

    required = set(CONDITIONS) if require_all_conditions else None
    candidates: list[SharedConfigPick] = []
    for key, per_cond in by_key.items():
        if required and set(per_cond) != required:
            continue
        scores = [metric_value(r, metric) for r in per_cond.values()]
        candidates.append(
            SharedConfigPick(
                hyperparam_key=key,
                mean_score=float(mean(scores)),
                per_condition=dict(per_cond),
            )
        )
    if not candidates:
        return None
    reverse = higher_is_better(metric)
    return sorted(candidates, key=lambda c: c.mean_score, reverse=reverse)[0]


def top_trials(records: list[TrialRecord], metric: str, *, limit: int) -> list[TrialRecord]:
    return sort_trials(records, metric)[:limit]
