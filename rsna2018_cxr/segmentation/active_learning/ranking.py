"""Severity-balanced review ranking.

Globally uncertain images tend to be severe or AP outliers. Selection is
therefore top-k inside each burden class.
"""

from __future__ import annotations

import pandas as pd


def active_learning_score(mean_pairwise_dice: float, anatomy_warning_count: int, boundary_uncertainty: float = 0.0) -> float:
    disagreement = 1.0 - mean_pairwise_dice if mean_pairwise_dice == mean_pairwise_dice else 1.0
    return float(disagreement + 0.05 * anatomy_warning_count + boundary_uncertainty)


def rank_review_queue(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Highest ``active_learning_score`` within each severity class, review cases only."""
    if per_class < 1:
        raise ValueError("per_class must be >= 1")
    pending = frame[frame["manual_review_required"].fillna(False).astype(bool)].copy()
    if "active_learning_score" not in pending.columns:
        raise ValueError("active_learning_score is required")
    pending = pending.sort_values(["severity_class", "active_learning_score"], ascending=[True, False])
    chosen = pending.groupby("severity_class", sort=False).head(per_class)
    return chosen.reset_index(drop=True)
