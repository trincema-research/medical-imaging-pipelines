"""Severity-stratified Dice summary.

``delta_dice`` is Normal minus that class. Positive means the class is worse
than Normal. Severity labels here are radiographic burden strata.
"""

from __future__ import annotations

import pandas as pd


def summarize_by_severity(frame: pd.DataFrame, dice_column: str = "dice") -> pd.DataFrame:
    if dice_column not in frame.columns or "severity_class" not in frame.columns:
        raise ValueError("Need severity_class and a dice column.")
    grouped = (
        frame.groupby("severity_class", sort=False)[dice_column]
        .agg(n="count", dice="mean")
        .reset_index()
    )
    normal = grouped.loc[grouped["severity_class"] == "Normal", "dice"]
    normal_dice = float(normal.iloc[0]) if not normal.empty else float("nan")
    grouped["delta_dice_vs_normal"] = normal_dice - grouped["dice"]
    grouped.loc[grouped["severity_class"] == "Normal", "delta_dice_vs_normal"] = 0.0
    return grouped
