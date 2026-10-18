"""Prediction container shared by every lung backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LungPrediction:
    left_lung: np.ndarray
    right_lung: np.ndarray
    merged_lungs: np.ndarray
    probability_map: np.ndarray | None
    runtime_ms: float
    model_name: str
    model_version: str
    metadata: dict = field(default_factory=dict)
