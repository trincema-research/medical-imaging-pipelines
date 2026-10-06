"""
Best-config runs (``best_config_runs``) — stage 3 after ``data/`` and ``nas/``.

Retrain each ``nas_best_*.json`` entry under a fixed protocol (default 5 split-seed
repeats) and write **train / val / test** metrics—including severity (OA, O-MAE, QWK, SER)—
to ``data/best_config_runs/results/``.

Flow: ``data/`` → ``nas/`` (search) → ``best_config_runs/`` (confirmed CSV metrics).
"""

from rsna2024_lumbar.best_config_runs.severity_metrics import (
    compute_severity_metrics,
    severe_error_rate,
)

__all__ = [
    "compute_severity_metrics",
    "severe_error_rate",
]
