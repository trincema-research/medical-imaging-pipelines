"""
Performance pipeline (``perf_pipeline``) — stage 3 after ``data/`` and ``nas/``.

**Name:** ``perf_pipeline`` (performance / confirmation pipeline).

**Purpose:** Take NAS-selected best hyperparameters per condition, retrain under a
fixed protocol (default 5 split-seed repeats), and publish comparable **train / val /
test** metrics—including severity scores (OA, O-MAE, QWK, SER)—in ``data/perf_pipeline/results/``.

Artifacts: ``data/`` (raw inputs) → ``nas/`` (search & best configs) → ``perf_pipeline/`` (confirmed metrics).
"""

from rsna2024_lumbar.perf_pipeline.severity_metrics import (
    compute_severity_metrics,
    severe_error_rate,
)

__all__ = [
    "compute_severity_metrics",
    "severe_error_rate",
]
