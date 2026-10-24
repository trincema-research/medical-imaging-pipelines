"""Severity class names and the three published specialist checkpoints."""

from __future__ import annotations

from pathlib import Path

YEAR_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = YEAR_ROOT / "data" / "severity_models"

NORMAL_CLASS = "normal"
SEVERITY_POSITIVE_CLASSES = {
    "severity_1": "severity_1_low_mild",
    "severity_2": "severity_2_moderate",
    "severity_3": "severity_3_severe",
}

# Three fine-tunes of the S1 specialist (normal vs severity_1_low_mild).
# Full training dumps (~1 GB, include optimizer) live on the GitHub Release;
# these names are the inference-only files the download CLI writes.
SPECIALIST_CHECKPOINTS = (
    {
        "name": "specialist_s1_val0.9470.pt",
        "positive_class": "severity_1_low_mild",
        "val_acc": 0.9470046082949308,
        "train_acc": 0.9875576036866359,
        "epoch": 33,
        "source_name": "best_model_val0.9470_train0.9876.pt",
    },
    {
        "name": "specialist_s1_val0.9433.pt",
        "positive_class": "severity_1_low_mild",
        "val_acc": 0.9433179723502304,
        "train_acc": 0.9932027649769585,
        "epoch": 37,
        "source_name": "best_model_val0.9433_train0.9932.pt",
    },
    {
        "name": "specialist_s1_val0.9415.pt",
        "positive_class": "severity_1_low_mild",
        "val_acc": 0.9414746543778801,
        "train_acc": 0.998963133640553,
        "epoch": 48,
        "source_name": "best_model_val0.9415_train0.9990.pt",
    },
)

RELEASE_TAG = "rsna2018-severity-s1-specialists"
GITHUB_REPO = "trincema-research/medical-imaging-pipelines"
