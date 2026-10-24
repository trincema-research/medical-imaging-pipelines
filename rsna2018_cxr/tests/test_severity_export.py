from pathlib import Path

import pytest

pytest.importorskip("torch")

from rsna2018_cxr.severity.export_weights import KEEP_KEYS, export_one
from rsna2018_cxr.severity.retrain import parse_args as parse_retrain


def test_retrain_parses_severity_shortcut():
    args = parse_retrain(["--severity", "severity_2", "--config-json", "missing.json"])
    assert args.severity == "severity_2"
    assert args.positive_class is None


def test_export_one_drops_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "full.pt"
    dest = tmp_path / "slim.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state": {"w": torch.tensor([1.0])},
            "optimizer_state": {"state": {"too": "big"}},
            "best_val_loss": 0.2,
            "config": {"variant": "vit_b_16"},
            "class_to_idx": {"normal": 0, "severity_1_low_mild": 1},
            "train_acc": 0.9,
            "val_acc": 0.8,
            "positive_class": "severity_1_low_mild",
            "split_csv": "val_test_images.csv",
        },
        source,
    )
    row = export_one(source, dest)
    loaded = torch.load(dest, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in loaded
    assert set(KEEP_KEYS).issuperset(loaded)
    assert row["positive_class"] == "severity_1_low_mild"
    assert Path(dest).is_file()
