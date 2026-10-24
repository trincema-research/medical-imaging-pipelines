import pytest

from rsna2024_lumbar.coordinate_prediction.constants import BACKBONE_CONVNEXT_TINY, LUMBAR_LEVELS
from rsna2024_lumbar.coordinate_prediction.dataset import list_condition_samples
from rsna2024_lumbar.tests.test_coordinate_prediction_densify import _axial_stack

torch = pytest.importorskip("torch")


def test_regressor_forward_shapes():
    from rsna2024_lumbar.coordinate_prediction.models import build_regressor, landmark_loss

    model = build_regressor(BACKBONE_CONVNEXT_TINY, pretrained=False)
    images = torch.zeros(2, 3, 64, 64)
    xy, vis = model(images)
    assert xy.shape == (2, len(LUMBAR_LEVELS), 2)
    assert vis.shape == (2, len(LUMBAR_LEVELS))
    target = torch.zeros_like(xy)
    visible = torch.ones(2, len(LUMBAR_LEVELS))
    loss, parts = landmark_loss(xy, vis, target, visible)
    assert loss.ndim == 0
    assert "l1" in parts


def test_coordinate_r2_perfect_and_mean():
    from rsna2024_lumbar.coordinate_prediction.models import coordinate_r2

    target = torch.tensor([[[0.1, 0.2], [0.8, 0.9]], [[0.3, 0.4], [0.6, 0.7]]], dtype=torch.float32)
    visible = torch.ones(2, 2)
    perfect = coordinate_r2(target, target, visible)
    assert abs(perfect["r2"] - 1.0) < 1e-6
    assert abs(perfect["r2_l1_l2"] - 1.0) < 1e-6
    assert abs(perfect["r2_l2_l3"] - 1.0) < 1e-6
    mean = target.mean(dim=(0, 1), keepdim=True).expand_as(target)
    baseline = coordinate_r2(mean, target, visible)
    assert abs(baseline["r2"]) < 1e-5


def test_coordinate_metrics_overall_and_per_level():
    from rsna2024_lumbar.coordinate_prediction.models import LEVEL_METRIC_KEYS, coordinate_metrics

    pred = torch.tensor([[[0.10, 0.20], [0.50, 0.50]], [[0.12, 0.20], [0.80, 0.80]]], dtype=torch.float32)
    target = torch.tensor([[[0.10, 0.20], [0.50, 0.50]], [[0.10, 0.20], [0.80, 0.80]]], dtype=torch.float32)
    visible = torch.ones(2, 2)
    out = coordinate_metrics(pred, target, visible, width=100, height=100)
    assert out["n"] == 2
    assert out["n_visible"] == 4
    assert abs(out["mae_x_px"] - 0.5) < 1e-5
    assert abs(out["mae_y_px"]) < 1e-5
    assert set(out["levels"]) == {"l1_l2", "l2_l3"}
    assert out["levels"]["l1_l2"]["n_visible"] == 2
    assert abs(out["levels"]["l1_l2"]["mae_px"] - 1.0) < 1e-5
    assert out["levels"]["l2_l3"]["mae_px"] == 0.0
    assert out["levels"]["l2_l3"]["pck5"] == 1.0
    assert out["pck10"] == out["pck5"] or out["pck10"] >= out["pck5"]
    for key in LEVEL_METRIC_KEYS:
        assert f"{key}_l1_l2" in out
        assert f"{key}_l2_l3" in out


def test_coordinate_metrics_hidden_level_has_zero_visible():
    from rsna2024_lumbar.coordinate_prediction.models import coordinate_metrics

    pred = torch.zeros(3, 2, 2)
    target = torch.zeros(3, 2, 2)
    visible = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    out = coordinate_metrics(pred, target, visible, width=64, height=64)
    assert out["levels"]["l1_l2"]["n_visible"] == 3
    assert out["levels"]["l2_l3"]["n_visible"] == 0
    assert out["n_visible_l2_l3"] == 0


def test_list_condition_samples_reads_axial_fixture(tmp_path):
    raw = _axial_stack(tmp_path / "raw")
    samples = list_condition_samples(raw, "left_subarticular_stenosis")
    assert len(samples) == 1
    assert samples[0].instance_number == 2
    assert float(samples[0].visible.sum()) == 1.0
    assert abs(float(samples[0].xy01[2, 0]) - 0.5) < 1e-5
