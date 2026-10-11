import pytest

from rsna2024_lumbar.landmarks.constants import BACKBONE_CONVNEXT_TINY, LUMBAR_LEVELS
from rsna2024_lumbar.landmarks.dataset import list_condition_samples
from rsna2024_lumbar.tests.test_landmark_densify import _axial_stack

torch = pytest.importorskip("torch")


def test_regressor_forward_shapes():
    from rsna2024_lumbar.landmarks.models import build_regressor, landmark_loss

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


def test_list_condition_samples_reads_axial_fixture(tmp_path):
    raw = _axial_stack(tmp_path / "raw")
    samples = list_condition_samples(raw, "left_subarticular_stenosis")
    assert len(samples) == 1
    assert samples[0].instance_number == 2
    assert float(samples[0].visible.sum()) == 1.0
    assert abs(float(samples[0].xy01[2, 0]) - 0.5) < 1e-5
