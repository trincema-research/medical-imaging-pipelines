from rsna2024_lumbar.preprocessing.constants import CROP_POLICY_CENTERED, CROP_POLICY_EXTEND50
from rsna2024_lumbar.preprocessing.crops import crop_bounds, crop_hw, crop_settings_for
from rsna2024_lumbar.tests.helpers import DOT_XY, SLICE


def test_centered_is_12p5_square():
    s = crop_settings_for("spinal_canal_stenosis", CROP_POLICY_CENTERED)
    h, w = crop_hw(SLICE, SLICE, s)
    assert (h, w) == (16, 16)
    x0, y0, x1, y1 = crop_bounds(*DOT_XY, h, w, SLICE, SLICE, s)
    assert (x1 - x0, y1 - y0) == (16, 16)
    assert x0 < DOT_XY[0] < x1
    assert y0 < DOT_XY[1] < y1


def test_extend50_scs_grows_left():
    s = crop_settings_for("spinal_canal_stenosis", CROP_POLICY_EXTEND50)
    h, w = crop_hw(SLICE, SLICE, s)
    assert (h, w) == (16, 24)
    x0, y0, x1, y1 = crop_bounds(*DOT_XY, h, w, SLICE, SLICE, s)
    centered = crop_settings_for("spinal_canal_stenosis", CROP_POLICY_CENTERED)
    cx0, _, cx1, _ = crop_bounds(*DOT_XY, 16, 16, SLICE, SLICE, centered)
    assert x1 == cx1
    assert x0 < cx0


def test_extend50_lss_grows_right():
    s = crop_settings_for("left_subarticular_stenosis", CROP_POLICY_EXTEND50)
    h, w = crop_hw(SLICE, SLICE, s)
    x0, _, x1, _ = crop_bounds(*DOT_XY, h, w, SLICE, SLICE, s)
    centered = crop_settings_for("left_subarticular_stenosis", CROP_POLICY_CENTERED)
    cx0, _, cx1, _ = crop_bounds(*DOT_XY, 16, 16, SLICE, SLICE, centered)
    assert x0 == cx0
    assert x1 > cx1
