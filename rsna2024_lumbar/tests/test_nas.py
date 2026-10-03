"""NAS search-space expansion. CPU only — no torch, no GPU."""

from __future__ import annotations

import pytest

from rsna2024_lumbar.nas.cli import main, parse_args
from rsna2024_lumbar.nas.families import FAMILIES, FAMILY_SPECS, CONFIGS, require_family
from rsna2024_lumbar.nas.search_space import (
    build_trial_configs,
    expand_family,
    load_search_space,
)
from rsna2024_lumbar.preprocessing.constants import (
    CROP_POLICY_CENTERED,
    CROP_POLICY_EXTEND50,
)

EXPECTED_TRIALS = {
    "vit": 2304,
    "maxvit": 1152,
    "convnext": 144,
    "convnext3d": 384,
    "efficientnet": 24,
}

EXPECTED_VARIANTS = {
    "vit": ["vit_b_16", "vit_l_16", "vit_b_32", "vit_l_32"],
    "maxvit": ["maxvit_tiny_tf_224", "maxvit_small_tf_224"],
    "convnext": ["convnext_small"],
    "convnext3d": ["cnn3d_s", "cnn3d_b"],
    "efficientnet": ["efficientnet_v2_s"],
}


@pytest.mark.parametrize("family,n_trials", EXPECTED_TRIALS.items())
def test_family_trial_count(family, n_trials):
    spec, _space, trials = expand_family(family)
    assert spec.config_file.is_file()
    assert len(trials) == n_trials
    assert {trial.variant for trial in trials} == set(EXPECTED_VARIANTS[family])
    assert all(trial.model_type == spec.model_type for trial in trials)
    assert all(trial.crop_policy == CROP_POLICY_CENTERED for trial in trials)
    assert all((trial.image_width, trial.image_height) == (64, 64) for trial in trials)


def test_output_bases_are_isolated():
    bases = [FAMILY_SPECS[name].output_base for name in FAMILIES]
    assert len(bases) == len(set(bases))
    assert all(base.startswith("runs/") for base in bases)


def test_legacy_convnext_24_grid_still_expands():
    spec = FAMILY_SPECS["convnext"]
    space = load_search_space(CONFIGS / "convnext_nas_search_space.json")
    trials = build_trial_configs(space, spec)
    assert len(trials) == 24


def test_extend50_uses_policy_png_size():
    _spec, _space, trials = expand_family("convnext", crop_policy=CROP_POLICY_EXTEND50)
    assert trials[0].crop_policy == CROP_POLICY_EXTEND50
    assert (trials[0].image_width, trials[0].image_height) == (96, 64)


def test_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown NAS family"):
        require_family("resnet")


def test_rejects_unknown_condition():
    with pytest.raises(KeyError, match="Unknown condition"):
        expand_family("vit", condition="not_a_condition")


def test_cli_list_vit(capsys):
    main(["--family", "vit", "--list", "--limit", "2"])
    out = capsys.readouterr().out
    assert "Family:      vit" in out
    assert "Output base: runs/lumbar_nas" in out
    assert "Trials:      2304" in out
    assert "64x64" in out
    assert "vit_b_16" in out
    assert "... 2302 more" in out


def test_cli_rejects_bad_family():
    with pytest.raises(SystemExit):
        parse_args(["--family", "resnet"])


def test_cli_rejects_negative_limit():
    with pytest.raises(SystemExit, match="limit"):
        main(["--family", "convnext", "--limit", "-1"])
