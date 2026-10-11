from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_spinal_level_results import (
    ARCH_ORDER,
    DEFAULT_MEANS,
    LEVEL_ORDER,
    REP_ORDER,
    compute_bottlenecks,
    compute_representation_level_means,
    generate_spinal_level_figures,
    load_and_validate_data,
)


def _means_rows() -> pd.DataFrame:
    rows: list[dict] = []
    f1 = {
        ("ViT", "2D"): (0.47, 0.55, 0.59, 0.61, 0.52),
        ("MaxViT", "2D"): (0.39, 0.50, 0.66, 0.70, 0.57),
        ("ConvNeXt", "2D"): (0.48, 0.58, 0.65, 0.68, 0.58),
        ("EfficientNet", "2D"): (0.39, 0.53, 0.63, 0.68, 0.59),
        ("ViT", "2.5D-to-3D"): (0.33, 0.33, 0.36, 0.40, 0.30),
        ("MaxViT", "2.5D-to-3D"): (0.41, 0.46, 0.51, 0.61, 0.51),
        ("ConvNeXt", "2.5D-to-3D"): (0.32, 0.31, 0.28, 0.24, 0.28),
        ("EfficientNet", "2.5D-to-3D"): (0.33, 0.35, 0.38, 0.40, 0.34),
    }
    for arch in ARCH_ORDER:
        for rep in REP_ORDER:
            for cond in ("SCS", "LFN", "RFN", "LSS", "RSS"):
                for level, value in zip(LEVEL_ORDER, f1[(arch, rep)]):
                    rows.append(
                        {
                            "architecture": arch,
                            "representation": rep,
                            "condition": cond,
                            "split": "validation",
                            "spinal_level": level.replace("_", "/"),
                            "accuracy_mean": 0.8,
                            "accuracy_std": 0.01,
                            "precision_macro_mean": 0.4,
                            "precision_macro_std": 0.01,
                            "recall_macro_mean": 0.4,
                            "recall_macro_std": 0.01,
                            "f1_macro_mean": value,
                            "f1_macro_std": 0.01,
                            "oa_mean": 0.8,
                            "oa_std": 0.01,
                            "omae_mean": 0.2,
                            "omae_std": 0.01,
                            "qwk_mean": 0.5 if rep == "2D" else 0.15,
                            "qwk_std": 0.01,
                            "ser_mean": 0.02,
                            "ser_std": 0.01,
                        }
                    )
    return pd.DataFrame(rows)


def test_load_validates_40_rows(tmp_path: Path):
    path = tmp_path / "spinal_level_means.csv"
    _means_rows().to_csv(path, index=False)
    df = load_and_validate_data(path)
    assert len(df) == 40
    assert set(df["level"]) == set(LEVEL_ORDER)


def test_bottleneck_from_data(tmp_path: Path):
    path = tmp_path / "spinal_level_means.csv"
    _means_rows().to_csv(path, index=False)
    df = load_and_validate_data(path)
    bottlenecks = compute_bottlenecks(df)
    conv = bottlenecks[(bottlenecks["arch"] == "ConvNeXt") & (bottlenecks["rep"] == "2.5D-to-3D")].iloc[0]
    assert conv["worst_level"] == "L4_L5"
    assert conv["best_level"] == "L1_L2"
    assert conv["spread"] == pytest.approx(0.08)


def test_overlay_has_10_rows(tmp_path: Path):
    path = tmp_path / "spinal_level_means.csv"
    _means_rows().to_csv(path, index=False)
    overlay = compute_representation_level_means(load_and_validate_data(path))
    assert len(overlay) == 10


def test_missing_column_stops(tmp_path: Path):
    df = _means_rows().drop(columns=["f1_macro_mean"])
    path = tmp_path / "spinal_level_means.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_and_validate_data(path)


@pytest.mark.skipif(not DEFAULT_MEANS.is_file(), reason="harvested spinal means not committed")
def test_real_harvest_loads_40_rows():
    df = load_and_validate_data(DEFAULT_MEANS)
    assert len(df) == 40
    assert set(df["arch"]) == set(ARCH_ORDER)


def test_generate_writes_pngs(tmp_path: Path):
    path = tmp_path / "spinal_level_means.csv"
    _means_rows().to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_spinal_level_figures(means_path=path, output_dir=out)
    assert paths["bottleneck_png"].is_file()
    assert paths["profile_png"].is_file()
    assert paths["overlay_png"].is_file()
    assert paths["heatmap_2d_png"].is_file()
    assert paths["heatmap_25d_png"].is_file()
    source = pd.read_csv(paths["plot_source"])
    assert len(source) == 40
