from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_condition_specific_results import (
    ARCH_ORDER,
    CONDITION_ORDER,
    DEFAULT_MEANS,
    REP_ORDER,
    compute_bottlenecks,
    generate_condition_specific_figures,
    incomplete_groups,
    load_and_validate_data,
)


def _wide_rows() -> pd.DataFrame:
    f1 = {
        ("ViT", "2D"): (0.55, 0.40, 0.48, 0.46, 0.47),
        ("MaxViT", "2D"): (0.58, 0.50, 0.52, 0.51, 0.53),
        ("ConvNeXt", "2D"): (0.62, 0.54, 0.56, 0.55, 0.57),
        ("EfficientNet", "2D"): (0.57, 0.49, 0.51, 0.50, 0.52),
        ("ViT", "2.5D-to-3D"): (0.38, 0.30, 0.32, 0.31, 0.29),
        ("MaxViT", "2.5D-to-3D"): (0.50, 0.44, 0.46, 0.45, 0.47),
        ("ConvNeXt", "2.5D-to-3D"): (0.34, 0.28, 0.26, 0.25, 0.27),
        ("EfficientNet", "2.5D-to-3D"): (0.36, 0.31, 0.33, 0.32, 0.24),
    }
    rows: list[dict] = []
    for arch in ARCH_ORDER:
        for rep in REP_ORDER:
            for cond, value in zip(CONDITION_ORDER, f1[(arch, rep)]):
                n_runs = 4 if (arch, rep, cond) in {("ConvNeXt", "2.5D-to-3D", "RFN"), ("EfficientNet", "2.5D-to-3D", "RSS")} else 5
                rows.append(
                    {
                        "architecture": arch,
                        "representation": rep,
                        "condition": cond,
                        "f1_mean": value,
                        "f1_std": 0.01,
                        "qwk_mean": 0.5 if rep == "2D" else 0.2,
                        "qwk_std": 0.01,
                        "omae_mean": 0.2,
                        "omae_std": 0.01,
                        "ser_mean": 0.02,
                        "ser_std": 0.01,
                        "n_runs": n_runs,
                    }
                )
    return pd.DataFrame(rows)


def test_load_validates_40_rows(tmp_path: Path):
    path = tmp_path / "condspec_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    df = load_and_validate_data(path)
    assert len(df) == 40


def test_bottleneck_from_data(tmp_path: Path):
    path = tmp_path / "condspec_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    summary = compute_bottlenecks(load_and_validate_data(path))
    conv = summary[(summary["arch"] == "ConvNeXt") & (summary["rep"] == "2.5D-to-3D")].iloc[0]
    assert conv["worst_condition"] == "LSS"
    assert conv["best_condition"] == "SCS"


def test_incomplete_groups_preserved(tmp_path: Path):
    path = tmp_path / "condspec_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    missing = incomplete_groups(load_and_validate_data(path))
    keys = {(row["arch"], row["rep"], row["condition"], row["n_runs"]) for row in missing}
    assert ("ConvNeXt", "2.5D-to-3D", "RFN", 4) in keys
    assert ("EfficientNet", "2.5D-to-3D", "RSS", 4) in keys


def test_missing_column_stops(tmp_path: Path):
    df = _wide_rows().drop(columns=["f1_mean"])
    path = tmp_path / "condspec_condition_means.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="f1"):
        load_and_validate_data(path)


@pytest.mark.skipif(not DEFAULT_MEANS.is_file(), reason="harvested condition-specific means not committed")
def test_real_harvest_loads_40_rows():
    df = load_and_validate_data(DEFAULT_MEANS)
    assert len(df) == 40
    missing = incomplete_groups(df)
    assert { (row["arch"], row["rep"], row["condition"]) for row in missing } == {
        ("ConvNeXt", "2.5D-to-3D", "RFN"),
        ("EfficientNet", "2.5D-to-3D", "RSS"),
    }


def test_generate_writes_pngs(tmp_path: Path):
    path = tmp_path / "condspec_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_condition_specific_figures(means_path=path, output_dir=out)
    assert paths["profile_png"].is_file()
    assert paths["radar_png"].is_file()
    assert paths["bottleneck_png"].is_file()
    assert paths["uniformity_png"].is_file()
    assert paths["heatmap_png"].is_file()
    source = pd.read_csv(paths["plot_source"])
    assert len(source) == 40
    bottlenecks = pd.read_csv(paths["bottleneck_source"])
    assert len(bottlenecks) == 8
