from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_ordinal_severity_results import (
    ARCH_ORDER,
    CONDITION_ORDER,
    DEFAULT_MEANS,
    REP_ORDER,
    compute_architecture_means,
    generate_ordinal_severity_figures,
    load_and_validate_data,
)


def _wide_rows() -> pd.DataFrame:
    oa = {
        ("ViT", "2D"): (0.90, 0.77, 0.85, 0.82, 0.83),
        ("MaxViT", "2D"): (0.91, 0.82, 0.86, 0.81, 0.83),
        ("ConvNeXt", "2D"): (0.92, 0.84, 0.87, 0.83, 0.84),
        ("EfficientNet", "2D"): (0.90, 0.81, 0.86, 0.82, 0.83),
        ("ViT", "2.5D-to-3D"): (0.86, 0.74, 0.80, 0.75, 0.76),
        ("MaxViT", "2.5D-to-3D"): (0.88, 0.79, 0.83, 0.80, 0.81),
        ("ConvNeXt", "2.5D-to-3D"): (0.85, 0.73, 0.78, 0.74, 0.75),
        ("EfficientNet", "2.5D-to-3D"): (0.86, 0.75, 0.80, 0.77, 0.79),
    }
    rows: list[dict] = []
    for arch in ARCH_ORDER:
        for rep in REP_ORDER:
            for cond, value in zip(CONDITION_ORDER, oa[(arch, rep)]):
                rows.append(
                    {
                        "architecture": arch,
                        "representation": rep,
                        "condition": cond,
                        "oa_mean": value,
                        "oa_std": 0.01,
                        "omae_mean": 0.20 if rep == "2D" else 0.28,
                        "omae_std": 0.01,
                        "qwk_mean": 0.70 if rep == "2D" else 0.25,
                        "qwk_std": 0.01,
                        "ser_mean": 0.01 if rep == "2D" else 0.05,
                        "ser_std": 0.01,
                        "n_runs": 5,
                    }
                )
    return pd.DataFrame(rows)


def _long_rows() -> pd.DataFrame:
    wide = _wide_rows()
    rows: list[dict] = []
    for rec in wide.to_dict(orient="records"):
        for metric in ("oa_overall", "omae_overall", "qwk_overall", "ser_overall"):
            dest = metric.replace("_overall", "")
            rows.append(
                {
                    "architecture": rec["architecture"],
                    "representation": rec["representation"],
                    "condition": rec["condition"],
                    "split": "validation",
                    "metric": metric,
                    "mean": rec[f"{dest}_mean"],
                    "std": rec[f"{dest}_std"],
                    "n_runs": rec["n_runs"],
                }
            )
    return pd.DataFrame(rows)


def test_load_wide_validates_40_rows(tmp_path: Path):
    path = tmp_path / "ordinal_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    df = load_and_validate_data(path)
    assert len(df) == 40
    assert set(df["condition"]) == set(CONDITION_ORDER)


def test_load_long_harvested_shape(tmp_path: Path):
    path = tmp_path / "condition_specific_by_condition.csv"
    _long_rows().to_csv(path, index=False)
    df = load_and_validate_data(path)
    assert len(df) == 40


def test_architecture_means_has_8_rows(tmp_path: Path):
    path = tmp_path / "ordinal_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    arch = compute_architecture_means(load_and_validate_data(path))
    assert len(arch) == 8


def test_missing_column_stops(tmp_path: Path):
    df = _wide_rows().drop(columns=["qwk_mean", "qwk_std"])
    path = tmp_path / "ordinal_condition_means.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="qwk"):
        load_and_validate_data(path)


@pytest.mark.skipif(not DEFAULT_MEANS.is_file(), reason="harvested ordinal means not committed")
def test_real_harvest_loads_40_rows():
    df = load_and_validate_data(DEFAULT_MEANS)
    assert len(df) == 40
    arch = compute_architecture_means(df)
    assert len(arch) == 8


def test_generate_writes_pngs(tmp_path: Path):
    path = tmp_path / "ordinal_condition_means.csv"
    _wide_rows().to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_ordinal_severity_figures(means_path=path, output_dir=out)
    assert paths["tradeoff_png"].is_file()
    assert paths["heatmaps_png"].is_file()
    assert paths["gain_png"].is_file()
    assert paths["ser_png"].is_file()
    source = pd.read_csv(paths["plot_source_condition"])
    assert len(source) == 40
    arch = pd.read_csv(paths["plot_source_architecture"])
    assert len(arch) == 8
