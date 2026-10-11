from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_patient_level_results import (
    generate_patient_level_figures,
    load_data,
    validate_plot_scope,
    validate_profile_sums,
    validate_repeat_summary_consistency,
    warn_repeat_counts,
)


def _summary_row(arch: str, rep: str, *, n: int = 5, exact: float = 0.8, omae: float = 0.2) -> dict:
    r0, r1, r2 = 0.5, 0.4, 0.1
    return {
        "architecture": arch,
        "representation": rep,
        "split": "validation",
        "n_repeats": n,
        "patient_macro_exact_rate_mean": exact,
        "patient_macro_exact_rate_std": 0.01,
        "patient_macro_omae_mean": omae,
        "patient_macro_omae_std": 0.02,
        "patient_exact_profile_rate_mean": r0,
        "patient_exact_profile_rate_std": 0.01,
        "patient_adjacent_only_rate_mean": r1,
        "patient_adjacent_only_rate_std": 0.01,
        "patient_severe_error_rate_mean": r2,
        "patient_severe_error_rate_std": 0.01,
    }


def _repeat_rows(arch: str, rep: str, *, n: int = 5, exact: float = 0.8, omae: float = 0.2) -> list[dict]:
    # Constant values → mean matches summary, std is 0 after ddof=1? No, constant → std 0.
    # Summary uses std 0.01, so vary slightly around the mean in a way that mean matches
    # and then we will set summary std from the repeats in the fixture builder.
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "architecture": arch,
                "representation": rep,
                "split": "validation",
                "repeat": i,
                "patient_macro_exact_rate": exact,
                "patient_macro_omae": omae,
                "patient_exact_profile_rate": 0.5,
                "patient_adjacent_only_rate": 0.4,
                "patient_severe_error_rate": 0.1,
            }
        )
    return rows


def _eight_configs(*, n_3d_conv_eff: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    repeat_rows: list[dict] = []
    for arch in ("ViT", "MaxViT", "ConvNeXt", "EfficientNet"):
        for rep in ("2D", "2.5D-to-3D"):
            n = n_3d_conv_eff if rep == "2.5D-to-3D" and arch in {"ConvNeXt", "EfficientNet"} else 5
            exact = 0.84 if rep == "2D" else 0.78
            omae = 0.16 if rep == "2D" else 0.24
            summary_rows.append(_summary_row(arch, rep, n=n, exact=exact, omae=omae))
            # overwrite std to 0 to match constant repeats
            summary_rows[-1]["patient_macro_exact_rate_std"] = 0.0
            summary_rows[-1]["patient_macro_omae_std"] = 0.0
            summary_rows[-1]["patient_exact_profile_rate_std"] = 0.0
            summary_rows[-1]["patient_adjacent_only_rate_std"] = 0.0
            summary_rows[-1]["patient_severe_error_rate_std"] = 0.0
            repeat_rows.extend(_repeat_rows(arch, rep, n=n, exact=exact, omae=omae))
    return pd.DataFrame(summary_rows), pd.DataFrame(repeat_rows)


def test_canonicalize_and_consistency(tmp_path: Path):
    summary, repeats = _eight_configs()
    summary_path = tmp_path / "summary.csv"
    repeat_path = tmp_path / "repeats.csv"
    summary.to_csv(summary_path, index=False)
    repeats.to_csv(repeat_path, index=False)
    loaded_summary, loaded_repeats = load_data(summary_path, repeat_path)
    validate_repeat_summary_consistency(loaded_summary, loaded_repeats)
    validate_profile_sums(loaded_summary)
    validation = validate_plot_scope(loaded_summary)
    assert len(validation) == 8


def test_mismatch_stops(tmp_path: Path):
    summary, repeats = _eight_configs()
    summary.loc[0, "patient_macro_exact_rate_mean"] = 0.11
    summary_path = tmp_path / "summary.csv"
    repeat_path = tmp_path / "repeats.csv"
    summary.to_csv(summary_path, index=False)
    repeats.to_csv(repeat_path, index=False)
    loaded_summary, loaded_repeats = load_data(summary_path, repeat_path)
    with pytest.raises(ValueError, match="STOP plotting"):
        validate_repeat_summary_consistency(loaded_summary, loaded_repeats)


def test_profile_sum_must_be_one():
    summary, _ = _eight_configs()
    summary.loc[0, "patient_severe_error_rate_mean"] = 0.5
    loaded = summary.rename(
        columns={
            "architecture": "arch",
            "representation": "rep",
            "patient_macro_exact_rate_mean": "exact_mean",
            "patient_macro_omae_mean": "omae_mean",
            "patient_exact_profile_rate_mean": "r0_mean",
            "patient_adjacent_only_rate_mean": "r1_mean",
            "patient_severe_error_rate_mean": "r2_mean",
        }
    )
    with pytest.raises(ValueError, match="R0\\+R1\\+R2"):
        validate_profile_sums(loaded)


def test_warns_when_repeat_count_differs_from_guide():
    summary, _ = _eight_configs(n_3d_conv_eff=5)
    loaded = summary.rename(
        columns={
            "architecture": "arch",
            "representation": "rep",
        }
    )
    warnings = warn_repeat_counts(loaded)
    assert any("ConvNeXt" in w and "2.5D-to-3D" in w for w in warnings)


def test_generate_writes_png_pdf_and_audit(tmp_path: Path):
    summary, repeats = _eight_configs()
    summary_path = tmp_path / "summary.csv"
    repeat_path = tmp_path / "repeats.csv"
    out = tmp_path / "figures"
    summary.to_csv(summary_path, index=False)
    repeats.to_csv(repeat_path, index=False)
    paths = generate_patient_level_figures(
        summary_path=summary_path,
        repeat_path=repeat_path,
        output_dir=out,
    )
    assert paths["profile_png"].is_file()
    assert paths["tradeoff_png"].is_file()
    assert paths["burden_png"].is_file()
    assert paths["profile_pdf"].is_file()
    source = pd.read_csv(paths["plot_source"])
    assert len(source) == 8
    meta = paths["profile_meta"].read_text(encoding="utf-8")
    assert "generated_from_real_results" in meta
