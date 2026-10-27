from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dicom_reuse import (
    DEFAULT_DUPLICATION,
    DEFAULT_PAIRWISE,
    DEFAULT_SUMMARY,
    MATH_COL_WIDTHS,
    MAX_SHOWN_MULT,
    _bin_multiplicity,
    generate_dicom_reuse_figures,
    load_duplication,
    load_overlap,
    reconciliation_table_rows,
)


def test_bin_multiplicity_keeps_exact_max_when_no_higher():
    labels, values = _bin_multiplicity({1: 10, 5: 3})
    assert labels == ["1", "2", "3", "4", "5"]
    assert values == [10, 0, 0, 0, 3]


def test_bin_multiplicity_collapses_above_max():
    labels, values = _bin_multiplicity({1: 10, MAX_SHOWN_MULT: 3, MAX_SHOWN_MULT + 2: 2})
    assert labels[-1] == f"{MAX_SHOWN_MULT}+"
    assert values[-1] == 5


def test_committed_duplication_loads():
    results = load_duplication(DEFAULT_DUPLICATION)
    overlap = load_overlap(DEFAULT_SUMMARY, DEFAULT_PAIRWISE)
    assert results["spinal_canal_stenosis"]["n_unique_dicoms"] == 2522
    assert results["left_subarticular_stenosis"]["n_coord_rows"] == 9608
    assert overlap["global_unique_dicoms"] == 24546
    assert overlap["pairwise_overlap"]["LSS&RSS"] == 5478
    assert overlap["check_sum_minus_extras_equals_global"] is True


def test_reconciliation_meaning_wraps_and_value_column_is_narrow():
    overlap = load_overlap(DEFAULT_SUMMARY, DEFAULT_PAIRWISE)
    rows = reconciliation_table_rows(overlap)
    assert "\n" in rows[0][2]
    assert "double-counts shared files" in rows[0][2]
    assert "shared fil" not in rows[0][2].split("\n")[0]
    assert MATH_COL_WIDTHS[1] <= 0.15
    assert MATH_COL_WIDTHS[1] < MATH_COL_WIDTHS[0]
    assert MATH_COL_WIDTHS[1] < MATH_COL_WIDTHS[2]


def test_load_duplication_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "duplication.csv"
    pd.DataFrame({"condition": ["spinal_canal_stenosis"], "multiplicity": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_duplication(path)


def test_load_overlap_rejects_bad_summary(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    pairwise = tmp_path / "pairwise.csv"
    pd.DataFrame({"metric": ["global_unique_dicoms"]}).to_csv(summary, index=False)
    pd.DataFrame({"condition": ["spinal_canal_stenosis"]}).to_csv(pairwise, index=False)
    with pytest.raises(ValueError, match="metric,value"):
        load_overlap(summary, pairwise)


def test_generate_writes_reuse_figure(tmp_path: Path):
    duplication = pd.read_csv(DEFAULT_DUPLICATION)
    summary = pd.read_csv(DEFAULT_SUMMARY)
    pairwise = pd.read_csv(DEFAULT_PAIRWISE)
    dup_path = tmp_path / "dicom_duplication_by_condition.csv"
    sum_path = tmp_path / "dicom_cross_condition_summary.csv"
    pair_path = tmp_path / "dicom_cross_condition_pairwise.csv"
    duplication.to_csv(dup_path, index=False)
    summary.to_csv(sum_path, index=False)
    pairwise.to_csv(pair_path, index=False)
    out = tmp_path / "figures"
    paths = generate_dicom_reuse_figures(
        duplication_path=dup_path,
        summary_path=sum_path,
        pairwise_path=pair_path,
        output_dir=out,
    )
    assert paths["reuse_png"].is_file()
    assert paths["reuse_pdf"].is_file()
    assert paths["source_duplication"].is_file()
    assert paths["meta"].is_file()


@pytest.mark.skipif(not DEFAULT_DUPLICATION.is_file(), reason="dicom reuse inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dicom_reuse_figures(output_dir=tmp_path)
    assert paths["reuse_png"].is_file()
