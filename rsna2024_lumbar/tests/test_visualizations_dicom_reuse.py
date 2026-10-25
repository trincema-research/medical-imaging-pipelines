from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dicom_reuse import (
    DEFAULT_DUPLICATION,
    DEFAULT_PAIRWISE,
    DEFAULT_SUMMARY,
    generate_dicom_reuse_figures,
    load_duplication,
    load_overlap,
)


def test_committed_duplication_loads():
    results = load_duplication(DEFAULT_DUPLICATION)
    overlap = load_overlap(DEFAULT_SUMMARY, DEFAULT_PAIRWISE)
    assert results["spinal_canal_stenosis"]["n_unique_dicoms"] == 2522
    assert results["left_subarticular_stenosis"]["n_coord_rows"] == 9608
    assert overlap["global_unique_dicoms"] == 24546
    assert overlap["pairwise_overlap"]["LSS&RSS"] == 5478


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


@pytest.mark.skipif(not DEFAULT_DUPLICATION.is_file(), reason="dicom reuse inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dicom_reuse_figures(output_dir=tmp_path)
    assert paths["reuse_png"].is_file()
