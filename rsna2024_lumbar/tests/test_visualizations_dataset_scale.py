from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dataset_scale import (
    DEFAULT_COUNTS,
    DEFAULT_SCALE,
    _plot_file_hierarchy_ax,
    _plot_severity_mix_ax,
    condition_severity_matrix,
    format_count,
    generate_dataset_scale_figures,
    load_condition_severity,
    load_scale,
)


def _tiny_scale_and_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    scale = pd.DataFrame(
        {
            "metric": [
                "raw_dicom_files",
                "unique_dicom_files_total",
                "usable_dicom_crops_total",
                "label_cells_total",
            ],
            "count": [147218, 23597, 47190, 49375],
        }
    )
    counts = pd.DataFrame(
        {
            "condition": [
                "spinal_canal_stenosis",
                "left_neural_foraminal_narrowing",
                "right_neural_foraminal_narrowing",
                "left_subarticular_stenosis",
                "right_subarticular_stenosis",
            ],
            "Normal/Mild": [8335, 7671, 7681, 6515, 6507],
            "Moderate": [705, 1792, 1766, 1660, 1690],
            "Severe": [455, 397, 378, 815, 823],
        }
    )
    return scale, counts


def test_format_count_groups_five_or_more_digits_only():
    assert format_count(147218) == "147,218"
    assert format_count(23597) == "23,597"
    assert format_count(10000) == "10,000"
    assert format_count(9999) == "9999"
    assert format_count(8335) == "8335"
    assert format_count(1792) == "1792"
    assert format_count(705) == "705"
    assert format_count(0) == "0"
    assert format_count(-147218) == "-147,218"


def test_committed_inventory_loads():
    scale = load_scale(DEFAULT_SCALE)
    counts = load_condition_severity(DEFAULT_COUNTS)
    assert scale["raw_dicom_files"] == 147218
    assert len(counts) == 5


def test_load_scale_rejects_missing_metric(tmp_path: Path):
    path = tmp_path / "dataset_scale.csv"
    pd.DataFrame({"metric": ["raw_dicom_files"], "count": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing metrics"):
        load_scale(path)


def test_load_condition_severity_rejects_bad_columns(tmp_path: Path):
    path = tmp_path / "counts.csv"
    pd.DataFrame({"condition": ["spinal_canal_stenosis"], "Normal/Mild": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_condition_severity(path)


def test_load_condition_severity_rejects_unexpected_condition(tmp_path: Path):
    path = tmp_path / "counts.csv"
    pd.DataFrame(
        {
            "condition": ["not_a_condition"],
            "Normal/Mild": [1],
            "Moderate": [0],
            "Severe": [0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected conditions"):
        load_condition_severity(path)


def test_condition_severity_matrix_matches_inventory_row():
    counts_df = load_condition_severity(DEFAULT_COUNTS)
    counts, row_pct = condition_severity_matrix(counts_df)
    assert counts[0, 0] == 8335
    assert counts[0].sum() == 9495
    assert row_pct[0].sum() == pytest.approx(100.0)


def test_scale_bar_labels_use_format_count():
    fig, ax = plt.subplots()
    try:
        _plot_file_hierarchy_ax(
            ax,
            {
                "raw_dicom_files": 147218,
                "unique_dicom_files_total": 23597,
                "usable_dicom_crops_total": 47190,
                "label_cells_total": 49375,
            },
        )
        texts = [text.get_text() for text in ax.texts]
        assert "147,218" in texts
        assert "23,597" in texts
        assert "47,190" in texts
        assert "49,375" in texts
    finally:
        plt.close(fig)


def test_severity_mix_labels_follow_format_count():
    counts_df = pd.DataFrame(
        {
            "condition": [
                "spinal_canal_stenosis",
                "left_neural_foraminal_narrowing",
                "right_neural_foraminal_narrowing",
                "left_subarticular_stenosis",
                "right_subarticular_stenosis",
            ],
            "Normal/Mild": [12000, 7671, 7681, 6515, 6507],
            "Moderate": [705, 1792, 1766, 1660, 1690],
            "Severe": [455, 397, 378, 815, 823],
        }
    )
    counts, _ = condition_severity_matrix(counts_df)
    fig, ax = plt.subplots()
    try:
        _plot_severity_mix_ax(ax, counts)
        texts = [text.get_text() for text in ax.texts]
        assert "12,000" in texts
        assert "7671" in texts
        assert "705" in texts
        assert "8,335" not in texts
        assert "7,671" not in texts
    finally:
        plt.close(fig)


def test_generate_writes_combo(tmp_path: Path):
    scale, counts = _tiny_scale_and_counts()
    scale_path = tmp_path / "dataset_scale.csv"
    counts_path = tmp_path / "counts_condition_x_severity.csv"
    scale.to_csv(scale_path, index=False)
    counts.to_csv(counts_path, index=False)
    out = tmp_path / "figures"
    paths = generate_dataset_scale_figures(scale_path=scale_path, counts_path=counts_path, output_dir=out)
    assert paths["combo_png"].is_file()
    assert paths["combo_pdf"].is_file()
    assert paths["source_scale"].is_file()
    assert paths["source_counts"].is_file()
    assert paths["meta"].is_file()


@pytest.mark.skipif(not DEFAULT_SCALE.is_file(), reason="dataset scale inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dataset_scale_figures(output_dir=tmp_path)
    assert paths["combo_png"].is_file()
