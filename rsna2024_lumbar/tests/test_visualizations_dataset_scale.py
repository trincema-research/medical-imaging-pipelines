from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dataset_scale import (
    DEFAULT_COUNTS,
    DEFAULT_SCALE,
    format_count,
    generate_dataset_scale_figures,
    load_condition_severity,
    load_scale,
)


def test_format_count_groups_five_or_more_digits_only():
    assert format_count(147218) == "147,218"
    assert format_count(23597) == "23,597"
    assert format_count(10000) == "10,000"
    assert format_count(9999) == "9999"
    assert format_count(8335) == "8335"
    assert format_count(1792) == "1792"
    assert format_count(705) == "705"


def test_committed_inventory_loads():
    scale = load_scale(DEFAULT_SCALE)
    counts = load_condition_severity(DEFAULT_COUNTS)
    assert scale["raw_dicom_files"] == 147218
    assert len(counts) == 5


def test_generate_writes_combo(tmp_path: Path):
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
    scale_path = tmp_path / "dataset_scale.csv"
    counts_path = tmp_path / "counts_condition_x_severity.csv"
    scale.to_csv(scale_path, index=False)
    counts.to_csv(counts_path, index=False)
    out = tmp_path / "figures"
    paths = generate_dataset_scale_figures(scale_path=scale_path, counts_path=counts_path, output_dir=out)
    assert paths["combo_png"].is_file()
    assert paths["combo_pdf"].is_file()


@pytest.mark.skipif(not DEFAULT_SCALE.is_file(), reason="dataset scale inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dataset_scale_figures(output_dir=tmp_path)
    assert paths["combo_png"].is_file()
