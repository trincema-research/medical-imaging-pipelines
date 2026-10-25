from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dicom_resolutions import (
    DEFAULT_COUNTS,
    _resolution_sort_key,
    generate_dicom_resolution_figures,
    load_resolution_counts,
    top_resolution_labels,
)


def test_resolution_sort_key_largest_area_first():
    assert _resolution_sort_key("768x768") < _resolution_sort_key("512x512")
    assert _resolution_sort_key("640×640") == _resolution_sort_key("640x640")


def test_top_resolution_labels_sorts_by_area():
    counts = pd.DataFrame(
        {
            "condition": ["spinal_canal_stenosis"] * 3,
            "resolution": ["320x320", "768x768", "512x512"],
            "count": [100, 10, 50],
        }
    )
    labels = top_resolution_labels(counts, top_n=3)
    assert labels == ["768x768", "512x512", "320x320"]


def test_load_resolution_counts_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "resolutions.csv"
    pd.DataFrame({"condition": ["spinal_canal_stenosis"], "resolution": ["512x512"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_resolution_counts(path)


def test_committed_resolution_counts_load():
    counts = load_resolution_counts(DEFAULT_COUNTS)
    assert set(counts["condition"]) == {
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    }
    assert int(counts.loc[counts["condition"] == "spinal_canal_stenosis", "count"].sum()) == 33569
    labels = top_resolution_labels(counts)
    assert "512x512" in labels
    assert labels[0] == sorted(labels, key=lambda s: -int(s.split("x")[0]) * int(s.split("x")[1]))[0]


def test_generate_writes_resolution_figure(tmp_path: Path):
    counts = pd.read_csv(DEFAULT_COUNTS)
    path = tmp_path / "dicom_resolution_counts_by_condition.csv"
    counts.to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_dicom_resolution_figures(counts_path=path, output_dir=out)
    assert paths["resolutions_png"].is_file()
    assert paths["resolutions_pdf"].is_file()
    assert paths["source_counts"].is_file()
    assert paths["meta"].is_file()


@pytest.mark.skipif(not DEFAULT_COUNTS.is_file(), reason="resolution inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dicom_resolution_figures(output_dir=tmp_path)
    assert paths["resolutions_png"].is_file()
