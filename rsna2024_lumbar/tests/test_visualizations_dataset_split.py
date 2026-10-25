from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dataset_scale import format_count
from rsna2024_lumbar.visualizations.plot_dataset_split import (
    DEFAULT_FRACTIONS,
    DEFAULT_SPLITS,
    generate_dataset_split_figures,
    load_split_summaries,
)


def test_committed_splits_load():
    summaries = load_split_summaries(DEFAULT_SPLITS)
    assert len(summaries) == 5
    scs = summaries[0]
    assert scs["condition_key"] == "spinal_canal_stenosis"
    assert scs["dicom_crops_train"] == 6635
    assert scs["studies_usable_val"] == 286
    assert scs["train_fraction_target"] == pytest.approx(DEFAULT_FRACTIONS["train"])


def test_split_table_labels_follow_format_count():
    summaries = load_split_summaries(DEFAULT_SPLITS)
    scs = summaries[0]
    assert format_count(scs["dicom_crops_train"]) == "6635"
    assert format_count(scs["studies_usable_train"]) == "1327"
    assert f"{format_count(19050)} crops" == "19,050 crops"


def test_load_split_summaries_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "splits.csv"
    pd.DataFrame({"condition": ["spinal_canal_stenosis"], "split": ["train"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_split_summaries(path)


def test_load_split_summaries_rejects_unexpected_condition(tmp_path: Path):
    path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "condition": ["not_a_condition"],
            "split": ["train"],
            "studies_usable": [1],
            "dicom_crops": [1],
            "fraction_actual": [1.0],
            "fraction_target": [0.7],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected conditions"):
        load_split_summaries(path)


def test_generate_writes_split_figure(tmp_path: Path):
    splits = pd.read_csv(DEFAULT_SPLITS)
    path = tmp_path / "dataset_split_all_conditions.csv"
    splits.to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_dataset_split_figures(splits_path=path, output_dir=out)
    assert paths["split_png"].is_file()
    assert paths["split_pdf"].is_file()
    assert paths["source_splits"].is_file()
    assert paths["meta"].is_file()


@pytest.mark.skipif(not DEFAULT_SPLITS.is_file(), reason="split inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dataset_split_figures(output_dir=tmp_path)
    assert paths["split_png"].is_file()
