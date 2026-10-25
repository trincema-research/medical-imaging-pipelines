from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsna2024_lumbar.visualizations.plot_dataset_split import (
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


def test_generate_writes_split_figure(tmp_path: Path):
    splits = pd.read_csv(DEFAULT_SPLITS)
    path = tmp_path / "dataset_split_all_conditions.csv"
    splits.to_csv(path, index=False)
    out = tmp_path / "figures"
    paths = generate_dataset_split_figures(splits_path=path, output_dir=out)
    assert paths["split_png"].is_file()
    assert paths["split_pdf"].is_file()


@pytest.mark.skipif(not DEFAULT_SPLITS.is_file(), reason="split inventory not committed")
def test_real_inventory_generates(tmp_path: Path):
    paths = generate_dataset_split_figures(output_dir=tmp_path)
    assert paths["split_png"].is_file()
