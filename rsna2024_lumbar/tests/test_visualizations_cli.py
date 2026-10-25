from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.visualizations.cli import main


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.parametrize(
    ("argv", "expected_output"),
    [
        (["dataset-scale"], "output_dir"),
        (["dicom-reuse"], "output_dir"),
        (["dicom-resolutions"], "output_dir"),
        (["dataset-split", "--train-fraction", "0.6"], "output_dir"),
    ],
)
def test_cli_dispatches_dataset_description_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    expected_output: str,
):
    called: dict[str, object] = {}

    def capture(**kwargs):
        called.update(kwargs)

    command = argv[0]
    targets = {
        "dataset-scale": "rsna2024_lumbar.visualizations.cli.generate_dataset_scale_figures",
        "dicom-reuse": "rsna2024_lumbar.visualizations.cli.generate_dicom_reuse_figures",
        "dicom-resolutions": "rsna2024_lumbar.visualizations.cli.generate_dicom_resolution_figures",
        "dataset-split": "rsna2024_lumbar.visualizations.cli.generate_dataset_split_figures",
    }
    monkeypatch.setattr(targets[command], capture)
    assert main([*argv, "--output-dir", str(tmp_path)]) == 0
    assert called[expected_output] == tmp_path
    if command == "dataset-split":
        assert called["train_fraction"] == 0.6
        assert called["val_fraction"] == 0.15
        assert called["test_fraction"] == 0.15
