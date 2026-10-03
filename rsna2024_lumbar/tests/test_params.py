"""Positive CLI/library params + guards for bad ones. Uses committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna2024_lumbar.preprocessing.catalog import iter_crop_jobs, validate_export_request
from rsna2024_lumbar.preprocessing.cli import main, parse_args
from rsna2024_lumbar.preprocessing.constants import (
    CROP_POLICY_CENTERED,
    CROP_POLICY_EXTEND50,
)
from rsna2024_lumbar.preprocessing.crops import crop_settings_for
from rsna2024_lumbar.preprocessing.export import export_crops


def test_cli_defaults():
    args = parse_args([])
    assert args.crop_policy == CROP_POLICY_CENTERED
    assert args.condition is None
    assert args.max_studies is None
    assert args.skip_existing is False


def test_cli_positive_flags():
    args = parse_args(
        [
            "--crop-policy",
            "extend50",
            "--condition",
            "spinal_canal_stenosis",
            "--max-studies",
            "1",
            "--skip-existing",
            "--data-root",
            "rsna2024_lumbar/tests/fixtures",
        ]
    )
    assert args.crop_policy == CROP_POLICY_EXTEND50
    assert args.condition == "spinal_canal_stenosis"
    assert args.max_studies == 1
    assert args.skip_existing is True


def test_cli_rejects_bad_policy():
    with pytest.raises(SystemExit):
        parse_args(["--crop-policy", "wide"])


def test_cli_rejects_bad_condition():
    with pytest.raises(SystemExit):
        parse_args(["--condition", "not_a_condition"])


def test_cli_rejects_max_studies_below_one(mini_raw, tmp_path):
    with pytest.raises(SystemExit, match="max-studies"):
        main(
            [
                "--data-root",
                str(mini_raw),
                "--output-root",
                str(tmp_path / "out"),
                "--max-studies",
                "0",
                "--condition",
                "spinal_canal_stenosis",
            ]
        )


def test_library_rejects_bad_policy(mini_raw):
    with pytest.raises(ValueError, match="crop policy"):
        validate_export_request(mini_raw, crop_policy="wide")


def test_library_rejects_bad_condition(mini_raw):
    with pytest.raises(KeyError, match="Unknown condition"):
        iter_crop_jobs(
            mini_raw,
            crop_policy=CROP_POLICY_CENTERED,
            conditions=["not_a_condition"],
        )


def test_library_rejects_missing_data_root(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="data-root"):
        validate_export_request(missing, crop_policy=CROP_POLICY_CENTERED)


def test_library_rejects_incomplete_layout(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Missing"):
        validate_export_request(tmp_path, crop_policy=CROP_POLICY_CENTERED)


def test_crop_settings_reject_bad_condition():
    with pytest.raises(KeyError, match="Unknown condition"):
        crop_settings_for("nope", CROP_POLICY_CENTERED)


def test_positive_centered_export_via_cli(mini_raw, tmp_path, capsys):
    out = tmp_path / "centered"
    main(
        [
            "--data-root",
            str(mini_raw),
            "--output-root",
            str(out),
            "--crop-policy",
            "centered",
            "--condition",
            "spinal_canal_stenosis",
        ]
    )
    captured = capsys.readouterr().out
    assert "Exported: 10" in captured
    assert (out / "manifest.csv").is_file()


def test_positive_extend50_and_max_studies(mini_raw, tmp_path):
    from PIL import Image

    out = tmp_path / "extend50"
    result = export_crops(
        mini_raw,
        out,
        crop_policy=CROP_POLICY_EXTEND50,
        conditions=["spinal_canal_stenosis"],
        max_studies=1,
    )
    assert result["exported"] == 5
    png = out / "spinal_canal_stenosis" / "Normal_Mild" / "study_1001_l1_l2.png"
    assert Image.open(png).size == (96, 64)


def test_committed_fixture_files_exist():
    root = Path(__file__).resolve().parent / "fixtures"
    assert (root / "train.csv").is_file()
    assert (root / "train_label_coordinates.csv").is_file()
    assert (root / "train_series_descriptions.csv").is_file()
    dcms = list((root / "train_images").rglob("*.dcm"))
    assert len(dcms) == 14
