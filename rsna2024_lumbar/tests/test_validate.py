from pathlib import Path

import pytest

from rsna2024_lumbar.data.validate import main, parse_args, validate_raw


def test_validate_cli_default_is_data_raw():
    args = parse_args([])
    assert args.data_root.name == "raw"


def test_validate_committed_fixtures_ok(mini_raw):
    report = validate_raw(mini_raw)
    assert report.ok
    assert report.n_studies == 3
    assert report.n_dicoms == 14
    assert report.n_coord_files_missing == 1  # study 1002 last instance
    assert report.n_coord_files_ok == 14
    assert any("missing DICOM" in w for w in report.warnings)


def test_validate_strict_fails_incomplete_fixture(mini_raw):
    report = validate_raw(mini_raw, strict=True)
    assert report.ok is False


def test_validate_missing_root_fails(tmp_path):
    report = validate_raw(tmp_path / "absent")
    assert report.ok is False
    assert any("does not exist" in e for e in report.errors)


def test_validate_incomplete_layout_fails(tmp_path):
    report = validate_raw(tmp_path)
    assert report.ok is False
    assert any("missing file" in e for e in report.errors)


def test_validate_cli_on_fixtures_exits_ok(mini_raw, capsys):
    main(["--data-root", str(mini_raw)])
    out = capsys.readouterr().out
    assert "OK" in out
    assert "studies:" in out


def test_validate_cli_rejects_empty(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--data-root", str(tmp_path)])
    assert exc.value.code == 1
